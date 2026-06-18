import torch
import torch.nn as nn
from .embedding import NmrEmbedding
from ..layers.feed_forward import FeedForward
from ..utils.scheduler import NoamScheduler

class NMR_BERT(nn.Module):
    def __init__(self, d_model=768, nhead=12, num_encoder_layers=12, dim_feedforward=3072, dropout=0.1,
                 h_bin_size=0.01, h_max=16.0, c_bin_size=0.1, c_max=230.0,
                 fp_sim_bin_size=0.005, fp_sim_mlp_hidden=512,
                 use_identity_embedding=True, use_count_embedding=False,
                 lr=1e-4, weight_decay=0.0, n_warmup_steps=None):
        """
        参数:
            d_model: embedding维度
            nhead: 注意力头数
            num_encoder_layers: transformer层数
            dim_feedforward: FFN维度
            dropout: dropout率
            h_bin_size, h_max: H原子的bin参数
            c_bin_size, c_max: C原子的bin参数
            fp_sim_bin_size: 指纹相似度分bin的间隔大小
            fp_sim_mlp_hidden: 指纹相似度MLP分类器的隐藏层维度
            use_identity_embedding: 是否使用identity embedding
            use_count_embedding: 是否使用count embedding
            lr: 学习率
            weight_decay: 权重衰减
            n_warmup_steps: warmup步数，如果设置则使用Noam scheduler
        """
        super().__init__()

        # Store optimizer parameters
        self.lr = lr
        self.weight_decay = weight_decay
        self.n_warmup_steps = n_warmup_steps
        self.embedding = NmrEmbedding(
            d_model,
            dropout,
            use_identity_embedding=use_identity_embedding,
            use_count_embedding=use_count_embedding
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            activation='gelu',
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # 计算bin数量
        self.h_bin_size = h_bin_size
        self.c_bin_size = c_bin_size
        self.h_max = h_max
        self.c_max = c_max
        self.h_num_bins = int(h_max / h_bin_size) + 1
        self.c_num_bins = int(c_max / c_bin_size) + 1 

        # 两个分类头：一个用于H，一个用于C
        self.h_classification_head = nn.Linear(d_model, self.h_num_bins)
        self.c_classification_head = nn.Linear(d_model, self.c_num_bins)

        self.cso_out = nn.Linear(2 * d_model, 1) #chemical shift order

        # 指纹相似度MLP分类器
        # 输入: 2 * d_model (两个embedding concat)
        # 输出: fp_sim_num_bins (Tanimoto相似度的bin数)
        self.fp_sim_bin_size = fp_sim_bin_size
        self.fp_sim_num_bins = int(1.0 / fp_sim_bin_size) + 1  # 1.0/0.05 = 20 bins

        # 3层MLP分类器
        #self.fp_sim_classifier = nn.Linear(2 * d_model, self.fp_sim_num_bins)
        self.fp_sim_classifier = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, self.fp_sim_num_bins)
        )

    def forward(self, shifts, counts, types, padding_mask):
        x = self.embedding(shifts, counts, types)
        output = self.transformer_encoder(x, src_key_padding_mask=padding_mask)
        # 对H和C分别使用不同的分类头
        h_logits = self.h_classification_head(output)  # [B, L, h_num_bins]
        c_logits = self.c_classification_head(output)  # [B, L, c_num_bins]

        return h_logits, c_logits, output

    def configure_optimizers(self):
        """
        Configure optimizer and learning rate scheduler.

        Returns:
            - optimizer only (if no warmup specified)
            - [optimizer], [lr_scheduler] (if warmup is specified)
        """
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            eps=1e-6
        )

        if self.n_warmup_steps and self.n_warmup_steps > 0:
            lr_scheduler = {
                'scheduler': NoamScheduler(optimizer, self.n_warmup_steps),
                'interval': 'step',
                'frequency': 1,
            }
            return [optimizer], [lr_scheduler]

        return optimizer
