import torch
import torch.nn as nn
from ..layers.fourier_features import FourierFeatures
from ..layers.feed_forward import FeedForward

class NmrEmbedding(nn.Module):
    def __init__(
        self,
        d_model=512,
        dropout=0.1,
        strategy='lin_float_int',     # 默认策略：lin_float_int
        h_x_min=0.01, h_x_max=16.0,
        c_x_min=0.01, c_x_max=230.0,
        num_freqs_per_type=None,      # 若为 None，则自动计算
        funcs='both',                 # 'both' | 'sin' | 'cos'
        trainable=False,
        sigma=10,
        use_identity_embedding=False,  # 是否使用identity embedding（负值直接进Fourier）
        use_count_embedding=True,      # 是否使用count embedding
    ):
        super().__init__()
        self.d_model = d_model
        self.use_identity_embedding = use_identity_embedding
        self.use_count_embedding = use_count_embedding

        # ====== 创建 Fourier 特征生成器 ======
        self.h_fourier = FourierFeatures(
            strategy=strategy, x_min=h_x_min, x_max=h_x_max,
            trainable=trainable, funcs=funcs, sigma=sigma,
            num_freqs=num_freqs_per_type or max(8, d_model // 4)
        )
        self.c_fourier = FourierFeatures(
            strategy=strategy, x_min=c_x_min, x_max=c_x_max,
            trainable=trainable, funcs=funcs, sigma=sigma,
            num_freqs=num_freqs_per_type or max(8, d_model // 4)
        )

        # 注意：此时两个 Fourier 输出维度可能不同！
        h_out_dim = self.h_fourier.num_features()
        c_out_dim = self.c_fourier.num_features()

        # ====== 分别为 H 和 C 建立 MLP 对齐到 d_model ======
        self.h_shift_ffn = FeedForward(in_dim=h_out_dim, out_dim=d_model, depth=5)
        self.c_shift_ffn = FeedForward(in_dim=c_out_dim, out_dim=d_model, depth=5)

        # H 原子还包含"计数"特征（可选）
        if use_count_embedding:
            self.h_count_ffn = FeedForward(in_dim=1, out_dim=d_model, depth=1)

        self.layer_norm = nn.LayerNorm(d_model, eps=1e-5)
        self.dropout = nn.Dropout(dropout)


    def forward(self, shifts, counts, types):
        """
        shifts: [B, L]   NMR位移值（可能包含负值表示masked identity: -1, -2, -3...）
        counts: [B, L]   峰面积/原子数量等
        types:  [B, L]   0=H, 1=C

        注意：当use_identity_embedding=True时，masked位置的shifts为负值（-1, -2, -3...）
        这些负值会直接输入到Fourier Features中进行编码
        """
        B, L = shifts.shape

        # ====== 直接使用shifts（包含负值），输入到Fourier Features ======
        shifts_unsqueezed = shifts.unsqueeze(-1)  # [B, L, 1]

        # ====== 各自的 Fourier 编码（可以处理负值） ======
        h_fourier_out = self.h_fourier(shifts_unsqueezed)  # [B, L, F_h]
        c_fourier_out = self.c_fourier(shifts_unsqueezed)  # [B, L, F_c]

        # ====== 各自的 MLP 映射到相同维度 ======
        h_embed = self.h_shift_ffn(h_fourier_out)  # [B, L, d_model]
        c_embed = self.c_shift_ffn(c_fourier_out)  # [B, L, d_model]

        # ====== 根据类型选择对应嵌入 ======
        is_h_mask = (types == 0).unsqueeze(-1)  # [B, L, 1]
        shift_embedding = torch.where(is_h_mask, h_embed, c_embed)  # [B, L, d_model]

        # ====== 加上 count 特征（仅 H 有效，且仅在启用时） ======
        if self.use_count_embedding:
            count_embedding = self.h_count_ffn(counts.unsqueeze(-1))     # [B, L, d_model]
            count_embedding = count_embedding * is_h_mask.float()        # 仅对 H 保留
            out = self.layer_norm(shift_embedding + count_embedding)
        else:
            out = self.layer_norm(shift_embedding)

        return self.dropout(out)
