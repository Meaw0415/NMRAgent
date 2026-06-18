"""
NMR Classifier Model

Loads a pretrained NMR_BERT encoder and adds a classification head
for superclass prediction.
"""

import torch
import torch.nn as nn
from .nmr_bert import NMR_BERT


class ClassificationHead(nn.Module):
    """Classification head with linear or MLP decoder."""

    def __init__(self, d_model, num_classes, head_type='linear', hidden_dim=512, dropout=0.1):
        """
        Args:
            d_model: Input dimension from encoder
            num_classes: Number of output classes
            head_type: 'linear' or 'mlp'
            hidden_dim: Hidden dimension for MLP (only used if head_type='mlp')
            dropout: Dropout rate
        """
        super().__init__()
        self.head_type = head_type

        if head_type == 'linear':
            # Simple linear classifier
            self.classifier = nn.Linear(d_model, num_classes)
        elif head_type == 'mlp':
            # Two-layer MLP classifier
            self.classifier = nn.Sequential(
                nn.Linear(d_model, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes)
            )
        else:
            raise ValueError(f"Unknown head_type: {head_type}. Must be 'linear' or 'mlp'")

    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, d_model] or [batch_size, d_model]
        Returns:
            logits: [batch_size, num_classes]
        """
        return self.classifier(x)


class NMRClassifier(nn.Module):
    """NMR superclass classifier using pretrained encoder."""

    def __init__(
        self,
        encoder,
        num_classes,
        head_type='linear',
        hidden_dim=512,
        dropout=0.1,
        freeze_encoder=True,
        pooling='mean'
    ):
        """
        Args:
            encoder: Pretrained NMR_BERT encoder
            num_classes: Number of output classes
            head_type: 'linear' or 'mlp'
            hidden_dim: Hidden dimension for MLP head
            dropout: Dropout rate
            freeze_encoder: Whether to freeze encoder weights
            pooling: Pooling strategy - 'mean', 'max', or 'cls'
        """
        super().__init__()

        self.encoder = encoder
        self.freeze_encoder = freeze_encoder
        self.pooling = pooling

        # Freeze encoder if specified
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        # Classification head
        d_model = encoder.embedding.d_model
        self.classifier = ClassificationHead(
            d_model=d_model,
            num_classes=num_classes,
            head_type=head_type,
            hidden_dim=hidden_dim,
            dropout=dropout
        )

    def forward(self, shifts, counts, types, padding_mask=None):
        """
        Forward pass for classification.

        Args:
            shifts: [batch_size, seq_len] - raw chemical shifts
            counts: [batch_size, seq_len] - peak counts
            types: [batch_size, seq_len] - atom types (0=H, 1=C)
            padding_mask: [batch_size, seq_len] - padding mask (True for padding)

        Returns:
            logits: [batch_size, num_classes]
        """
        # Get encoder embeddings
        x = self.encoder.embedding(shifts, counts, types)  # [batch_size, seq_len, d_model]

        # Pass through transformer encoder
        encoder_output = self.encoder.transformer_encoder(
            x,
            src_key_padding_mask=padding_mask
        )  # [batch_size, seq_len, d_model]

        # Pool the sequence
        if self.pooling == 'mean':
            # Mean pooling (excluding padding)
            if padding_mask is not None:
                # Set padded positions to 0
                mask = ~padding_mask  # [batch_size, seq_len]
                mask = mask.unsqueeze(-1).float()  # [batch_size, seq_len, 1]
                pooled = (encoder_output * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            else:
                pooled = encoder_output.mean(dim=1)
        elif self.pooling == 'max':
            # Max pooling
            if padding_mask is not None:
                encoder_output = encoder_output.masked_fill(padding_mask.unsqueeze(-1), float('-inf'))
            pooled = encoder_output.max(dim=1)[0]
        elif self.pooling == 'cls':
            # Use first token as representation (like BERT [CLS])
            pooled = encoder_output[:, 0, :]
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling}")

        # Classify
        logits = self.classifier(pooled)  # [batch_size, num_classes]

        return logits

    def unfreeze_encoder(self):
        """Unfreeze encoder for fine-tuning."""
        for param in self.encoder.parameters():
            param.requires_grad = True
        self.freeze_encoder = False

    def freeze_encoder_layers(self):
        """Freeze encoder layers."""
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.freeze_encoder = True


def build_nmr_classifier(
    config,
    pretrained_encoder_path,
    num_classes,
    head_type='linear',
    hidden_dim=512,
    dropout=0.1,
    freeze_encoder=True,
    pooling='mean',
    encoder_config=None,
    device='cuda'
):
    """
    Build NMR classifier from pretrained encoder checkpoint.

    Args:
        pretrained_encoder_path: Path to pretrained encoder checkpoint
        num_classes: Number of output classes
        head_type: 'linear' or 'mlp'
        hidden_dim: Hidden dimension for MLP head
        dropout: Dropout rate
        freeze_encoder: Whether to freeze encoder weights
        pooling: Pooling strategy
        encoder_config: Config dict for encoder (if not loading from checkpoint)
        device: Device to load model on

    Returns:
        model: NMRClassifier model
    """
    encoder = NMR_BERT(
    d_model=config.get('d_model', 768),
    nhead=config.get('nhead', 12),
    num_encoder_layers=config.get('num_encoder_layers', 16),
    dim_feedforward=config.get('dim_feedforward', 3072),
    dropout=config.get('dropout', 0.1),
    h_bin_size=config.get('h_bin_size', 0.01),
    h_max=config.get('h_max', 16.0),
    c_bin_size=config.get('c_bin_size', 0.1),
    c_max=config.get('c_max', 230.0),
    use_identity_embedding=config.get('use_identity_embedding', True),
    use_count_embedding=config.get('use_count_embedding', False),)
    # Load pretrained encoder
    if pretrained_encoder_path:
        print(f"Loading pretrained encoder from: {pretrained_encoder_path}")
        checkpoint = torch.load(pretrained_encoder_path, map_location='cpu')
        encoder.load_state_dict(checkpoint['model_state_dict'])

    # Build classifier
    model = NMRClassifier(
        encoder=encoder,
        num_classes=num_classes,
        head_type=head_type,
        hidden_dim=hidden_dim,
        dropout=dropout,
        freeze_encoder=freeze_encoder,
        pooling=pooling
    )

    model = model.to(device)

    # Print model info
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model built:")
    print(f"  Total parameters: {num_params / 1e6:.2f}M")
    print(f"  Trainable parameters: {num_trainable / 1e6:.2f}M")
    print(f"  Encoder frozen: {freeze_encoder}")
    print(f"  Head type: {head_type}")
    print(f"  Pooling: {pooling}")

    return model
