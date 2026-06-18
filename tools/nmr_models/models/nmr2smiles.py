"""
NMR2SMILES Model

Encoder-Decoder architecture for NMR to SMILES generation.
- Encoder: Pretrained NMR-BERT
- Decoder: Transformer decoder for autoregressive SMILES generation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class SmilesDecoder(nn.Module):
    """
    Transformer Decoder for SMILES generation.

    Takes encoder output (NMR embeddings) and generates SMILES autoregressively.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 768,
        nhead: int = 12,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 3072,
        dropout: float = 0.1,
        max_seq_len: int = 512,
        pad_idx: int = 0,
        pretrained_embedding: Optional[torch.Tensor] = None,
        freeze_embedding: bool = False,
    ):
        """
        Args:
            vocab_size: Size of SMILES vocabulary
            d_model: Model dimension (should match encoder)
            nhead: Number of attention heads
            num_decoder_layers: Number of decoder layers
            dim_feedforward: FFN dimension
            dropout: Dropout rate
            max_seq_len: Maximum sequence length
            pad_idx: Padding token index
            pretrained_embedding: Pretrained embedding weights (vocab_size, d_model)
            freeze_embedding: Whether to freeze embedding weights
        """
        super().__init__()

        self.d_model = d_model
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx

        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)

        # Load pretrained embedding if provided
        if pretrained_embedding is not None:
            self.token_embedding.weight.data.copy_(pretrained_embedding)
            if freeze_embedding:
                self.token_embedding.weight.requires_grad = False

        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)

        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_decoder_layers,
        )

        # Output projection
        self.output_projection = nn.Linear(d_model, vocab_size)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.output_projection.weight, std=0.02)
        nn.init.zeros_(self.output_projection.bias)

    def generate_square_subsequent_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        """Generate causal mask for autoregressive decoding."""
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1).bool()
        return mask

    def forward(
        self,
        encoder_output: torch.Tensor,  # (B, src_len, d_model)
        encoder_padding_mask: torch.Tensor,  # (B, src_len)
        tgt_ids: torch.Tensor,  # (B, tgt_len)
        tgt_padding_mask: Optional[torch.Tensor] = None,  # (B, tgt_len)
    ) -> torch.Tensor:
        """
        Forward pass for training.

        Args:
            encoder_output: Encoder output from NMR-BERT
            encoder_padding_mask: Padding mask for encoder (True for padding)
            tgt_ids: Target SMILES token indices
            tgt_padding_mask: Padding mask for target (True for padding)

        Returns:
            logits: (B, tgt_len, vocab_size)
        """
        batch_size, tgt_len = tgt_ids.shape
        device = tgt_ids.device

        # Embed target tokens
        tgt_emb = self.token_embedding(tgt_ids) * math.sqrt(self.d_model)
        tgt_emb = self.pos_encoding(tgt_emb)

        # Generate causal mask
        tgt_mask = self.generate_square_subsequent_mask(tgt_len, device)

        # Decode
        decoder_output = self.transformer_decoder(
            tgt=tgt_emb,
            memory=encoder_output,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=encoder_padding_mask,
        )

        # Project to vocabulary
        logits = self.output_projection(decoder_output)

        return logits

    @torch.no_grad()
    def generate(
        self,
        encoder_output: torch.Tensor,
        encoder_padding_mask: torch.Tensor,
        bos_idx: int,
        eos_idx: int,
        max_len: int = 256,
        temperature: float = 0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Autoregressive generation.

        Args:
            encoder_output: (B, src_len, d_model)
            encoder_padding_mask: (B, src_len)
            bos_idx: BOS token index
            eos_idx: EOS token index
            max_len: Maximum generation length
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling threshold

        Returns:
            generated_ids: (B, gen_len)
        """
        batch_size = encoder_output.shape[0]
        device = encoder_output.device

        # Start with BOS token
        generated = torch.full((batch_size, 1), bos_idx, dtype=torch.long, device=device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(max_len - 1):
            # Forward pass
            logits = self.forward(
                encoder_output=encoder_output,
                encoder_padding_mask=encoder_padding_mask,
                tgt_ids=generated,
                tgt_padding_mask=None,
            )

            # Get logits for last position
            next_logits = logits[:, -1, :]

            # Apply temperature scaling (only when temperature > 0)
            if temperature > 0:
                next_logits = next_logits / temperature

            # Apply top-k filtering
            if top_k is not None and top_k > 0:
                indices_to_remove = next_logits < torch.topk(next_logits, top_k)[0][..., -1, None]
                next_logits[indices_to_remove] = float('-inf')

            # Apply top-p (nucleus) filtering
            if top_p is not None and top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_logits[indices_to_remove] = float('-inf')

            # Sample or argmax
            if temperature > 0:
                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = next_logits.argmax(dim=-1, keepdim=True)

            # Update finished status
            finished = finished | (next_token.squeeze(-1) == eos_idx)

            # Append to generated sequence
            generated = torch.cat([generated, next_token], dim=1)

            # Stop if all sequences are finished
            if finished.all():
                break

        return generated

    @torch.no_grad()
    def beam_search_generate(
        self,
        encoder_output: torch.Tensor,
        encoder_padding_mask: torch.Tensor,
        bos_idx: int,
        eos_idx: int,
        beam_size: int = 5,
        max_len: int = 256,
        length_penalty: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Beam search generation.

        Args:
            encoder_output: (B, src_len, d_model)
            encoder_padding_mask: (B, src_len)
            bos_idx: BOS token index
            eos_idx: EOS token index
            beam_size: Number of beams
            max_len: Maximum generation length
            length_penalty: Length penalty (alpha in Google NMT paper)
                           score = log_prob / (length ** length_penalty)

        Returns:
            sequences: (B, beam_size, max_len) - top beam_size sequences
            scores: (B, beam_size) - normalized scores for each sequence
        """
        batch_size = encoder_output.shape[0]
        device = encoder_output.device

        # Expand encoder outputs for beam search: (B, src_len, d_model) -> (B*beam_size, src_len, d_model)
        encoder_output_expanded = encoder_output.unsqueeze(1).repeat(1, beam_size, 1, 1).view(
            batch_size * beam_size, -1, self.d_model
        )
        encoder_padding_mask_expanded = encoder_padding_mask.unsqueeze(1).repeat(1, beam_size, 1).view(
            batch_size * beam_size, -1
        )

        # Initialize sequences with BOS token: (B, beam_size, 1)
        sequences = torch.full((batch_size, beam_size, 1), bos_idx, dtype=torch.long, device=device)
        # Initialize scores: (B, beam_size)
        beam_scores = torch.zeros((batch_size, beam_size), device=device)
        beam_scores[:, 1:] = float('-inf')  # Only first beam is active initially

        # Track which beams are finished and their final lengths
        finished = torch.zeros((batch_size, beam_size), dtype=torch.bool, device=device)
        # Store the length when each beam finished (for length penalty)
        finished_lengths = torch.zeros((batch_size, beam_size), device=device)
        # Store the score when each beam finished (to prevent score degradation)
        finished_scores = torch.zeros((batch_size, beam_size), device=device)

        for step in range(max_len - 1):
            # Skip computation if all beams are finished
            if finished.all():
                break

            # Reshape sequences for forward pass: (B, beam_size, seq_len) -> (B*beam_size, seq_len)
            current_sequences = sequences.view(batch_size * beam_size, -1)

            # Forward pass
            logits = self.forward(
                encoder_output=encoder_output_expanded,
                encoder_padding_mask=encoder_padding_mask_expanded,
                tgt_ids=current_sequences,
                tgt_padding_mask=None,
            )

            # Get logits for last position: (B*beam_size, vocab_size)
            next_logits = logits[:, -1, :]
            next_log_probs = F.log_softmax(next_logits, dim=-1)

            # Reshape to (B, beam_size, vocab_size)
            vocab_size = next_log_probs.shape[-1]
            next_log_probs = next_log_probs.view(batch_size, beam_size, -1)

            # For finished beams, we don't add new log_probs - keep their score fixed
            # Create candidate scores: for unfinished beams, add log_probs; for finished, keep old score
            candidate_scores = beam_scores.unsqueeze(-1) + next_log_probs

            # For finished beams: set all tokens to -inf except we'll handle them separately
            finished_mask = finished.unsqueeze(-1).expand(-1, -1, vocab_size)
            candidate_scores = candidate_scores.masked_fill(finished_mask, float('-inf'))

            # For finished beams, we need to keep them in the beam with their original score
            # We do this by setting a "dummy" slot with the finished score
            # Use EOS token position to carry the finished beam's score forward
            # But the score should NOT accumulate - use the stored finished_scores
            finished_beam_scores = finished_scores.unsqueeze(-1).expand(-1, -1, vocab_size)
            # Only set EOS position for finished beams to their finished score
            eos_scores_for_finished = torch.full_like(candidate_scores, float('-inf'))
            eos_scores_for_finished[:, :, eos_idx] = finished_scores
            candidate_scores = torch.where(finished_mask, eos_scores_for_finished, candidate_scores)

            # Flatten to (B, beam_size * vocab_size) and get top beam_size candidates
            candidate_scores_flat = candidate_scores.view(batch_size, -1)
            top_scores, top_indices = torch.topk(candidate_scores_flat, beam_size, dim=-1)

            # Compute which beam and which token each candidate came from
            beam_indices = top_indices // vocab_size  # (B, beam_size)
            token_indices = top_indices % vocab_size  # (B, beam_size)

            # Update sequences by selecting from beams and appending new tokens
            # Gather sequences from selected beams: (B, beam_size, seq_len)
            selected_sequences = torch.gather(
                sequences,
                1,
                beam_indices.unsqueeze(-1).expand(-1, -1, sequences.shape[-1])
            )
            # Append new tokens: (B, beam_size, seq_len+1)
            sequences = torch.cat([selected_sequences, token_indices.unsqueeze(-1)], dim=-1)

            # Update beam scores
            beam_scores = top_scores

            # Update finished status
            was_finished = torch.gather(finished, 1, beam_indices)
            newly_finished = (token_indices == eos_idx) & ~was_finished

            # Update finished_scores and finished_lengths for newly finished beams
            finished_scores = torch.gather(finished_scores, 1, beam_indices)
            finished_lengths = torch.gather(finished_lengths, 1, beam_indices)

            # For newly finished beams, record their score and length
            current_length = step + 2  # +1 for BOS, +1 for current token
            finished_scores = torch.where(newly_finished, beam_scores, finished_scores)
            finished_lengths = torch.where(
                newly_finished,
                torch.full_like(finished_lengths, current_length),
                finished_lengths
            )

            # Update finished status
            finished = was_finished | newly_finished

        # Compute final lengths for beams that never finished
        # For unfinished beams, length is the full sequence length
        final_lengths = torch.where(
            finished,
            finished_lengths,
            torch.full_like(finished_lengths, sequences.shape[-1], dtype=torch.float)
        )

        # Use finished_scores for finished beams, beam_scores for unfinished
        final_scores = torch.where(finished, finished_scores, beam_scores)

        # Apply length penalty
        final_lengths = torch.clamp(final_lengths, min=1.0)
        normalized_scores = final_scores / (final_lengths ** length_penalty)

        # Sort by normalized score
        sorted_scores, sorted_indices = torch.sort(normalized_scores, dim=-1, descending=True)
        sorted_sequences = torch.gather(
            sequences,
            1,
            sorted_indices.unsqueeze(-1).expand(-1, -1, sequences.shape[-1])
        )

        return sorted_sequences, sorted_scores


class FormulaEmbedding(nn.Module):
    """
    Formula embedding module - embeds molecular formula as a d-dimensional vector.

    Takes an m-dimensional vector where each dimension represents the count of a
    specific atom type (e.g., C, H, O, N, S, etc.), and projects it through a
    2-layer MLP to produce a d_model dimensional embedding.
    """

    def __init__(
        self,
        num_atom_types: int,
        d_model: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
    ):
        """
        Args:
            num_atom_types: Number of possible atom types (m dimension)
            d_model: Output embedding dimension
            hidden_dim: Hidden dimension in MLP (defaults to d_model)
            dropout: Dropout rate
        """
        super().__init__()

        self.num_atom_types = num_atom_types
        self.d_model = d_model
        hidden_dim = hidden_dim or d_model

        # 2-layer MLP: m -> hidden_dim -> d_model
        self.mlp = nn.Sequential(
            nn.Linear(num_atom_types, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                nn.init.zeros_(module.bias)

    def forward(self, formula_counts: torch.Tensor) -> torch.Tensor:
        """
        Args:
            formula_counts: (B, num_atom_types) - count of each atom type

        Returns:
            formula_emb: (B, 1, d_model) - formula embedding (ready to concat)
        """
        # (B, num_atom_types) -> (B, d_model)
        emb = self.mlp(formula_counts.float())
        # Add sequence dimension for concatenation: (B, d_model) -> (B, 1, d_model)
        return emb.unsqueeze(1)


class CountFusion(nn.Module):
    """
    Count信息融合模块 - 在encoder输出后、decoder输入前融合count信息。

    使用加法融合策略:
    encoder_output_fused = encoder_output + count_projection(counts)
    """

    def __init__(self, d_model: int, dropout: float = 0.1):
        """
        Args:
            d_model: 模型维度 (应与encoder输出维度匹配)
            dropout: Dropout比率
        """
        super().__init__()

        # Count投影: 将标量count值投影到d_model维度
        self.count_projection = nn.Sequential(
            nn.Linear(1, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, d_model),
        )

        # 融合后的LayerNorm
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        for module in self.count_projection:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        encoder_output: torch.Tensor,  # (B, L, d_model)
        counts: torch.Tensor,          # (B, L)
        types: torch.Tensor,           # (B, L) - 0=H, 1=C
    ) -> torch.Tensor:
        """
        融合count信息到encoder输出。

        Args:
            encoder_output: Encoder输出 (B, L, d_model)
            counts: Count值 (B, L)
            types: 类型标识 (B, L), 0=H, 1=C

        Returns:
            fused_output: 融合后的输出 (B, L, d_model)
        """
        # 投影count到d_model维度: (B, L) -> (B, L, 1) -> (B, L, d_model)
        count_emb = self.count_projection(counts.unsqueeze(-1))

        # 只对H原子 (type=0) 应用count融合，C原子的count通常为0
        h_mask = (types == 0).unsqueeze(-1).float()  # (B, L, 1)
        count_emb = count_emb * h_mask

        # 加法融合
        fused_output = encoder_output + count_emb

        # LayerNorm + Dropout
        fused_output = self.layer_norm(fused_output)
        fused_output = self.dropout(fused_output)

        return fused_output


class NMR2SmilesModel(nn.Module):
    """
    Full NMR2SMILES model combining pretrained encoder and decoder.
    """

    def __init__(
        self,
        encoder: nn.Module,
        decoder: SmilesDecoder,
        freeze_encoder: bool = False,
        use_count_fusion: bool = True,
        count_fusion_dropout: float = 0.1,
        use_formula_embedding: bool = False,
        num_atom_types: int = 50,
        formula_hidden_dim: Optional[int] = None,
    ):
        """
        Args:
            encoder: Pretrained NMR-BERT encoder
            decoder: SmilesDecoder instance
            freeze_encoder: Whether to freeze encoder weights
            use_count_fusion: 是否在encoder后、decoder前融合count信息
            count_fusion_dropout: Count融合模块的dropout比率
            use_formula_embedding: 是否在peak embedding前concat formula embedding
            num_atom_types: Number of atom types for formula embedding (default: 50)
            formula_hidden_dim: Hidden dimension for formula MLP (defaults to d_model)
        """
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.freeze_encoder = freeze_encoder
        self.use_count_fusion = use_count_fusion
        self.use_formula_embedding = use_formula_embedding

        d_model = decoder.d_model

        # Count融合模块
        if use_count_fusion:
            self.count_fusion = CountFusion(d_model, dropout=count_fusion_dropout)

        # Formula embedding模块
        if use_formula_embedding:
            self.formula_embedding = FormulaEmbedding(
                num_atom_types=num_atom_types,
                d_model=d_model,
                hidden_dim=formula_hidden_dim,
                dropout=count_fusion_dropout,
            )

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def forward(
        self,
        shifts: torch.Tensor,
        counts: torch.Tensor,
        types: torch.Tensor,
        nmr_padding_mask: torch.Tensor,
        tgt_ids: torch.Tensor,
        tgt_padding_mask: Optional[torch.Tensor] = None,
        formula_counts: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            shifts: NMR shifts (B, src_len)
            counts: NMR counts (B, src_len)
            types: Atom types (B, src_len)
            nmr_padding_mask: NMR padding mask (B, src_len)
            tgt_ids: Target SMILES ids (B, tgt_len)
            tgt_padding_mask: Target padding mask (B, tgt_len)
            formula_counts: Formula atom counts (B, num_atom_types), optional

        Returns:
            logits: (B, tgt_len, vocab_size)
        """
        # Encode NMR
        if self.freeze_encoder:
            with torch.no_grad():
                _, _, encoder_output = self.encoder(shifts, counts, types, nmr_padding_mask)
        else:
            _, _, encoder_output = self.encoder(shifts, counts, types, nmr_padding_mask)

        # 在encoder后、decoder前融合count信息
        if self.use_count_fusion:
            encoder_output = self.count_fusion(encoder_output, counts, types)

        # Concat formula embedding at the front of encoder output
        if self.use_formula_embedding and formula_counts is not None:
            formula_emb = self.formula_embedding(formula_counts)  # (B, 1, d_model)
            encoder_output = torch.cat([formula_emb, encoder_output], dim=1)  # (B, 1+src_len, d_model)
            # Extend padding mask for formula token (not padding)
            batch_size = nmr_padding_mask.shape[0]
            formula_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=nmr_padding_mask.device)
            nmr_padding_mask = torch.cat([formula_mask, nmr_padding_mask], dim=1)

        # Decode to SMILES
        logits = self.decoder(
            encoder_output=encoder_output,
            encoder_padding_mask=nmr_padding_mask,
            tgt_ids=tgt_ids,
            tgt_padding_mask=tgt_padding_mask,
        )

        return logits

    @torch.no_grad()
    def generate(
        self,
        shifts: torch.Tensor,
        counts: torch.Tensor,
        types: torch.Tensor,
        nmr_padding_mask: torch.Tensor,
        bos_idx: int,
        eos_idx: int,
        formula_counts: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Generate SMILES from NMR."""
        # Encode NMR
        _, _, encoder_output = self.encoder(shifts, counts, types, nmr_padding_mask)

        # 在encoder后、decoder前融合count信息
        if self.use_count_fusion:
            encoder_output = self.count_fusion(encoder_output, counts, types)

        # Concat formula embedding at the front of encoder output
        if self.use_formula_embedding and formula_counts is not None:
            formula_emb = self.formula_embedding(formula_counts)  # (B, 1, d_model)
            encoder_output = torch.cat([formula_emb, encoder_output], dim=1)  # (B, 1+src_len, d_model)
            # Extend padding mask for formula token (not padding)
            batch_size = nmr_padding_mask.shape[0]
            formula_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=nmr_padding_mask.device)
            nmr_padding_mask = torch.cat([formula_mask, nmr_padding_mask], dim=1)

        # Generate SMILES
        generated = self.decoder.generate(
            encoder_output=encoder_output,
            encoder_padding_mask=nmr_padding_mask,
            bos_idx=bos_idx,
            eos_idx=eos_idx,
            **kwargs,
        )

        return generated

    @torch.no_grad()
    def beam_search_generate(
        self,
        shifts: torch.Tensor,
        counts: torch.Tensor,
        types: torch.Tensor,
        nmr_padding_mask: torch.Tensor,
        bos_idx: int,
        eos_idx: int,
        formula_counts: Optional[torch.Tensor] = None,
        beam_size: int = 5,
        max_len: int = 256,
        length_penalty: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Beam search generation from NMR.

        Returns:
            sequences: (B, beam_size, max_len) - top beam_size sequences for each sample
            scores: (B, beam_size) - log probabilities
        """
        # Encode NMR
        _, _, encoder_output = self.encoder(shifts, counts, types, nmr_padding_mask)

        # 在encoder后、decoder前融合count信息
        if self.use_count_fusion:
            encoder_output = self.count_fusion(encoder_output, counts, types)

        # Concat formula embedding at the front of encoder output
        if self.use_formula_embedding and formula_counts is not None:
            formula_emb = self.formula_embedding(formula_counts)  # (B, 1, d_model)
            encoder_output = torch.cat([formula_emb, encoder_output], dim=1)  # (B, 1+src_len, d_model)
            # Extend padding mask for formula token (not padding)
            batch_size = nmr_padding_mask.shape[0]
            formula_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=nmr_padding_mask.device)
            nmr_padding_mask = torch.cat([formula_mask, nmr_padding_mask], dim=1)

        # Beam search
        sequences, scores = self.decoder.beam_search_generate(
            encoder_output=encoder_output,
            encoder_padding_mask=nmr_padding_mask,
            bos_idx=bos_idx,
            eos_idx=eos_idx,
            beam_size=beam_size,
            max_len=max_len,
            length_penalty=length_penalty,
        )

        return sequences, scores
