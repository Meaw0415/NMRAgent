"""
Integrated NMR to 3D Structure Model.

Combines pretrained NMR_BERT encoder with ChefNMR's Diffusion Transformer decoder.
"""

import torch
import torch.nn as nn

from .nmr_bert import NMR_BERT
from .chefnmr_modules.diffusion import AtomDiffusion
from .chefnmr_modules.score_models import DiffusionModuleTransformer
from omegaconf import OmegaConf


class NMRBERTConditionEmbedder(nn.Module):
    """
    Adapter that wraps NMR_BERT encoder to produce conditioning embeddings
    compatible with ChefNMR's diffusion transformer.
    """

    def __init__(
        self,
        nmr_bert: NMR_BERT,
        d_model: int = 768,
        dropout: float = 0.1,
        freeze_encoder: bool = True,
    ):
        """
        Args:
            nmr_bert: Pretrained NMR_BERT encoder
            d_model: Hidden dimension of NMR_BERT (should match encoder)
            output_dim: Output dimension for diffusion model conditioning
            dropout: Dropout rate
            freeze_encoder: Whether to freeze NMR_BERT weights
        """
        super().__init__()
        self.encoder = nmr_bert
        self.d_model = d_model

        # Freeze encoder if specified
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def forward(self, shifts, counts, types, padding_mask):
        """
        Forward pass through NMR encoder.

        Args:
            shifts: NMR chemical shifts (B, L)
            counts: Peak counts for H atoms (B, L)
            types: Atom types (0=H, 1=C) (B, L)
            padding_mask: Padding mask (B, L)

        Returns:
            Conditioning embedding (B, output_dim)
        """
        # Get encoder outputs (h_logits, c_logits, embeddings)
        # If encoder is frozen, run in no_grad context to save memory
        # But we need to re-enable gradients for the embeddings so that
        # gradients can flow back to the diffusion model
        encoder_frozen = all(not p.requires_grad for p in self.encoder.parameters())

        if encoder_frozen:
            # Encoder is frozen: run in no_grad context
            with torch.no_grad():
                _, _, embeddings = self.encoder(shifts, counts, types, padding_mask)
            # Re-enable gradients for embeddings (detach and require grad)
            # This allows gradients to flow to downstream layers (y_embedder)
            # while preventing gradients from flowing to the frozen encoder
            embeddings = embeddings.detach().requires_grad_(True)
        else:
            # Encoder is trainable: compute with gradients
            _, _, embeddings = self.encoder(shifts, counts, types, padding_mask)

        # Pool over sequence dimension (mean pooling over non-padded tokens)
        # padding_mask is True for padding, so we invert it
        mask = (~padding_mask).float().unsqueeze(-1)  # (B, L, 1)
        pooled = (embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)  # (B, d_model)

        # Project to output dimension
        conditioning = pooled

        return conditioning


class NMRToDiffusion3D(nn.Module):
    """
    Complete NMR to 3D structure model using diffusion.

    Architecture:
        1. NMR_BERT encoder: Processes NMR spectra to embeddings
        2. Diffusion Transformer: Generates 3D coordinates conditioned on NMR
    """

    def __init__(
        self,
        nmr_encoder: NMR_BERT,
        diffusion_config: dict,
        encoder_output_dim: int = 768,
        freeze_encoder: bool = True,
        remove_h: bool = False,
    ):
        """
        Args:
            nmr_encoder: Pretrained NMR_BERT model
            diffusion_config: Configuration dict for diffusion model
            encoder_output_dim: Output dimension from encoder adapter
            freeze_encoder: Whether to freeze encoder weights
            remove_h: Whether to remove hydrogen atoms
        """
        super().__init__()

        self.remove_h = remove_h

        # Wrap encoder with adapter
        self.condition_embedder = NMRBERTConditionEmbedder(
            nmr_bert=nmr_encoder,
            d_model=nmr_encoder.embedding.d_model,
            dropout=diffusion_config.get('dropout', 0.1),
            freeze_encoder=freeze_encoder,
        )

        # Create diffusion model config
        score_model_args = OmegaConf.create({
            'model_name': 'DiffusionModuleTransformer',
            'DiffusionModuleTransformer': {
                'in_atom_feature_size': diffusion_config['in_atom_feature_size'],
                'out_atom_coords_size': 3,
                'condition': 'PreEmbedded',  # Use pre-embedded conditioning
                'in_condition_size': encoder_output_dim,
                'max_n_atoms': diffusion_config['max_n_atoms'],
                'drop_transform': 'zero',
                'n_blocks': diffusion_config.get('n_blocks', 10),
                'n_heads': diffusion_config.get('n_heads', 8),
                'hidden_size': diffusion_config.get('hidden_size', 512),
                'mlp_ratio': diffusion_config.get('mlp_ratio', 4.0),
                'embedder_args': None,  # Not needed for pre-embedded conditioning
            }
        })

        # Create diffusion model directly without monkey-patching
        # The DiffusionModuleTransformer now supports PreEmbedded condition type
        self.diffusion = AtomDiffusion(
            score_model_args=score_model_args,
            train_sigma_distribution_type=diffusion_config.get('train_sigma_distribution_type', 'af3'),
            train_sigma_args=diffusion_config.get('train_sigma_args', {'edm_P_mean': -1.2, 'edm_P_std': 1.3}),
            sample_sigma_schedule_type=diffusion_config.get('sample_sigma_schedule_type', 'edm'),
            sample_gamma_schedule_type=diffusion_config.get('sample_gamma_schedule_type', 'edm'),
            num_sampling_steps=diffusion_config.get('num_sampling_steps', 50),
            sigma_min=diffusion_config.get('sigma_min', 0.001),
            sigma_max=diffusion_config.get('sigma_max', 80.0),
            gamma_min=diffusion_config.get('gamma_min', 1.0),
            noise_scale=diffusion_config.get('noise_scale', 1.0),
            step_scale=diffusion_config.get('step_scale', 1.0),
            guidance_scale=diffusion_config.get('guidance_scale', 0.0),
            synchronize_sigmas=diffusion_config.get('synchronize_sigmas', False),
            coordinate_transformation_when_training=diffusion_config.get(
                'coordinate_transformation_when_training', 'centering_rotation_translation'
            ),
            edm_args=diffusion_config.get('edm_args', {
                'sigma_data': 3.0,
                'rho': 7,
                'use_heun_solver': True,
                'gamma_0': 0.8
            }),
        )

    def forward(self, nmr_data, atom_features, multiplicity=1):
        """
        Forward pass for training.

        Args:
            nmr_data: Dict with keys 'shifts', 'counts', 'types', 'padding_mask'
            atom_features: Dict with keys 'atom_coords', 'atom_one_hot', 'atom_mask'
            multiplicity: Number of diffusion samples per input

        Returns:
            Dict containing diffusion outputs
        """
        # Get conditioning from NMR encoder
        condition = self.condition_embedder(
            nmr_data['shifts'],
            nmr_data['counts'],
            nmr_data['types'],
            nmr_data['padding_mask']
        )

        # Prepare model inputs for diffusion
        model_inputs = {
            'atom_one_hot': atom_features['atom_one_hot'],
            'atom_mask': atom_features['atom_mask'],
            'condition': condition,
        }

        # Run diffusion forward pass
        dict_out = self.diffusion(
            model_inputs=model_inputs,
            atom_coords=atom_features['atom_coords'],
            multiplicity=multiplicity,
        )

        return dict_out

    def compute_loss(self, dict_out, atom_features, multiplicity=1,
                     add_smooth_lddt_loss=True, lddt_loss_threshold=[0.5, 1.0, 2.0, 4.0]):
        """
        Compute diffusion loss.

        Args:
            dict_out: Output from forward pass
            atom_features: Ground truth atom features
            multiplicity: Diffusion multiplicity
            add_smooth_lddt_loss: Whether to add smooth LDDT loss
            lddt_loss_threshold: Thresholds for LDDT loss

        Returns:
            Dict with 'loss' and 'loss_breakdown'
        """
        model_inputs = {
            'atom_mask': atom_features['atom_mask'],
        }

        return self.diffusion.compute_loss(
            model_inputs=model_inputs,
            dict_out=dict_out,
            multiplicity=multiplicity,
            add_smooth_lddt_loss=add_smooth_lddt_loss,
            lddt_loss_threshold=lddt_loss_threshold,
        )

    def sample(self, nmr_data, atom_features, num_sampling_steps=None,
               multiplicity=1, n_chain_frames=1):
        """
        Sample 3D structures from NMR spectra.

        Args:
            nmr_data: Dict with NMR data
            atom_features: Dict with atom types and mask (no coordinates needed)
            num_sampling_steps: Number of diffusion sampling steps
            multiplicity: Number of samples to generate
            n_chain_frames: Number of intermediate frames to save

        Returns:
            Tuple of (final_coords, coord_chains)
        """
        # Get conditioning
        condition = self.condition_embedder(
            nmr_data['shifts'],
            nmr_data['counts'],
            nmr_data['types'],
            nmr_data['padding_mask']
        )

        # Prepare model inputs
        model_inputs = {
            'atom_one_hot': atom_features['atom_one_hot'],
            'atom_mask': atom_features['atom_mask'],
            'condition': condition,
        }

        # Sample from diffusion model
        return self.diffusion.sample(
            model_inputs=model_inputs,
            num_sampling_steps=num_sampling_steps,
            multiplicity=multiplicity,
            n_chain_frames=n_chain_frames,
        )
