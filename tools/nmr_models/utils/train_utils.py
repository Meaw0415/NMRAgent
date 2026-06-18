import torch
import torch.distributed as dist
import logging
import os
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter  # NEW: Import for type hinting
# ---- 辅助函数：将字节转换为GB ----
def bytes_to_gb(bytes_val):
    return bytes_val / (1024**3)

# ---- Identity Embedding Masking ----
def apply_identity_masking(raw_shifts, mask, types):
    """
    将masked peaks根据原始shift值进行排序标记。
    最小的masked peak设置为-1,第二小的为-2,以此类推。
    氢谱和碳谱分开排序。

    Args:
        raw_shifts: [B, L] 原始化学位移值
        mask: [B, L] bool tensor, True表示该位置被mask
        types: [B, L] 0=H, 1=C

    Returns:
        masked_raw_shifts: [B, L] 应用identity masking后的shifts
    """
    batch_size, seq_len = raw_shifts.shape
    device = raw_shifts.device

    masked_raw_shifts = raw_shifts.clone()

    ranks = -(torch.arange(seq_len, device=device) + 1).float()
    ranks_expanded = ranks.expand(batch_size, seq_len)

    for atom_type in [0, 1]:
        type_mask = (types == atom_type) & mask
        if not type_mask.any():
            continue

        shifts_to_sort = raw_shifts.clone()
        shifts_to_sort[~type_mask] = float('inf')

        _, sorted_original_indices = torch.sort(shifts_to_sort, dim=1)

        identity_dest = torch.zeros_like(raw_shifts)
        identity_dest.scatter_(dim=1, index=sorted_original_indices, src=ranks_expanded)

        masked_raw_shifts[type_mask] = identity_dest[type_mask]

    return masked_raw_shifts

# NEW: Updated function signature to accept writer, global_step, rank, scheduler, bf16 parameters, and gradient clipping
def train_one_epoch(model, dataloader, optimizer, criterion, device, epoch, num_epochs,
                    writer: SummaryWriter, global_step: int, use_identity_embedding: bool = False,
                    rank: int = 0, save_dir: str = "checkpoints", save_interval: int = 100000,
                    world_size: int = 1, tensorboard_dir: str = None, scheduler=None,
                    use_bf16: bool = False, bf16_backend: str = "autocast", scaler=None,
                    max_grad_norm: float = None, grad_clip_mode: str = "norm"):

    model.train()

    # DataLoader now starts from the correct position via ResumableSampler
    # No need to use islice anymore

    if rank == 0:
        progress_bar = tqdm(
            dataloader,
            total=len(dataloader),
            desc=f"Epoch {epoch+1}/{num_epochs}",
            ncols=150
        )
    else:
        progress_bar = dataloader

    for batch in progress_bar:
        shifts = batch["raw_shifts"].to(device)
        counts = batch['counts'].to(device)
        types = batch['types'].to(device)
        raw_shifts = batch["raw_shifts"].to(device)
        padding_mask = batch['padding_mask'].to(device)
        fingerprints = batch['fingerprints'].to(device)
        h_c_shifts = batch['h_c_shifts'].to(device)
        c_shifts_list = batch['c_shifts_list']

        labels = batch["shifts"].to(device)
        # 随机 mask
        mask_prob = 0.15
        prob_matrix = torch.rand(shifts.shape, device=device)
        mask = (prob_matrix < mask_prob) & (~padding_mask)

        # 根据use_identity_embedding选择masking方式
        if use_identity_embedding:
            masked_raw_shifts = apply_identity_masking(raw_shifts, mask, types)
        else:
            masked_raw_shifts = raw_shifts.clone()
            masked_raw_shifts[mask] = -1

        optimizer.zero_grad()

        # Determine dtype for autocast
        if use_bf16 and bf16_backend == "autocast":
            autocast_dtype = torch.bfloat16
        else:
            autocast_dtype = None

        # Forward pass with optional autocast
        if autocast_dtype is not None:
            with torch.cuda.amp.autocast(dtype=autocast_dtype):
                # Handle both DDP wrapped and non-wrapped models
                if world_size > 1:
                    h_logits, c_logits, sequence_output = model.module(masked_raw_shifts, counts, types, padding_mask)
                else:
                    h_logits, c_logits, sequence_output = model(masked_raw_shifts, counts, types, padding_mask)

                total_loss, loss_cs, loss_cso, loss_fp_sim, loss_infonce = criterion(
                    h_logits, c_logits, sequence_output,
                    shifts, labels, mask, types, model, padding_mask,
                    fingerprints, h_c_shifts, c_shifts_list
                )
        else:
            # Handle both DDP wrapped and non-wrapped models
            if world_size > 1:
                h_logits, c_logits, sequence_output = model.module(masked_raw_shifts, counts, types, padding_mask)
            else:
                h_logits, c_logits, sequence_output = model(masked_raw_shifts, counts, types, padding_mask)

            total_loss, loss_cs, loss_cso, loss_fp_sim, loss_infonce = criterion(
                h_logits, c_logits, sequence_output,
                shifts, labels, mask, types, model, padding_mask,
                fingerprints, h_c_shifts, c_shifts_list
            )

        if torch.isnan(total_loss):
            logging.warning(f"NaN loss detected at step {global_step}. Skipping batch.")
            continue

        # Backward pass with optional GradScaler
        if scaler is not None:
            scaler.scale(total_loss).backward()

            # Gradient clipping (applied to scaled gradients)
            if max_grad_norm is not None:
                scaler.unscale_(optimizer)  # Unscale gradients before clipping
                if grad_clip_mode == "norm":
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                elif grad_clip_mode == "value":
                    torch.nn.utils.clip_grad_value_(model.parameters(), max_grad_norm)
                    grad_norm = None
                else:
                    raise ValueError(f"Unknown grad_clip_mode: {grad_clip_mode}")

                # Log gradient norm if available
                if rank == 0 and grad_norm is not None and global_step % 100 == 0:
                    writer.add_scalar('Gradients/grad_norm', grad_norm.item(), global_step)

            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()

            # Gradient clipping (applied to unscaled gradients)
            if max_grad_norm is not None:
                if grad_clip_mode == "norm":
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                elif grad_clip_mode == "value":
                    torch.nn.utils.clip_grad_value_(model.parameters(), max_grad_norm)
                    grad_norm = None
                else:
                    raise ValueError(f"Unknown grad_clip_mode: {grad_clip_mode}")

                # Log gradient norm if available
                if rank == 0 and grad_norm is not None and global_step % 100 == 0:
                    writer.add_scalar('Gradients/grad_norm', grad_norm.item(), global_step)

            optimizer.step()

        # Step the learning rate scheduler if provided
        if scheduler is not None:
            scheduler.step()

        # ---- TensorBoard Logging (Per Step) - only on rank 0 ----
        if rank == 0 and writer is not None and global_step % 100 == 0:
            writer.add_scalar('Loss/train_step', total_loss.item(), global_step)
            writer.add_scalar('Loss/cs_step', loss_cs.item(), global_step)
            writer.add_scalar('Loss/cso_step', loss_cso.item(), global_step)
            writer.add_scalar('Loss/fp_sim_step', loss_fp_sim.item(), global_step)
            writer.add_scalar('Loss/infonce_step', loss_infonce.item(), global_step)
            writer.add_scalar('LearningRate/lr', optimizer.param_groups[0]['lr'], global_step)

        global_step += 1

        # ---- Save checkpoint every save_interval steps (only on rank 0) ----
        if global_step % save_interval == 0:
            # IMPORTANT: All ranks must reach this point before any rank proceeds
            # First, synchronize all ranks before checkpoint saving
            # if world_size > 1:
            #     dist.barrier()

            if rank == 0:
                try:
                    save_path = os.path.join(save_dir, f"model_step_{global_step}.pth")
                    # Extract model state dict (unwrap DDP if needed)
                    model_state_dict = model.module.state_dict() if world_size > 1 else model.state_dict()
                    checkpoint_dict = {
                        'epoch': epoch + 1,
                        'model_state_dict': model_state_dict,
                        'optimizer_state_dict': optimizer.state_dict(),
                        'global_step': global_step,
                    }
                    # Add scheduler state if available
                    if scheduler is not None:
                        checkpoint_dict['scheduler_state_dict'] = scheduler.state_dict()
                    # Add GradScaler state if available
                    if scaler is not None:
                        checkpoint_dict['scaler_state_dict'] = scaler.state_dict()
                    # Add tensorboard_dir if provided
                    if tensorboard_dir:
                        checkpoint_dict['tensorboard_dir'] = tensorboard_dir

                    # Save to temporary file first, then rename (atomic operation)
                    temp_save_path = save_path + '.tmp'
                    torch.save(checkpoint_dict, temp_save_path)
                    os.replace(temp_save_path, save_path)  # Atomic rename
                    logging.info(f"Saved checkpoint at step {global_step} to {save_path}")
                except Exception as e:
                    logging.error(f"Failed to save checkpoint at step {global_step}: {e}")
                    # Don't crash the training, just log the error

            # Synchronize all processes after saving checkpoint
            # if world_size > 1:
            #     dist.barrier()

        # 统计 mask 数量
        num_masked = mask.sum().item()

        # Update progress bar only on rank 0
        if rank == 0:
            progress_bar.set_postfix_str(
                f"loss={total_loss.item():.4f} "
                f"cs={loss_cs.item():.4f} "
                f"cso={loss_cso.item():.4f} "
                f"fp={loss_fp_sim.item():.4f} "
                f"info={loss_infonce.item():.4f} "
                f"lr={optimizer.param_groups[0]['lr']:.1g}"
            )

    return global_step