import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import time
# from logger import MetricsLogger  # Remove logger
from models.modeling_pretrain import pretrain_videomae_small_patch16_224, pretrain_videomae_base_patch16_224  # Ensure correct model import
from datasets.AudioSet import get_dataset_20K, get_dataset_2M
from utils.mask_generator import MaskGenerator
import argparse
import tqdm
import einops
from torch.utils.tensorboard import SummaryWriter

args = argparse.ArgumentParser('MAE pre-training', add_help=False)

from optimizer import create_optimizer
# Assume args is an object containing optimizer configurations
args.opt = 'adafactor'  # or 'adafactor'
args.lr = 0.0001
args.weight_decay = 0.0001
args.opt_eps = 1e-8
args.opt_betas = (0.9, 0.95)

# Audio configuration
audio_conf = {
    'sample_rate': 16000,
    'n_fft': 2048,
    'H': 256,
    'W': 256,
    'T': 4,
    'overlap_rate': 0.2,
    'per_frame_overlap': False,
    'use_mel': True,
    'hop_length': None,
    'mixup': 0.0,  # Set mixup to 0.0 as per literature
    'mean': -4.268,  # Set dataset mean as per literature
    'std': 4.569,    # Set dataset standard deviation as per literature
    'skip_norm': False,
    'noise': False,
    'patch_h': 16,
    'patch_w': 16,
    'tubelet': 1,
    'batchsize': 64,  # Original batch size was 4
    'workers': 24,
    'root': 'as2m csv',
    'encoder_msk_rate': 0.6, 
    'decoder_msk_rate': 0.5,
    'encoder_msk_type': 'tube',
    'decoder_msk_type': 'random'
}

def cosine_scheduler(base_lr, final_lr, epochs, niter_per_ep, warmup_epochs=0, start_warmup_lr=0.0):
    warmup_iters = warmup_epochs * niter_per_ep
    total_iters = epochs * niter_per_ep
    lr_schedule = np.array([])
    if warmup_iters > 0:
        warmup_lr_schedule = np.linspace(start_warmup_lr, base_lr, warmup_iters)
        lr_schedule = np.concatenate((lr_schedule, warmup_lr_schedule))
    iters = np.arange(total_iters - warmup_iters)
    cosine_lr_schedule = final_lr + 0.5 * (base_lr - final_lr) * (1 + np.cos(np.pi * iters / (total_iters - warmup_iters)))
    lr_schedule = np.concatenate((lr_schedule, cosine_lr_schedule))
    return lr_schedule

def main():
    # Hyperparameter settings
    n_epochs = 60
    base_lr = args.lr  # Base learning rate
    warmup_epochs = 4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load datasets
    train_loader, test_loader = get_dataset_2M(audio_conf=audio_conf)
    cls = audio_conf.get('patch_h') * audio_conf.get('patch_w') * audio_conf.get('tubelet')
    
    # Define model
    # model = pretrain_videomae_base_patch16_224(
    #     in_chans=1,
    #     all_frames=audio_conf.get('T'),
    #     img_size = audio_conf.get('H'),
    #     patch_size = audio_conf.get('patch_h'),
    #     tubelet_size = audio_conf.get('tubelet'),
    #     decoder_num_classes = cls
    # ).to(device)
    model = pretrain_videomae_small_patch16_224(
        in_chans=1,
        all_frames=audio_conf.get('T'),
        img_size = audio_conf.get('H'),
        patch_size = audio_conf.get('patch_h'),
        tubelet_size = audio_conf.get('tubelet'),
        decoder_num_classes = cls
    ).to(device)
    
    # Initialize Mask Generator
    mask_generator = MaskGenerator(
        input_shape=(audio_conf.get('T'), audio_conf.get('H'), audio_conf.get('W')),
        patch_size=(audio_conf.get('patch_h'), audio_conf.get('patch_w')),
        en_mask_rate=audio_conf.get('encoder_msk_rate'),
        de_mask_rate=audio_conf.get('decoder_msk_rate'),
        en_mask_type=audio_conf.get('encoder_msk_type'),
        de_mask_type=audio_conf.get('decoder_msk_type'),
        tubelet_size=audio_conf.get('tubelet')
    )

    # Calculate effective batch size and actual learning rate
    batch_size = audio_conf['batchsize']  # Original batch size was 4
    world_size = 1  # Assume distributed training is not used

    # Set gradient accumulation steps to achieve an effective batch size of 2048
    desired_effective_batch_size = 2048
    accum_iter = desired_effective_batch_size // (batch_size * world_size)  # Calculate accumulation steps
    print(f"Gradient accumulation steps: {accum_iter}")

    eff_batch_size = batch_size * accum_iter * world_size
    actual_lr = base_lr * eff_batch_size / 256  # Calculate based on --blr $blr

    # Optimizer
    optimizer = create_optimizer(args, model)

    # Learning rate scheduler
    lr_schedule = cosine_scheduler(
        base_lr=actual_lr,
        final_lr=1e-8,
        epochs=n_epochs,
        niter_per_ep=len(train_loader),
        warmup_epochs=warmup_epochs
    )

    # Loss scaler for mixed precision training
    loss_scaler = GradScaler()

    # Define loss function
    reconstruction_loss_fn = nn.MSELoss(reduction='mean')

    # Initialize TensorBoard's SummaryWriter
    writer = SummaryWriter(log_dir='logs')  # You can specify the log directory

    model.train()
    since = time.time()
    optimizer.zero_grad()
    for epoch in range(n_epochs):
        print(f"Epoch {epoch + 1}/{n_epochs}")
        pbar = tqdm.tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}")
        running_loss = 0.0  # Used to calculate average loss
        for data_iter_step, data in pbar:
            global_step = epoch * len(train_loader) + data_iter_step

            # Update learning rate
            lr = lr_schedule[global_step]
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            # Get data
            en_mask, de_mask = mask_generator.generate_masks()
            en_mask = np.tile(en_mask, (batch_size, 1))
            de_mask = np.tile(de_mask, (batch_size, 1))
            x, _ = data  # Assume labels are not needed
            x = x.unsqueeze(1)  # Adjust dimensions as per model requirements (B, 1, T, H, W)
            x = x.to(device)
            en_mask = torch.tensor(en_mask, dtype=bool).to(device)
            de_mask = torch.tensor(de_mask, dtype=bool).to(device)
            
            # Forward pass and loss computation
            with autocast():
                reconstructed = model(x, en_mask, de_mask)  # Model returns reconstructed masked patches (B * N_masked, C, patch_h, patch_w)
                patches = einops.rearrange(
                    x, 
                    'b c t (h p1) (w p2) -> b (t h w) (c p1 p2)', 
                    p1=16, p2=16
                )
                # Extract decoder masked patches as targets
                x_decoder_patches = patches[~de_mask].reshape(batch_size, -1, patches.shape[-1])
                # x_decoder_patches = patches[en_mask].reshape(batch_size, -1, patches.shape[-1])
                # print('reconed')
                # print(x_decoder_patches.shape)
                # Calculate reconstruction loss
                loss = reconstruction_loss_fn(reconstructed, x_decoder_patches)  # Scalar
                loss_value = loss.item()

            # Gradient accumulation
            loss = loss / accum_iter  # Average the loss to prevent gradients from becoming too large
            loss_scaler.scale(loss).backward()

            # Perform optimizer step and zero gradients when accumulation steps are reached
            if (data_iter_step + 1) % accum_iter == 0 or (data_iter_step + 1) == len(train_loader):
                loss_scaler.step(optimizer)
                loss_scaler.update()
                optimizer.zero_grad()

                # After gradient update, log average loss and learning rate to TensorBoard
                avg_loss = running_loss / accum_iter  # Calculate average loss
                current_lr = optimizer.param_groups[0]['lr']
                writer.add_scalar('Loss/train', avg_loss, global_step)
                writer.add_scalar('Learning Rate', current_lr, global_step)
                running_loss = 0.0  # Reset running loss

            # Accumulate running loss
            running_loss += loss_value

            # Update progress bar display
            pbar.set_postfix({'Loss': f'{loss_value:.6f}', 'LR': f'{lr:.6e}'})
        
        # Save the model after each epoch
        if (epoch + 1) % 4 == 0 or (epoch + 1) == n_epochs:
            checkpoint_path = f"checkpoint_epoch_{epoch+1}.pth"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved model to {checkpoint_path}")

    time_used = time.time() - since
    print('Training time: {:.0f}m {:.0f}s'.format(time_used // 60, time_used % 60))
    writer.close()  # Close TensorBoard's SummaryWriter

if __name__ == '__main__':
    main()
