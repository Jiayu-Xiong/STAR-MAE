# -*- coding: utf-8 -*-
# Author: Jiayu Xiong
# Reference: STAR-MAE / Distribution-aware Loss Reweighting (DLR), PR-D-25-06353_R2.
import argparse
import math
import os
import time
from functools import partial
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import tqdm

from dlr.reweighting import DistributionAwareLossReweighting
from datasets.audio_npz import AudiosetSpec
from models.modeling_pretrain import PretrainVisionTransformer
from optimizer import create_optimizer
from utils.mask_generator import MaskGenerator


def parse_args():
    parser = argparse.ArgumentParser(
        description='STAR-MAE pre-training with torch 2.10 SDPA/FlashAttention and DLR')

    # Data paths. Defaults follow the npz reader in datasets/audio_npz.py.
    parser.add_argument('--data-format', default='npz', choices=['npz', 'wav'])
    parser.add_argument(
        '--dataset-root',
        default=None,
        help='Dataset root directory. Required unless --csv, --npz-root/--wav-root, and --label-csv are all provided.')
    parser.add_argument('--dataset-split', default='unbal', choices=['auto', 'unbal', 'bal', 'eval'])
    parser.add_argument('--csv', default=None)
    parser.add_argument('--npz-root', default=None)
    parser.add_argument('--wav-root', default=None)
    parser.add_argument('--label-csv', default=None)
    parser.add_argument('--target-length', type=int, default=1024)
    parser.add_argument('--melbins', type=int, default=128)
    # Mel reference presets:
    # PANNs: sr=32000, mel_fmin=50, mel_fmax=14000.
    # AudioMAE: sr=16000, mel_fmin=50, mel_fmax=8000.
    parser.add_argument(
        '--sample-rate',
        type=int,
        default=16000,
        help='Audio sample rate. Presets: PANNs=32000, AudioMAE=16000.')
    parser.add_argument(
        '--mel-fmin',
        type=float,
        default=50.0,
        help='Minimum mel frequency in Hz. Presets: PANNs=50, AudioMAE=50.')
    parser.add_argument(
        '--mel-fmax',
        type=float,
        default=8000.0,
        help='Maximum mel frequency in Hz. Presets: PANNs=14000, AudioMAE=8000.')
    parser.add_argument('--mean', type=float, default=-4.421761)
    parser.add_argument('--std', type=float, default=4.32408)
    parser.add_argument('--skip-norm', action='store_true')

    # STAR-MAE representation.
    parser.add_argument('--T', type=int, default=8)
    parser.add_argument('--H', type=int, default=128)
    parser.add_argument('--W', type=int, default=128)
    parser.add_argument('--patch-h', type=int, default=16)
    parser.add_argument('--patch-w', type=int, default=16)
    parser.add_argument('--tubelet', type=int, default=1)
    parser.add_argument('--in-chans', type=int, default=1)

    # Masking.
    parser.add_argument('--encoder-mask-rate', type=float, default=0.8)
    parser.add_argument('--decoder-mask-rate', type=float, default=0.5)
    parser.add_argument('--encoder-mask-type', default='tube', choices=['random', 'tube'])
    parser.add_argument('--decoder-mask-type', default='random', choices=['random', 'cell'])

    # Model and optimization.
    parser.add_argument(
        '--encoder-embed-dim',
        type=int,
        default=768,
        help='Encoder embedding dimension. Default: ViT-B=768.')
    parser.add_argument(
        '--encoder-depth',
        type=int,
        default=12,
        help='Encoder layer count. Default: ViT-B=12.')
    parser.add_argument(
        '--encoder-num-heads',
        type=int,
        default=12,
        help='Encoder attention heads. Default: ViT-B=12.')
    parser.add_argument(
        '--decoder-embed-dim',
        type=int,
        default=384,
        help='Decoder embedding dimension. Default: ViT-B decoder=384.')
    parser.add_argument(
        '--decoder-depth',
        type=int,
        default=12,
        help='Decoder layer count. Default: STAR-MAE paper setting=12.')
    parser.add_argument(
        '--decoder-num-heads',
        type=int,
        default=6,
        help='Decoder attention heads. Default: ViT-B decoder=6.')
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--warmup-epochs', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--effective-batch-size', type=int, default=2048)
    parser.add_argument('--workers', type=int, default=24)
    parser.add_argument('--prefetch', type=int, default=2)
    parser.add_argument('--pin-memory', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--min-lr', type=float, default=1e-8)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--opt', default='adafactor')
    parser.add_argument('--opt-eps', type=float, default=1e-8)
    parser.add_argument('--opt-betas', type=float, nargs=2, default=(0.9, 0.95))
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--amp-dtype', default='bf16', choices=['bf16', 'fp16', 'fp32'])
    parser.add_argument('--compile', action='store_true')

    # Distribution-aware Loss Reweighting (DLR).
    parser.add_argument('--use-dlr', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--dlr-lr', type=float, default=1e-3)
    parser.add_argument('--dlr-left', type=float, default=0.25)
    parser.add_argument('--dlr-right', type=float, default=0.75)
    parser.add_argument('--dlr-momentum', type=float, default=0.99)
    parser.add_argument('--dlr-start-ratio', type=float, default=0.6)
    parser.add_argument('--dlr-allow-back', action='store_true')
    parser.add_argument('--dlr-with-ratio', action=argparse.BooleanOptionalAction, default=True)

    # Runtime.
    parser.add_argument('--output-dir', default='runs/star_mae_dlr')
    parser.add_argument('--log-dir', default=None)
    parser.add_argument('--save-every', type=int, default=4)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--drop-last', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--smoke-test', action='store_true')
    parser.add_argument('--max-steps-per-epoch', type=int, default=0)
    return parser.parse_args()


def fill_default_paths(args):
    if args.dataset_root is None and (args.csv is None or args.label_csv is None):
        raise ValueError(
            'Please provide --dataset-root, or explicitly set --csv, --label-csv, '
            'and the matching --npz-root/--wav-root paths for your local dataset.')
    split_layout = {
        'unbal': ('un_train_index_cleaned.csv', 'unbal', 'unbal'),
        'bal': ('train_index_cleaned.csv', 'bal_npz', 'bal'),
        'eval': ('eval_index_cleaned.csv', 'eval_npz', 'eval'),
    }
    split = args.dataset_split
    if split == 'auto':
        root_index = 1 if args.data_format == 'npz' else 2
        for candidate in ('unbal', 'bal', 'eval'):
            csv_name, npz_dir, wav_dir = split_layout[candidate]
            data_dir = (npz_dir, wav_dir)[root_index - 1]
            if (
                os.path.isfile(os.path.join(args.dataset_root, csv_name))
                and os.path.isdir(os.path.join(args.dataset_root, data_dir))
            ):
                split = candidate
                break
        else:
            split = 'eval'
    csv_name, npz_dir, wav_dir = split_layout[split]
    args.dataset_split = split
    args.csv = args.csv or os.path.join(args.dataset_root, csv_name)
    args.npz_root = args.npz_root or os.path.join(args.dataset_root, npz_dir)
    args.wav_root = args.wav_root or os.path.join(args.dataset_root, wav_dir)
    args.label_csv = args.label_csv or os.path.join(args.dataset_root, 'class_labels_indices.csv')
    args.log_dir = args.log_dir or os.path.join(args.output_dir, 'logs')
    for name in ('csv', 'label_csv'):
        path = getattr(args, name)
        if path and not os.path.isfile(path):
            raise FileNotFoundError(f'--{name.replace("_", "-")} does not exist: {path}')
    root_name = 'npz_root' if args.data_format == 'npz' else 'wav_root'
    root_path = getattr(args, root_name)
    if root_path and not os.path.isdir(root_path):
        raise FileNotFoundError(f'--{root_name.replace("_", "-")} does not exist: {root_path}')
    return args


def resolve_arch_config(args):
    return SimpleNamespace(
        encoder_embed_dim=args.encoder_embed_dim,
        encoder_depth=args.encoder_depth,
        encoder_num_heads=args.encoder_num_heads,
        decoder_embed_dim=args.decoder_embed_dim,
        decoder_depth=args.decoder_depth,
        decoder_num_heads=args.decoder_num_heads,
    )

def configure_torch(args):
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision('high')
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_math_sdp(True)


class NpzVideoDataset(Dataset):
    def __init__(self, base_dataset, T, H, W):
        self.base_dataset = base_dataset
        self.T = T
        self.H = H
        self.W = W

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        fbank, label = self.base_dataset[index]
        if fbank.ndim == 3:
            frames = fbank
        elif fbank.ndim == 2:
            expected = self.T * self.H
            if fbank.shape[0] != expected or fbank.shape[1] != self.W:
                raise ValueError(
                    f'NPZ shape {tuple(fbank.shape)} cannot reshape to '
                    f'({self.T}, {self.H}, {self.W}); set --target-length, --melbins, --T, --H, --W.')
            frames = fbank.reshape(self.T, self.H, self.W)
        else:
            raise ValueError(f'Unsupported NPZ tensor shape: {tuple(fbank.shape)}')
        return frames.contiguous().clone(), label.contiguous().clone()


def collate_npz_video(batch):
    frames, labels = zip(*batch)
    return torch.stack(frames, dim=0), torch.stack(labels, dim=0)


def build_dataloader(args):
    if args.data_format == 'npz':
        args.mode = 'train'
        dataset = NpzVideoDataset(
            AudiosetSpec(args.csv, args, label_csv=args.label_csv),
            args.T,
            args.H,
            args.W,
        )
    else:
        from datasets.audio_set import get_dataset_2M
        train_loader, _ = get_dataset_2M(args)
        return train_loader

    loader_kwargs = {
        'dataset': dataset,
        'batch_size': args.batch_size,
        'shuffle': True,
        'num_workers': args.workers,
        'pin_memory': args.pin_memory,
        'drop_last': args.drop_last,
        'persistent_workers': args.workers > 0,
        'collate_fn': collate_npz_video,
    }
    if args.workers > 0:
        loader_kwargs['prefetch_factor'] = args.prefetch
    return DataLoader(**loader_kwargs)


def build_model(args, device):
    arch = resolve_arch_config(args)
    model = PretrainVisionTransformer(
        in_chans=args.in_chans,
        encoder_in_chans=args.in_chans,
        encoder_num_classes=0,
        encoder_embed_dim=arch.encoder_embed_dim,
        encoder_depth=arch.encoder_depth,
        encoder_num_heads=arch.encoder_num_heads,
        all_frames=args.T,
        img_size=(args.H, args.W),
        patch_size=args.patch_h,
        tubelet_size=args.tubelet,
        decoder_embed_dim=arch.decoder_embed_dim,
        decoder_depth=arch.decoder_depth,
        decoder_num_heads=arch.decoder_num_heads,
        decoder_num_classes=args.in_chans * args.tubelet * args.patch_h * args.patch_w,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
    ).to(device)
    if args.compile:
        model = torch.compile(model)
    return model


def cosine_scheduler(base_lr, final_lr, epochs, niter_per_ep, warmup_epochs=0, start_warmup_lr=0.0):
    total_iters = epochs * niter_per_ep
    warmup_iters = warmup_epochs * niter_per_ep
    if total_iters <= 0:
        return []
    schedule = []
    for i in range(total_iters):
        if i < warmup_iters:
            alpha = i / max(1, warmup_iters)
            schedule.append(start_warmup_lr + alpha * (base_lr - start_warmup_lr))
        else:
            progress = (i - warmup_iters) / max(1, total_iters - warmup_iters)
            schedule.append(final_lr + 0.5 * (base_lr - final_lr) * (1 + math.cos(math.pi * progress)))
    return schedule


def patchify_video(x, patch_h, patch_w, tubelet):
    b, c, t, h, w = x.shape
    if t % tubelet != 0 or h % patch_h != 0 or w % patch_w != 0:
        raise ValueError(f'Input shape {tuple(x.shape)} is not divisible by patch/tubelet sizes.')
    x = x.unfold(2, tubelet, tubelet).unfold(3, patch_h, patch_h).unfold(4, patch_w, patch_w)
    x = x.permute(0, 2, 3, 4, 1, 5, 6, 7)
    return x.reshape(b, -1, c * tubelet * patch_h * patch_w)


def dlr_weighted_mse(pred, target, dlr, dlr_optimizer, global_step,
                     eps=1e-12, return_stats=False):
    patch_mse = F.mse_loss(pred, target, reduction='none').mean(dim=-1)
    log_patch_mse = torch.log(patch_mse.detach().clamp_min(eps)).reshape(-1, 1)
    dnf_grad_norm = patch_mse.new_tensor(0.0)

    if dlr_optimizer is not None:
        dlr_optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=patch_mse.device.type, enabled=False):
            mle_loss = dlr.train_dnf_for_mle(log_patch_mse.float())
        mle_loss.backward()
        grad_sq_sum = patch_mse.new_tensor(0.0)
        for param in dlr.dual_flow.online_flow.parameters():
            if param.grad is not None:
                grad_sq_sum = grad_sq_sum + param.grad.detach().float().pow(2).sum()
        dnf_grad_norm = grad_sq_sum.sqrt()
        dlr_optimizer.step()
        dlr.update_momentum_flow()
    else:
        mle_loss = patch_mse.new_tensor(0.0)

    weights = dlr.generate_weight(log_patch_mse, global_step).reshape_as(patch_mse)
    loss = (patch_mse * weights.to(dtype=patch_mse.dtype)).mean()
    if not return_stats:
        return loss, mle_loss.detach()

    stats = {
        'mle_loss': mle_loss.detach(),
        'dnf_grad_norm': dnf_grad_norm.detach(),
        'pss_scale': patch_mse.new_tensor(float(dlr.pss_module(global_step))),
        'start_iter': patch_mse.new_tensor(float(dlr.start_iter)),
        'interval_left': patch_mse.new_tensor(float(dlr.igr.left)),
        'interval_right': patch_mse.new_tensor(float(dlr.igr.right)),
        'interval_width': patch_mse.new_tensor(float(dlr.igr.right - dlr.igr.left)),
        'weight_mean': weights.detach().mean(),
        'weight_std': weights.detach().std(unbiased=False),
        'weight_min': weights.detach().min(),
        'weight_max': weights.detach().max(),
        'log_patch_mse_mean': log_patch_mse.detach().mean(),
        'log_patch_mse_std': log_patch_mse.detach().std(unbiased=False),
        'patch_mse_mean': patch_mse.detach().mean(),
    }
    return loss, mle_loss.detach(), stats


def get_amp_dtype(args):
    if args.amp_dtype == 'bf16':
        return torch.bfloat16
    if args.amp_dtype == 'fp16':
        return torch.float16
    return torch.float32


def main():
    args = fill_default_paths(parse_args())
    configure_torch(args)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader = build_dataloader(args)
    steps_per_epoch = args.max_steps_per_epoch if args.max_steps_per_epoch > 0 else len(train_loader)
    total_steps = args.epochs * steps_per_epoch

    model = build_model(args, device)
    mask_generator = MaskGenerator(
        input_shape=(args.T, args.H, args.W),
        patch_size=(args.patch_h, args.patch_w),
        en_mask_rate=args.encoder_mask_rate,
        de_mask_rate=args.decoder_mask_rate,
        en_mask_type=args.encoder_mask_type,
        de_mask_type=args.decoder_mask_type,
        tubelet_size=args.tubelet,
    )

    optim_args = SimpleNamespace(
        opt=args.opt,
        lr=args.lr,
        weight_decay=args.weight_decay,
        opt_eps=args.opt_eps,
        opt_betas=tuple(args.opt_betas),
        momentum=args.momentum,
    )
    optimizer = create_optimizer(optim_args, model)
    accum_iter = max(1, args.effective_batch_size // max(1, args.batch_size))
    actual_lr = args.lr * (args.batch_size * accum_iter) / 256
    lr_schedule = cosine_scheduler(actual_lr, args.min_lr, args.epochs, steps_per_epoch, args.warmup_epochs)

    dlr = None
    dlr_optimizer = None
    if args.use_dlr:
        dlr = DistributionAwareLossReweighting(
            total_iter=max(1, total_steps),
            left=args.dlr_left,
            right=args.dlr_right,
            allow_back=args.dlr_allow_back,
            with_ratio=args.dlr_with_ratio,
            start_iter=int(args.dlr_start_ratio * max(1, total_steps)),
            momentum=args.dlr_momentum,
        ).to(device)
        dlr_optimizer = torch.optim.AdamW(dlr.dual_flow.online_flow.parameters(), lr=args.dlr_lr)

    amp_dtype = get_amp_dtype(args)
    use_amp = device.type == 'cuda' and amp_dtype is not torch.float32
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp and amp_dtype is torch.float16)
    writer = SummaryWriter(log_dir=args.log_dir)

    print(f'device={device}, torch={torch.__version__}, amp={args.amp_dtype}, flash_sdp=True')
    print(f'data={args.data_format}, split={args.dataset_split}, csv={args.csv}, npz_root={args.npz_root}')
    arch = resolve_arch_config(args)
    print(
        f'encoder=(dim={arch.encoder_embed_dim}, depth={arch.encoder_depth}, heads={arch.encoder_num_heads}), '
        f'decoder=(dim={arch.decoder_embed_dim}, depth={arch.decoder_depth}, heads={arch.decoder_num_heads})')
    print(f'shape=(T={args.T}, H={args.H}, W={args.W}), patch=({args.patch_h},{args.patch_w}), tubelet={args.tubelet}')
    print(f'accum_iter={accum_iter}, actual_lr={actual_lr:.6e}, dlr={args.use_dlr}')
    if dlr is not None:
        dlr_start_epoch = dlr.start_iter / max(1, steps_per_epoch)
        print(
            f'dlr_dnf_learning=all_steps, dlr_constraint_start_iter={dlr.start_iter}, '
            f'dlr_constraint_start_epoch={dlr_start_epoch:.2f}')

    if args.smoke_test:
        x, y = next(iter(train_loader))
        x = x.unsqueeze(1).to(device, non_blocking=True)
        en_mask, de_mask = mask_generator.generate_batch_masks(x.shape[0], device=x.device)
        reconstruction_overlap = int(((~de_mask) & (~en_mask)).sum().item())
        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            pred = model(x, en_mask, de_mask)
            patches = patchify_video(x, args.patch_h, args.patch_w, args.tubelet)
            target = patches[~de_mask].reshape(x.shape[0], -1, patches.shape[-1])
            if dlr is not None:
                loss, _ = dlr_weighted_mse(pred, target, dlr, dlr_optimizer, 0)
            else:
                loss = F.mse_loss(pred, target)
        loss_name = 'dlr_weighted_mse' if dlr is not None else 'mse'
        print(f'smoke x={tuple(x.shape)}, pred={tuple(pred.shape)}, target={tuple(target.shape)}, loss={loss.item():.6f}, loss_name={loss_name}, decoder_encoder_visible_overlap={reconstruction_overlap}')
        return

    model.train()
    optimizer.zero_grad(set_to_none=True)
    start = time.time()
    global_step = 0
    for epoch in range(args.epochs):
        pbar = tqdm.tqdm(train_loader, total=steps_per_epoch, desc=f'Epoch {epoch + 1}/{args.epochs}')
        running_loss = 0.0
        for step, (x, _) in enumerate(pbar):
            if args.max_steps_per_epoch > 0 and step >= args.max_steps_per_epoch:
                break
            if global_step < len(lr_schedule):
                for group in optimizer.param_groups:
                    group['lr'] = lr_schedule[global_step] * group.get('lr_scale', 1.0)

            x = x.unsqueeze(1).to(device, non_blocking=True)
            en_mask, de_mask = mask_generator.generate_batch_masks(x.shape[0], device=x.device)
            reconstruction_overlap = ((~de_mask) & (~en_mask)).sum()
            if reconstruction_overlap.item() != 0:
                raise RuntimeError(
                    f'Decoder targets include {int(reconstruction_overlap.item())} encoder-visible patches.')

            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                pred = model(x, en_mask, de_mask)
                patches = patchify_video(x, args.patch_h, args.patch_w, args.tubelet)
                target = patches[~de_mask].reshape(x.shape[0], -1, patches.shape[-1])
                if dlr is not None:
                    loss, mle_loss, dlr_stats = dlr_weighted_mse(
                        pred, target, dlr, dlr_optimizer, global_step, return_stats=True)
                else:
                    loss = F.mse_loss(pred, target)
                    mle_loss = loss.new_tensor(0.0)
                    dlr_stats = {}

            loss_value = float(loss.detach())
            scaler.scale(loss / accum_iter).backward()
            if (step + 1) % accum_iter == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            lr = optimizer.param_groups[0]['lr']
            running_loss += loss_value
            writer.add_scalar('Loss/train', loss_value, global_step)
            writer.add_scalar('Loss/dlr_mle', float(mle_loss), global_step)
            writer.add_scalar('LearningRate/train', lr, global_step)
            writer.add_scalar(
                'Mask/decoder_encoder_visible_overlap',
                float(reconstruction_overlap.detach()),
                global_step,
            )
            if dlr_stats:
                writer.add_scalar('DLR/mle_loss', float(dlr_stats['mle_loss']), global_step)
                writer.add_scalar('DLR/dnf_grad_norm', float(dlr_stats['dnf_grad_norm']), global_step)
                writer.add_scalar('DLR/pss_scale', float(dlr_stats['pss_scale']), global_step)
                writer.add_scalar('DLR/start_iter', float(dlr_stats['start_iter']), global_step)
                writer.add_scalar('DLR/interval_left', float(dlr_stats['interval_left']), global_step)
                writer.add_scalar('DLR/interval_right', float(dlr_stats['interval_right']), global_step)
                writer.add_scalar('DLR/interval_width', float(dlr_stats['interval_width']), global_step)
                writer.add_scalar('DLR/weight_mean', float(dlr_stats['weight_mean']), global_step)
                writer.add_scalar('DLR/weight_std', float(dlr_stats['weight_std']), global_step)
                writer.add_scalar('DLR/weight_min', float(dlr_stats['weight_min']), global_step)
                writer.add_scalar('DLR/weight_max', float(dlr_stats['weight_max']), global_step)
                writer.add_scalar('DLR/log_patch_mse_mean', float(dlr_stats['log_patch_mse_mean']), global_step)
                writer.add_scalar('DLR/log_patch_mse_std', float(dlr_stats['log_patch_mse_std']), global_step)
                writer.add_scalar('DLR/patch_mse_mean', float(dlr_stats['patch_mse_mean']), global_step)
            pbar.set_postfix(loss=f'{loss_value:.6f}', lr=f'{lr:.3e}')
            global_step += 1

        epoch_loss = running_loss / max(1, step + 1)
        writer.add_scalar('Loss/epoch', epoch_loss, epoch + 1)
        if (epoch + 1) % args.save_every == 0 or (epoch + 1) == args.epochs:
            ckpt = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'dlr': dlr.state_dict() if dlr is not None else None,
                'epoch': epoch + 1,
                'args': vars(args),
            }
            path = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch + 1}.pth')
            torch.save(ckpt, path)
            print(f'saved {path}')

    elapsed = time.time() - start
    print('training time: {:.0f}m {:.0f}s'.format(elapsed // 60, elapsed % 60))
    writer.close()


if __name__ == '__main__':
    main()
