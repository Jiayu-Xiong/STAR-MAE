# Author: Jiayu Xiong
# Reference: STAR-MAE / Distribution-aware Loss Reweighting (DLR), PR-D-25-06353_R2.
import argparse
import csv
import os
import sys
import time
from types import SimpleNamespace

import torch
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dlr.reweighting import DistributionAwareLossReweighting
from optimizer import create_optimizer
from train_star_mae_dlr import (
    build_model,
    configure_torch,
    dlr_weighted_mse,
    patchify_video,
    resolve_arch_config,
)
from utils.mask_generator import MaskGenerator


def parse_args():
    parser = argparse.ArgumentParser(description='Benchmark STAR-MAE batch size and FlashAttention.')
    parser.add_argument('--batches', type=int, nargs='+', default=[32, 64, 128, 256, 512])
    parser.add_argument('--T', type=int, default=8)
    parser.add_argument('--H', type=int, default=128)
    parser.add_argument('--W', type=int, default=128)
    parser.add_argument('--patch-h', type=int, default=16)
    parser.add_argument('--patch-w', type=int, default=16)
    parser.add_argument('--tubelet', type=int, default=1)
    parser.add_argument('--encoder-embed-dim', type=int, default=768)
    parser.add_argument('--encoder-depth', type=int, default=12)
    parser.add_argument('--encoder-num-heads', type=int, default=12)
    parser.add_argument('--decoder-embed-dim', type=int, default=384)
    parser.add_argument('--decoder-depth', type=int, default=12)
    parser.add_argument('--decoder-num-heads', type=int, default=6)
    parser.add_argument('--encoder-mask-rates', type=float, nargs='+', default=[0.8])
    parser.add_argument('--decoder-mask-rates', type=float, nargs='+', default=[0.5])
    parser.add_argument('--encoder-mask-type', default='tube', choices=['random', 'tube'])
    parser.add_argument('--decoder-mask-type', default='random', choices=['random', 'cell'])
    parser.add_argument('--amp-dtype', default='bf16', choices=['bf16', 'fp16'])
    parser.add_argument('--steps', type=int, default=2)
    parser.add_argument('--warmup-steps', type=int, default=2)
    parser.add_argument('--with-dlr', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--opt', default='adafactor')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--opt-eps', type=float, default=1e-8)
    parser.add_argument('--opt-betas', type=float, nargs=2, default=(0.9, 0.95))
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--input-path', required=True, help='Path to a user-provided input tensor with shape (N, 1, T, H, W).')
    parser.add_argument('--output', default='benchmarks/results/batch_flash.csv')
    return parser.parse_args()


def set_flash_attention(enabled):
    torch.backends.cuda.enable_flash_sdp(enabled)
    torch.backends.cuda.enable_mem_efficient_sdp(enabled)
    torch.backends.cuda.enable_math_sdp(True)


def load_input_batch(path, device):
    if path.endswith(('.pt', '.pth')):
        tensor = torch.load(path, map_location=device)
    elif path.endswith('.npy'):
        tensor = torch.from_numpy(np.load(path))
    elif path.endswith('.npz'):
        with np.load(path) as data:
            if len(data.files) != 1:
                raise ValueError(f'{path} must contain exactly one array, found {data.files}')
            tensor = torch.from_numpy(data[data.files[0]])
    else:
        raise ValueError(f'Unsupported input tensor file extension: {path}')
    if isinstance(tensor, dict):
        raise ValueError(f'{path} must store a tensor directly, not a dict.')
    if tensor.ndim != 5:
        raise ValueError(f'input tensor must have shape (N, 1, T, H, W), got {tuple(tensor.shape)}')
    return tensor.to(device=device, dtype=torch.float32)


def resize_batch(input_batch, batch_size):
    if input_batch.shape[0] == batch_size:
        return input_batch
    repeat = (batch_size + input_batch.shape[0] - 1) // input_batch.shape[0]
    return input_batch.repeat((repeat, 1, 1, 1, 1))[:batch_size]


def run_case(args, input_batch, batch_size, flash_enabled, encoder_mask_rate, decoder_mask_rate):
    if not torch.cuda.is_available():
        return {'status': 'no_cuda'}

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    device = torch.device('cuda')
    amp_dtype = torch.bfloat16 if args.amp_dtype == 'bf16' else torch.float16

    model_args = SimpleNamespace(
        seed=args.seed,
        in_chans=1,
        T=args.T,
        H=args.H,
        W=args.W,
        patch_h=args.patch_h,
        patch_w=args.patch_w,
        tubelet=args.tubelet,
        encoder_embed_dim=args.encoder_embed_dim,
        encoder_depth=args.encoder_depth,
        encoder_num_heads=args.encoder_num_heads,
        decoder_embed_dim=args.decoder_embed_dim,
        decoder_depth=args.decoder_depth,
        decoder_num_heads=args.decoder_num_heads,
        compile=False,
    )
    configure_torch(model_args)
    set_flash_attention(flash_enabled)
    model = build_model(model_args, device).train()
    optimizer_args = SimpleNamespace(
        opt=args.opt,
        lr=args.lr,
        weight_decay=args.weight_decay,
        opt_eps=args.opt_eps,
        opt_betas=tuple(args.opt_betas),
        momentum=0.9,
    )
    optimizer = create_optimizer(optimizer_args, model)
    scaler = torch.amp.GradScaler('cuda', enabled=args.amp_dtype == 'fp16')
    mask_generator = MaskGenerator(
        input_shape=(args.T, args.H, args.W),
        patch_size=(args.patch_h, args.patch_w),
        en_mask_rate=encoder_mask_rate,
        de_mask_rate=decoder_mask_rate,
        en_mask_type=args.encoder_mask_type,
        de_mask_type=args.decoder_mask_type,
        tubelet_size=args.tubelet,
    )

    dlr = None
    dlr_optimizer = None
    if args.with_dlr:
        dlr = DistributionAwareLossReweighting(
            total_iter=max(1, args.steps),
            left=0.25,
            right=0.75,
            with_ratio=True,
            momentum=0.99,
        ).to(device)
        dlr_optimizer = torch.optim.AdamW(dlr.dual_flow.online_flow.parameters(), lr=1e-3)

    elapsed = []
    try:
        total_steps = args.warmup_steps + args.steps
        for step in range(total_steps):
            torch.cuda.synchronize()
            start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            x = resize_batch(input_batch, batch_size)
            if tuple(x.shape[2:]) != (args.T, args.H, args.W):
                raise ValueError(f'input spatial shape {tuple(x.shape[2:])} does not match T/H/W {(args.T, args.H, args.W)}')
            en_mask, de_mask = mask_generator.generate_batch_masks(batch_size, device=device)
            overlap = ((~de_mask) & (~en_mask)).sum()
            if overlap.item() != 0:
                raise RuntimeError(
                    f'Decoder targets include {int(overlap.item())} encoder-visible patches.')
            with torch.amp.autocast(device_type='cuda', dtype=amp_dtype):
                pred = model(x, en_mask, de_mask)
                patches = patchify_video(x, args.patch_h, args.patch_w, args.tubelet)
                target = patches[~de_mask].reshape(batch_size, -1, patches.shape[-1])
                if dlr is None:
                    loss = F.mse_loss(pred, target)
                else:
                    loss, _ = dlr_weighted_mse(pred, target, dlr, dlr_optimizer, step)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            torch.cuda.synchronize()
            if step >= args.warmup_steps:
                elapsed.append(time.perf_counter() - start)
    except RuntimeError as exc:
        if 'out of memory' not in str(exc).lower():
            raise
        torch.cuda.empty_cache()
        return {'status': 'oom'}

    return {
        'status': 'ok',
        'seconds': sum(elapsed) / len(elapsed),
        'peak_gb': torch.cuda.max_memory_allocated() / 1024 ** 3,
    }


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    input_batch = load_input_batch(args.input_path, device)
    rows = []
    for encoder_mask_rate in args.encoder_mask_rates:
        for decoder_mask_rate in args.decoder_mask_rates:
            for batch_size in args.batches:
                for flash_enabled in [True, False]:
                    result = run_case(
                        args,
                        input_batch,
                        batch_size,
                        flash_enabled,
                        encoder_mask_rate,
                        decoder_mask_rate,
                    )
                    arch = resolve_arch_config(args)
                    row = {
                        'model': 'explicit-vit',
                        'encoder_embed_dim': arch.encoder_embed_dim,
                        'encoder_depth': arch.encoder_depth,
                        'encoder_num_heads': arch.encoder_num_heads,
                        'decoder_embed_dim': arch.decoder_embed_dim,
                        'decoder_depth': arch.decoder_depth,
                        'decoder_num_heads': arch.decoder_num_heads,
                        'encoder_mask_rate': encoder_mask_rate,
                        'decoder_mask_rate': decoder_mask_rate,
                        'batch_size': batch_size,
                        'flash_attention': flash_enabled,
                        'status': result['status'],
                        'seconds_per_step': result.get('seconds', ''),
                        'peak_memory_gb': result.get('peak_gb', ''),
                        'amp_dtype': args.amp_dtype,
                        'with_dlr': args.with_dlr,
                        'optimizer': args.opt,
                    }
                    rows.append(row)
                    print(row)
    with open(args.output, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f'wrote {args.output}')


if __name__ == '__main__':
    main()
