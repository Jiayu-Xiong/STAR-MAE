import torch
import numpy as np
import einops
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
from datasets.AudioSet import get_eval_set
from utils.mask_generator import MaskGenerator
from models.modeling_pretrain import pretrain_videomae_small_patch16_224, pretrain_videomae_base_patch16_224
from datasets.ESC50 import get_ESC50_dataloader
from datasets.AudioSet_FT import get_dataset_20K
from datasets.SpeechCommandCustom import create_speechcommands_dataloaders
from datasets.Opera import get_opera_dataloader


def load_model(checkpoint_path, model, device):
    """
    Load and return a pretrained model.
    :param checkpoint_path: str, path to the trained model's weights file
    :param model: initialized model
    :param device: torch.device
    :return: model with loaded weights (before eval)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint, strict=True)
    return model

import os
import torch
import einops
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches  # Alias for patches module to avoid naming conflicts

@torch.no_grad()
def visualize_single_sample(
    model,
    data_loader,
    device,
    mask_generator,
    audio_conf
):
    """
    Extract a single sample (B=1) from the data_loader, perform forward inference,
    apply inverse transformations to both the reconstruction result and the original input,
    and save them as two separate images:
      - recon_frame_{t_i}.pdf
      - gt_frame_{t_i}.pdf
    Additionally, draw white rectangles around the patches encoded by the Encoder on both images.
    """
    model.eval()

    # Extract a batch (with only 1 sample)
    x, _ = next(iter(data_loader))  # x.shape: (1, T, H, W)
    batch_size = x.size(0)  # Should be 1

    print(f"[Info] Retrieved single sample x.shape = {x.shape}")

    # Generate encoder/decoder masks
    en_mask, de_mask = mask_generator.generate_masks()
    # If mask_generator returns (total_patches,) instead of (B, total_patches), tile it
    en_mask = np.tile(en_mask, (batch_size, 1))
    de_mask = np.tile(de_mask, (batch_size, 1))

    # Convert to torch.bool
    en_mask = torch.tensor(en_mask, dtype=torch.bool, device=device)
    de_mask = torch.tensor(de_mask, dtype=torch.bool, device=device)

    # (B=1, 1, T, H, W)
    x = x.unsqueeze(1).to(device)

    # Forward inference to get the reconstructed masked parts
    reconstructed = model(x, en_mask, de_mask)  # shape: (B*N_masked, patch_dim)

    # Split the original image into patches: (B, total_patches, patch_dim)
    patches_ = einops.rearrange(
        x,
        'b c t (h p1) (w p2) -> b (t h w) (c p1 p2)',
        p1=audio_conf['patch_h'],
        p2=audio_conf['patch_w']
    )

    # Reshape reconstructed to (B, N_masked, patch_dim)
    reconstructed = reconstructed.view(batch_size, -1, patches_.shape[-1])

    # Prepare a tensor with the same shape to hold the "fully reconstructed" patches
    full_recon_patches = patches_.clone()
    # Replace the patches masked by the decoder with the reconstructed results
    full_recon_patches[~de_mask] = reconstructed.view(-1, patches_.shape[-1])

    # Now full_recon_patches is the fully reconstructed (B, total_patches, patch_dim)
    # Reshape back to (B, 1, T, H, W)
    full_recon = einops.rearrange(
        full_recon_patches,
        'b (t h w) (c p1 p2) -> b c t (h p1) (w p2)',
        t=audio_conf['T'],
        h=audio_conf['H'] // audio_conf['patch_h'],
        w=audio_conf['W'] // audio_conf['patch_w'],
        p1=audio_conf['patch_h'],
        p2=audio_conf['patch_w']
    )

    # Remove the channel dimension => (B=1, T, H, W)
    full_recon = full_recon.squeeze(1)
    x_orig = x.squeeze(1)

    # ==============
    # Apply Inverse Transformations ( +mean then * 2*std )
    # ==============
    mean_ = audio_conf['mean']
    std_ = audio_conf['std']
    # Inverse transform for the original image
    x_inv = (x_orig + mean_) * (2 * std_)
    # Inverse transform for the reconstruction result
    recon_inv = (full_recon + mean_) * (2 * std_)

    # ==============
    # Frame-by-Frame Visualization and Saving
    # ==============
    n_patches_per_frame = (audio_conf['H'] // audio_conf['patch_h']) * (audio_conf['W'] // audio_conf['patch_w'])
    T = x_inv.shape[1]

    # Create the directory if it does not exist
    os.makedirs("recon", exist_ok=True)

    for t_i in range(T):
        # Extract the current frame data (H, W)
        orig_frame = x_inv[0, t_i].cpu().numpy()
        recon_frame = recon_inv[0, t_i].cpu().numpy()

        # Compute the encoder mask for the current frame
        frame_en_mask = en_mask[0, t_i*n_patches_per_frame : (t_i+1)*n_patches_per_frame]
        # Decoder mask can also be retrieved if needed
        frame_de_mask = de_mask[0, t_i*n_patches_per_frame : (t_i+1)*n_patches_per_frame]

        # Number of patch rows and columns
        n_patch_cols = audio_conf['W'] // audio_conf['patch_w']
        n_patch_rows = audio_conf['H'] // audio_conf['patch_h']

        # --------------------------------------------------
        # 1) Draw the [Reconstructed Frame] and save as recon_frame_{t_i}.pdf
        # --------------------------------------------------
        fig_recon, ax_recon = plt.subplots(
            nrows=1, ncols=1,
            figsize=(6, 6),  # Ensure it's square
            dpi=100
        )
        ax_recon.imshow(recon_frame, origin='lower', aspect='equal', cmap='magma')
        ax_recon.axis('off')  # Remove axes

        # Draw white rectangles around "encoder encoded patches"
        # Assume en_mask==False => not masked => encoder visible => need to draw white rectangles
        for patch_idx in range(n_patches_per_frame):
            if not frame_en_mask[patch_idx]:
                patch_row = patch_idx // n_patch_cols
                patch_col = patch_idx % n_patch_cols
                left = patch_col * audio_conf['patch_w']
                bottom = patch_row * audio_conf['patch_h']
                rect = mpatches.Rectangle(
                    (left, bottom),
                    audio_conf['patch_w'],
                    audio_conf['patch_h'],
                    linewidth=1,
                    edgecolor='white',
                    facecolor='none'
                )
                ax_recon.add_patch(rect)

        save_path_recon = f"recon/recon_frame_{t_i}.pdf"
        plt.savefig(save_path_recon, bbox_inches='tight', pad_inches=0.0)
        plt.close(fig_recon)
        print(f"[Info] Saved reconstructed frame: {save_path_recon}")

        # --------------------------------------------------
        # 2) Draw the [Ground Truth Frame] and save as gt_frame_{t_i}.pdf
        # --------------------------------------------------
        fig_gt, ax_gt = plt.subplots(
            nrows=1, ncols=1,
            figsize=(6, 6),  # Also square
            dpi=100
        )
        ax_gt.imshow(orig_frame, origin='lower', aspect='equal', cmap='magma')
        ax_gt.axis('off')

        # Similarly, draw white rectangles around encoder encoded patches on the original image
        for patch_idx in range(n_patches_per_frame):
            if False: # not frame_en_mask[patch_idx]:  
                patch_row = patch_idx // n_patch_cols
                patch_col = patch_idx % n_patch_cols
                left = patch_col * audio_conf['patch_w']
                bottom = patch_row * audio_conf['patch_h']
                rect = mpatches.Rectangle(
                    (left, bottom),
                    audio_conf['patch_w'],
                    audio_conf['patch_h'],
                    linewidth=1,
                    edgecolor='white',
                    facecolor='none'
                )
                ax_gt.add_patch(rect)

        save_path_gt = f"recon/gt_frame_{t_i}.pdf"
        plt.savefig(save_path_gt, bbox_inches='tight', pad_inches=0.0)
        plt.close(fig_gt)
        print(f"[Info] Saved GT frame: {save_path_gt}")



if __name__ == "__main__":
    audio_conf = {
        'sample_rate': 16000,
        'n_fft': 2048,
        'H': 256,
        'W': 256,
        'T': 4,
        'overlap_rate': 0.0,
        'per_frame_overlap': False,
        'use_mel': True,
        'hop_length': None,
        'mixup': 0.0,
        'mean': -4.268,
        'std': 4.569,
        'skip_norm': False,
        'noise': False,
        'patch_h': 16,
        'patch_w': 16,
        'tubelet': 1,
        'batchsize': 1,  # Key: BatchSize = 1
        'workers': 4,
        'root': '/path',
        'encoder_msk_rate': 0.65, 
        'decoder_msk_rate': 0.0,
        'encoder_msk_type': 'tube',
        'decoder_msk_type': 'random'
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # _, test_loader = get_eval_set(audio_conf=audio_conf)
    audio_conf = {
    'sample_rate': 16000,
    'n_fft': 2048,
    'H': 256,
    'W': 256,
    'T': 2,
    'overlap_rate': 0.0,
    'per_frame_overlap': False,
    'use_mel': True,
    'hop_length': 160,
    'mixup': 0.0,
    'skip_norm': False,
    'noise': False,
    'patch_h': 16,
    'patch_w': 16,
    'tubelet': 1,
    'batchsize': 1,
    'workers': 24,
    'mean': -5.402,
    'std': 5.971,
    'root': '/path',
    'encoder_msk_rate': 0.85, 
    'decoder_msk_rate': 0.0,
    'encoder_msk_type': 'tube',
    'decoder_msk_type': 'random',
    'num_classes': 50
    }
    # train_loader, test_loader =  get_ESC50_dataloader(audio_conf, 1)

    audio_conf = {
        'sample_rate': 16000,
        'n_fft': 2048,
        'H': 128,
        'W': 128,
        'T': 4,
        'overlap_rate': 0.0,
        'per_frame_overlap': False,
        'use_mel': True,
        'hop_length': 160,
        'mixup': 1.0,
        'mean': -10.032,
        'std': 6.480,
        'skip_norm': False,
        'noise': True,
        'patch_h': 16,
        'patch_w': 16,
        'tubelet': 1,
        'batchsize': 1,
        'workers': 24,
        'root': '/path',
        'encoder_msk_rate': 0.85, 
        'decoder_msk_rate': 0.0,
        'encoder_msk_type': 'tube',
        'decoder_msk_type': 'random',
        'num_classes': 4
    }
    test_loader, _ = get_opera_dataloader(audio_conf, 0)
    # Prepare MaskGenerator
    from utils.mask_generator import MaskGenerator
    np.random.seed(1)
    mask_generator = MaskGenerator(
        input_shape=(audio_conf['T'], audio_conf['H'], audio_conf['W']),
        patch_size=(audio_conf['patch_h'], audio_conf['patch_w']),
        en_mask_rate=audio_conf['encoder_msk_rate'],
        de_mask_rate=audio_conf['decoder_msk_rate'],
        en_mask_type=audio_conf['encoder_msk_type'],
        de_mask_type=audio_conf['decoder_msk_type'],
        tubelet_size=audio_conf['tubelet']
    )

    cls = audio_conf['patch_h'] * audio_conf['patch_w'] * audio_conf['tubelet']
    model = pretrain_videomae_base_patch16_224(
        in_chans=1,
        all_frames=audio_conf['T'],
        img_size=(audio_conf['H'],audio_conf['W']),
        patch_size=audio_conf['patch_h'],
        tubelet_size=audio_conf['tubelet'],
        decoder_num_classes=cls
    ).to(device)

    checkpoint_path = "pth"
    model = load_model(checkpoint_path, model, device)


    visualize_single_sample(model, test_loader, device, mask_generator, audio_conf)
