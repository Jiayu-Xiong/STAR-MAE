# Author: Jiayu Xiong
# Reference: STAR-MAE / Distribution-aware Loss Reweighting (DLR), PR-D-25-06353_R2.
import torch
import numpy as np


class FinetuneMaskGenerator:
    def __init__(self, input_shape, patch_size, en_mask_rate, tubelet_size=1):
        self.T, self.H, self.W = input_shape
        self.patch_size_h, self.patch_size_w = patch_size
        self.tubelet_size = tubelet_size
        self.en_mask_rate = en_mask_rate

        # Calculate the number of patches in each dimension
        self.T_patches = self.T // self.tubelet_size
        self.H_patches = self.H // self.patch_size_h
        self.W_patches = self.W // self.patch_size_w
        self.N = self.T_patches * self.H_patches * self.W_patches
        self.en_mask = DiagonalMaskingGenerator((self.T_patches, self.H_patches, self.W_patches), self.en_mask_rate)

    def generate_masks(self):
        return self.en_mask()


class DiagonalMaskingGenerator:
    """
    A class for masking based on diagonals.
    When (x + y) mod step = 0, the mask value is set to 1, otherwise 0.
    The same strategy is repeated along the time dimension (number of frames).
    """

    def __init__(self, input_size, step_factor):
        """
        Args:
            input_size (tuple or int): (frames, height, width) or a single number
            step_factor (float): Must be one of {0.5, 0.75, 0.875}
                                  Corresponding to step = 2, 4, 8 respectively
        """
        # If input is int, automatically expand to 3D (frames, height, width)
        if not isinstance(input_size, tuple):
            input_size = (input_size, ) * 3
        self.frames, self.height, self.width = input_size

        # Convert step_factor to step as per the requirement
        valid_factors = {0.5: 2, 0.75: 4, 0.875: 8}
        if step_factor not in valid_factors:
            raise ValueError(
                f"step_factor must be one of {list(valid_factors.keys())}, but got {step_factor}"
            )
        self.step = valid_factors[step_factor]

        # Calculate the total number of patches in the entire video block (frames × height × width)
        self.num_patches = self.frames * self.height * self.width

    def __repr__(self):
        return f"DiagonalMaskingGenerator(frames={self.frames}, " \
               f"height={self.height}, width={self.width}, step={self.step})"

    def __call__(self):
        """
        Returns:
            mask (np.ndarray): shape = [frames * height * width]
                               Positions where (x + y) mod step == 0 are 1, others are 0.
        """

        # ========== 1. Generate 2D diagonal mask ==========

        # Method 2: Vectorized (recommended, more concise)
        xx, yy = np.mgrid[0:self.height, 0:self.width]
        # Positions where (xx + yy) mod step == 0 are True (1), else False (0)
        mask_2d = ((xx + yy) % self.step == 0).astype(np.int32)

        # ========== 2. Replicate this 2D mask along the time dimension ==========

        # Obtain a 3D mask of shape (frames, height, width)
        # np.tile: replicate mask_2d, (frames,1,1) means only replicate along the 0th dimension frames times
        mask_3d = np.tile(mask_2d, (self.frames, 1, 1))

        # ========== 3. Flatten to 1D and return ==========

        # shape -> (frames * height * width,)
        mask = mask_3d.reshape(-1)

        return mask


# ----------------  Example Test Case  ----------------
if __name__ == "__main__":
    generator = DiagonalMaskingGenerator(input_size=(2, 4, 4), step_factor=0.5)
    print(generator)
    mask = generator()
    print("Generated mask shape:", mask.shape)
    print("Mask array (flattened):\n", mask)

    # If you want to view it in 3D
    mask_3d = mask.reshape(2, 4, 4)
    print("Mask array (reshaped to 3D):\n", mask_3d)


class MaskGenerator:
    def __init__(self, input_shape, patch_size, en_mask_rate, de_mask_rate,
                 en_mask_type='random', de_mask_type='random', tubelet_size=1):
        self.T, self.H, self.W = input_shape
        self.patch_size_h, self.patch_size_w = patch_size
        self.tubelet_size = tubelet_size
        self.en_mask_rate = en_mask_rate
        self.de_mask_rate = de_mask_rate
        self.en_mask_type = en_mask_type
        self.de_mask_type = de_mask_type

        # Calculate the number of patches in each dimension
        self.T_patches = self.T // self.tubelet_size
        self.H_patches = self.H // self.patch_size_h
        self.W_patches = self.W // self.patch_size_w
        self.N = self.T_patches * self.H_patches * self.W_patches
        if self.en_mask_type == 'random':
            self.en_mask = RandomMaskingGenerator((self.T_patches, self.H_patches, self.W_patches), self.en_mask_rate)
        elif self.en_mask_type == 'tube':
            self.en_mask = TubeMaskingGenerator((self.T_patches, self.H_patches, self.W_patches), self.en_mask_rate)
        if self.de_mask_type == 'random':
            self.de_mask = RandomMaskingGenerator((self.T_patches, self.H_patches, self.W_patches), self.de_mask_rate)
        elif self.de_mask_type == 'cell':
            self.de_mask = RunningCellMaskingGenerator((self.T_patches, self.H_patches, self.W_patches), self.de_mask_rate)

    def generate_masks(self):
        return self.en_mask(), self.de_mask()

    def generate_batch_masks(self, batch_size, device=None):
        """Return masks with explicit STAR-MAE semantics.

        en_mask=True means the patch is hidden from the encoder.
        de_mask=True means the patch is not reconstructed by the decoder.
        Therefore, decoder reconstruction targets are ~de_mask and are sampled
        only from encoder-hidden patches, so (~de_mask) & (~en_mask) is empty.
        """
        en_mask = self._generate_batch_mask(
            self.en_mask_type, self.en_mask_rate, batch_size, device)
        de_mask = self._generate_decoder_target_mask(en_mask, batch_size, device)
        self.assert_decoder_targets_not_encoder_visible(en_mask, de_mask)
        return en_mask, de_mask

    def _generate_decoder_target_mask(self, en_mask, batch_size, device=None):
        target_count = min(int(self.de_mask_rate * self.N), int(en_mask.sum(dim=1).min().item()))
        de_mask = torch.ones(batch_size, self.N, dtype=torch.bool, device=device)
        if target_count <= 0:
            return de_mask
        if self.de_mask_type == 'random':
            for batch_idx in range(batch_size):
                candidates = en_mask[batch_idx].nonzero(as_tuple=False).flatten()
                selected = candidates[torch.randperm(candidates.numel(), device=candidates.device)[:target_count]]
                de_mask[batch_idx, selected] = False
            return de_mask
        if self.de_mask_type == 'cell':
            cell_masks = [
                torch.from_numpy(self.de_mask()).bool().to(device=device)
                for _ in range(batch_size)
            ]
            cell_targets = ~torch.stack(cell_masks, dim=0)
            target_candidates = cell_targets & en_mask
            for batch_idx in range(batch_size):
                candidates = target_candidates[batch_idx].nonzero(as_tuple=False).flatten()
                if candidates.numel() < target_count:
                    candidates = en_mask[batch_idx].nonzero(as_tuple=False).flatten()
                selected = candidates[torch.randperm(candidates.numel(), device=candidates.device)[:target_count]]
                de_mask[batch_idx, selected] = False
            return de_mask
        raise ValueError(f'Unsupported decoder mask type: {self.de_mask_type}')

    @staticmethod
    def assert_decoder_targets_not_encoder_visible(en_mask, de_mask):
        overlap = (~de_mask) & (~en_mask)
        if overlap.any():
            raise RuntimeError(
                f'Decoder targets include {int(overlap.sum().item())} encoder-visible patches.')

    def _generate_batch_mask(self, mask_type, mask_rate, batch_size, device=None):
        if mask_type == 'random':
            return self._random_mask(batch_size, self.N, int(mask_rate * self.N), device)
        if mask_type == 'tube':
            spatial = self.H_patches * self.W_patches
            num_mask = int(mask_rate * spatial)
            mask_per_frame = self._random_mask(batch_size, spatial, num_mask, device)
            return mask_per_frame.repeat(1, self.T_patches).reshape(batch_size, self.N)
        if mask_type == 'cell':
            masks = [torch.from_numpy(self.de_mask()).bool() for _ in range(batch_size)]
            return torch.stack(masks, dim=0).to(device=device)
        raise ValueError(f'Unsupported mask type: {mask_type}')

    @staticmethod
    def _random_mask(batch_size, num_patches, num_mask, device=None):
        noise = torch.rand(batch_size, num_patches, device=device)
        ids = noise.argsort(dim=1)
        mask = torch.zeros(batch_size, num_patches, dtype=torch.bool, device=device)
        mask.scatter_(1, ids[:, :num_mask], True)
        return mask


class Cell():

    def __init__(self, num_masks, num_patches):
        self.num_masks = num_masks
        self.num_patches = num_patches
        self.size = num_masks + num_patches
        self.queue = np.hstack([np.ones(num_masks), np.zeros(num_patches)])
        self.queue_ptr = 0

    def set_ptr(self, pos=-1):
        self.queue_ptr = np.random.randint(self.size) if pos < 0 else pos

    def get_cell(self):
        cell_idx = (np.arange(self.size) + self.queue_ptr) % self.size
        return self.queue[cell_idx]

    def run_cell(self):
        self.queue_ptr += 1


class RandomMaskingGenerator:

    def __init__(self, input_size, mask_ratio):
        if not isinstance(input_size, tuple):
            input_size = (input_size, ) * 3

        self.frames, self.height, self.width = input_size

        self.num_patches = self.frames * self.height * self.width  # e.g., 8x14x14
        self.num_mask = int(mask_ratio * self.num_patches)

    def __repr__(self):
        repr_str = "Mask: total patches {}, mask patches {}".format(
            self.num_patches, self.num_mask)
        return repr_str

    def __call__(self):
        mask = np.hstack([
            np.zeros(self.num_patches - self.num_mask),
            np.ones(self.num_mask),
        ])
        np.random.shuffle(mask)
        return mask  # [196*8]


class TubeMaskingGenerator:

    def __init__(self, input_size, mask_ratio):
        self.frames, self.height, self.width = input_size
        self.num_patches_per_frame = self.height * self.width  # e.g., 14x14
        self.total_patches = self.frames * self.num_patches_per_frame
        self.num_masks_per_frame = int(mask_ratio * self.num_patches_per_frame)
        self.total_masks = self.frames * self.num_masks_per_frame

    def __repr__(self):
        repr_str = "Tube Masking: total patches {}, mask patches {}".format(
            self.total_patches, self.total_masks)
        return repr_str

    def __call__(self):
        mask_per_frame = np.hstack([
            np.zeros(self.num_patches_per_frame - self.num_masks_per_frame),
            np.ones(self.num_masks_per_frame),
        ])
        np.random.shuffle(mask_per_frame)
        mask = np.tile(mask_per_frame, (self.frames, 1))
        flattened_mask = mask.flatten()
        return flattened_mask  # [196*8]


class RunningCellMaskingGenerator:

    def __init__(self, input_size, mask_ratio=0.5):
        self.frames, self.height, self.width = input_size
        self.mask_ratio = mask_ratio

        num_masks_per_cell = int(4 * self.mask_ratio)
        assert 0 < num_masks_per_cell < 4
        num_patches_per_cell = 4 - num_masks_per_cell

        self.cell = Cell(num_masks_per_cell, num_patches_per_cell)
        self.cell_size = self.cell.size

        mask_list = []
        for ptr_pos in range(self.cell_size):
            self.cell.set_ptr(ptr_pos)
            mask = []
            for _ in range(self.frames):
                self.cell.run_cell()
                mask_unit = self.cell.get_cell().reshape(2, 2)
                mask_map = np.tile(mask_unit,
                                   [self.height // 2, self.width // 2])
                mask.append(mask_map.flatten())
            mask = np.stack(mask, axis=0)
            mask_list.append(mask)
        self.all_mask_maps = np.stack(mask_list, axis=0)

    def __repr__(self):
        repr_str = f"Running Cell Masking with mask ratio {self.mask_ratio}"
        return repr_str

    def __call__(self):
        mask = self.all_mask_maps[np.random.randint(self.cell_size)]
        flattened_mask = mask.flatten()
        return np.copy(flattened_mask)
