# STAR-MAE ⭐

Official implementation for **Masked Autoencoders for Spatio-Temporal Audio Representations: Theory and Optimization**.

STAR-MAE learns audio representations with masked autoencoding over spatio-temporally stacked spectrogram frames. The method uses structured masking to reduce large missing regions in audio reconstruction and introduces **Distribution-aware Loss Reweighting (DLR)** to down-weight noisy extremes while emphasizing informative reconstruction signals.

## ✨ Highlights

- Spatio-temporal audio masking for more reliable reconstruction at high masking ratios.
- Distribution-aware Loss Reweighting (DLR), a lightweight training-only module for stabilizing reconstruction loss.
- Pre-training recipe for AudioSet-style metadata and NPZ spectrogram features.
- Validation and benchmark scripts for mask behavior, DLR gradients, and batch/FlashAttention capacity.

## 📁 Repository Layout

```text
.
|-- train_star_mae_dlr.py        # STAR-MAE pre-training entry point
|-- dlr/                         # Distribution-aware Loss Reweighting modules
|-- models/                      # Pre-training and fine-tuning model definitions
|-- datasets/                    # AudioSet-style dataset readers and NPZ preprocessing
|-- utils/                       # Mask generation and audio feature helpers
|-- benchmarks/                  # Validation and local benchmark scripts
|-- figures/                     # Figure scripts and reconstruction PDFs
`-- checkpoints/                 # Reserved for released weights
```

## 🛠️ Installation

Create your own Python environment, then install the packages used by the code:

```bash
pip install torch torchvision torchaudio timm numpy scipy pandas matplotlib seaborn tqdm tensorboard
```

Use the PyTorch install command that matches your CUDA/CPU platform from the official PyTorch website.

## 📦 Data Preparation

This repository does not contain datasets or local feature caches. Prepare your own AudioSet-style files:

- Metadata CSV, for example `<DATASET_ROOT>/un_train_index_cleaned.csv`
- Label index CSV, for example `<DATASET_ROOT>/class_labels_indices.csv`
- WAV root or precomputed NPZ root

To preprocess WAV files into NPZ features:

```bash
python datasets/audio_npz.py \
  --csv <PATH_TO_METADATA_CSV> \
  --wav-root <PATH_TO_WAV_ROOT> \
  --root <PATH_TO_OUTPUT_NPZ_ROOT> \
  --label-csv <PATH_TO_LABEL_CSV> \
  --sample-rate 16000 \
  --mel-fmax 8000
```

## 🚀 Pre-training

Run STAR-MAE pre-training with your own paths:

```bash
python train_star_mae_dlr.py \
  --dataset-root <DATASET_ROOT> \
  --dataset-split unbal \
  --npz-root <PATH_TO_NPZ_ROOT> \
  --label-csv <PATH_TO_LABEL_CSV> \
  --output-dir <PATH_TO_OUTPUT_DIR>
```

You can also bypass `--dataset-root` and provide `--csv`, `--npz-root` or `--wav-root`, and `--label-csv` explicitly.
Please refer [AudioRWKV](https://github.com/Jiayu-Xiong/AudioRWKV)

## 🧩 Weights

Pre-trained weights are not included in this initial code package. They will be uploaded to the `checkpoints/` release area after preparation. Until then, please train from scratch with the commands above.

Planned weight slots:

| Model | Pre-training data | Link |
| --- | --- | --- |
| STAR-MAE ViT-B | AudioSet 2M | To be released |
| STAR-MAE ViT-B | Pattern-aligned subset | To be released |

## ✅ Validation

Run quick checks after installation:

```bash
python benchmarks/validate_masks.py \
  --encoder-mask-path <PATH_TO_ENCODER_MASK> \
  --decoder-mask-path <PATH_TO_DECODER_MASK>

python benchmarks/validate_dlr.py \
  --pred-path <PATH_TO_PRED_TENSOR> \
  --target-path <PATH_TO_TARGET_TENSOR> \
  --device cuda
```

Local benchmark outputs are intentionally ignored by git:

```bash
python benchmarks/benchmark_batch_flash.py \
  --input-path <PATH_TO_INPUT_TENSOR> \
  --output benchmarks/results/batch_flash.csv
```

## 📚 Citation

If this repository is useful for your work, please cite:

```bibtex
@article{xiong2026masked,
  title = {Masked autoencoders for spatio-temporal audio representations: Theory and optimization},
  author = {Xiong, Jiayu and Wang, Jing and Wang, Wanlong and Lyu, Xiaosen and Kwan, Jianlong and Xue, Jun},
  journal = {Pattern Recognition},
  volume = {175},
  pages = {113133},
  year = {2026},
  doi = {10.1016/j.patcog.2026.113133},
  url = {https://doi.org/10.1016/j.patcog.2026.113133}
}
```

## 📄 License

This project is released under the BSD 3-Clause License. See [LICENSE](LICENSE) for details.
