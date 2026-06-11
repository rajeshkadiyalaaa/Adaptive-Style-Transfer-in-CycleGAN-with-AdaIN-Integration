# Adaptive Style Transfer in CycleGAN with AdaIN Integration

[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A PyTorch implementation that combines **CycleGAN's** unpaired image-to-image translation with **Adaptive Instance Normalization (AdaIN)**: a translator to a target domain whose exact appearance is steered per-image by a style exemplar, with adversarial realism and well-defined cycle consistency.

## How It Works

The generator is a ResNet-style CycleGAN generator whose residual blocks replace instance normalization with **AdaIN**:

```
AdaIN(c, s) = σ(s) · (c − μ(c)) / (σ(c) + ε) + μ(s)
```

Style statistics come from a frozen **VGG19 encoder** (through `conv2_2`) applied to the style exemplar; each residual block maps them to its own channel count with a learned 1×1 projection.

Two design points distinguish this from a naive CycleGAN+AdaIN mashup:

1. **Well-defined cycle consistency.** The forward mapping stylizes with the exemplar, but each reverse mapping is conditioned on the *source image's own* style statistics — so the reconstruction target is unambiguous even though the forward style is arbitrary:

   ```
   fake_B = G_A(real_A, style_exemplar)      rec_A = G_B(fake_B, style(real_A))
   fake_A = G_B(real_B, style(real_A))       rec_B = G_A(fake_A, style(real_B))
   ```

2. **Feature-space style weight.** The style-strength control α is applied inside the AdaIN layers — `c + α·(AdaIN(c,s) − c)` — not as a pixel-space blend of output images. `α = 0` keeps the content untouched, `α = 1` applies the full style, `α > 1` extrapolates.

### Losses

| Loss | Definition | Default weight |
|---|---|---|
| Adversarial | LSGAN (MSE against real/fake targets) | 1 |
| Cycle consistency | L1 between reconstructions and sources | `λ_A = λ_B = 10` |
| Identity | L1 when a generator receives its own target domain | `λ_idt = 0.5` |
| Style | MSE of VGG feature mean/std vs. the exemplar | `λ_style = 1` |

## Project Structure

```
.
├── models/
│   ├── adain.py                # AdaIN layer, VGG19 StyleEncoder, AdaIN residual block
│   └── adain_cycle_gan.py      # Generators, PatchGAN discriminators, full training model
├── config/                      # argparse options (base / train / test)
├── data/
│   ├── datasets.py             # Unpaired A/B + style-exemplar datasets
│   └── download_data.py        # Dataset download helper
├── utils/
│   ├── image_pool.py           # Image buffer
│   └── metrics.py              # PSNR, SSIM, FID (LPIPS-feature based)
├── datasets/Project_dataset/    # Bundled dataset (trainA/B, testA/B, style/)
├── examples/                    # Sample content/ and styles/ images for the demo
├── templates/                   # Web UI (Flask)
├── document/                    # Full documentation (research, architecture, process, fixes)
├── Research/                    # Reference papers (CycleGAN, AdaIN, domain transfer)
├── train.py                     # Training
├── test.py                      # Evaluation (PSNR / SSIM / FID)
├── demo.py                      # CLI inference
├── app.py                       # Flask web app
└── requirements.txt
```

## Installation

```bash
git clone https://github.com/rajeshkadiyalaaa/Adaptive-Style-Transfer-in-CycleGAN-with-AdaIN-Integration.git
cd Adaptive-Style-Transfer-in-CycleGAN-with-AdaIN-Integration

python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Pure PyTorch stack — no TensorFlow. CUDA is used automatically when available; CPU works for inference.

> **Note:** no pre-trained checkpoint ships with this repository. Train a model first (below) before running the demo or web app.

## Usage

### 1. Train

The repository bundles a ready-to-use dataset at `datasets/Project_dataset` (`trainA/`, `trainB/`, `testA/`, `testB/`, `style/`):

```bash
python train.py \
  --dataroot ./datasets/Project_dataset \
  --name adain_cyclegan \
  --use_adain \
  --batch_size 4 \
  --n_epochs 100 --n_epochs_decay 100
```

Monitor with TensorBoard (losses `G_A`, `G_B`, `cycle_A/B`, `idt_A/B`, `style`, `D_A/B` plus image grids):

```bash
tensorboard --logdir ./checkpoints/adain_cyclegan/logs
```

Checkpoints are written to `./checkpoints/adain_cyclegan/` (`latest_net.pth` + per-epoch snapshots).

### 2. Command-line inference

```bash
python demo.py \
  --content examples/content/seaport.jpg \
  --style "examples/styles/oil paint.jpg" \
  --output result.jpg \
  --model ./checkpoints/adain_cyclegan/latest_net.pth \
  --style_weight 1.0
```

`--style_weight` (0.0–2.0) is the feature-space α described above.

### 3. Web interface

```bash
# optional — defaults to ./checkpoints/adain_cyclegan/latest_net.pth
export CHECKPOINT_PATH=./checkpoints/adain_cyclegan/latest_net.pth
python app.py
```

Open `http://localhost:5000`, upload (or webcam-capture) a content image, upload a style image, set the style-weight slider, and generate. The app fails fast with a clear error if the checkpoint is missing — there is intentionally no fallback model.

### 4. Evaluate

```bash
python test.py \
  --dataroot ./datasets/Project_dataset \
  --name adain_cyclegan \
  --results_dir ./results \
  --compute_metrics
```

Reports PSNR and SSIM on cycle reconstructions, and optionally FID. Caveat: the FID implementation extracts features with LPIPS/AlexNet rather than InceptionV3, so values are only comparable *between checkpoints of this project*, not with published numbers.

## Training Options

| Option | Default | Description |
|--------|---------|-------------|
| `--batch_size` | 8 | Batch size |
| `--n_epochs` / `--n_epochs_decay` | 100 / 100 | Epochs at initial LR / linear-decay epochs |
| `--lr` | 0.0002 | Adam learning rate (β₁ = 0.5) |
| `--lambda_A` / `--lambda_B` | 10.0 | Cycle-consistency weights |
| `--lambda_identity` | 0.5 | Identity loss weight |
| `--lambda_style` | 1.0 | Style loss weight |
| `--n_style_images` | 5 | Style exemplars sampled per iteration |

Full list: `python train.py --help`.

## Documentation

The `document/` folder is the canonical documentation set:

- [Research background](document/01_research.md) — CycleGAN, AdaIN, MUNIT, and the design tension this project resolves
- [Architecture](document/02_architecture.md) — networks, style injection, loss formulation, data flow
- [Build & repair process](document/03_process.md) — step-by-step engineering log
- [Fixes & changes](document/04_fixes_and_changes.md) — verification-report findings (F1–F10) mapped to code changes
- [Evaluation & deployment](document/05_evaluation_and_deployment.md) — training, metrics, and serving guide
- [CycleGAN paper notes](document/06_cyclegan_notes.md) — working notes on the original paper

## Why Combine CycleGAN and AdaIN?

- **CycleGAN alone** maps to one fixed target domain with no per-image style control.
- **AdaIN alone** performs arbitrary style transfer on unpaired data, but has no adversarial or cycle constraints, so outputs can drift from the realism of the target domain.
- **Combined**: domain-level realism from adversarial + cycle losses, plus exemplar-level style control from AdaIN.

Honest caveat: for pure arbitrary style transfer with no domain prior, AdaIN alone is simpler and sufficient. This hybrid is justified when you want both a domain prior and exemplar control — see [document/01_research.md](document/01_research.md).

## References

1. **CycleGAN** — J. Zhu, T. Park, P. Isola, A. A. Efros. *Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks.* ICCV 2017. [arXiv:1703.10593](https://arxiv.org/abs/1703.10593)
2. **AdaIN** — X. Huang, S. Belongie. *Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization.* ICCV 2017. [arXiv:1703.06868](https://arxiv.org/abs/1703.06868)
3. **MUNIT** — X. Huang, M. Liu, S. Belongie, J. Kautz. *Multimodal Unsupervised Image-to-Image Translation.* ECCV 2018. [arXiv:1804.04732](https://arxiv.org/abs/1804.04732)

## License

MIT — see [LICENSE](LICENSE).
