# Evaluation & Deployment

## 1. Training

Dataset layout (unpaired domains + a style exemplar pool):

```
datasets/your_dataset/
├── trainA/   # domain A (e.g. photos)
├── trainB/   # domain B (e.g. paintings)
├── testA/
├── testB/
└── style/    # style exemplars sampled during training
```

Run:

```bash
python train.py \
  --dataroot ./datasets/your_dataset \
  --name adain_cyclegan \
  --use_adain \
  --batch_size 4 \
  --n_epochs 100 --n_epochs_decay 100
```

Key hyperparameters (see `config/train_options.py`): `λ_A = λ_B = 10` (cycle), `λ_idt = 0.5`, `λ_style = 1`, LSGAN objective, Adam (`lr 2e-4`, `β₁ 0.5`) with linear decay over the second half of training, gradient clipping at max-norm 1.0.

Monitoring:

```bash
tensorboard --logdir ./checkpoints/adain_cyclegan/logs
```

Logged: `G_A`, `G_B`, `cycle_A`, `cycle_B`, `D_A`, `D_B`, `style`, `idt_A`, `idt_B`, image grids of `real/fake/rec` for both directions.

Checkpoints: `net_epoch_<n>.pth` every `--save_epoch_freq` epochs plus `latest_net.pth`, each containing `netG_A`, `netG_B`, `netD_A`, `netD_B`, and optimizer states.

## 2. Evaluation (`test.py`, `utils/metrics.py`)

```bash
python test.py \
  --dataroot ./datasets/your_dataset \
  --name adain_cyclegan \
  --results_dir ./results \
  --compute_metrics
```

| Metric | Measures | Direction |
|---|---|---|
| **PSNR** | pixel fidelity of cycle reconstructions (`rec` vs `real`) | higher = better |
| **SSIM** | structural similarity of reconstructions | higher = better |
| **FID** | distributional distance between generated and reference images | lower = better |

Caveats to report honestly alongside numbers:

- PSNR/SSIM here evaluate **reconstruction** (cycle) quality, not stylization quality
- The FID implementation extracts features with LPIPS/AlexNet rather than the standard InceptionV3, so values are **not comparable** with published FID numbers — use it only for relative comparison between checkpoints of this project
- Style fidelity is best judged with the style loss (VGG mean/std distance) plus qualitative inspection across a fixed grid of exemplars and α values

Suggested qualitative protocol: fix 5 content images × 5 style exemplars × α ∈ {0.25, 0.5, 1.0}, regenerate the 75-image grid for every candidate checkpoint, and compare side by side.

## 3. CLI inference

```bash
python demo.py \
  --content path/to/content.jpg \
  --style path/to/style.jpg \
  --output result.jpg \
  --model ./checkpoints/adain_cyclegan/latest_net.pth \
  --style_weight 1.0
```

## 4. Web app

```bash
# optional: point at a specific checkpoint
export CHECKPOINT_PATH=./checkpoints/adain_cyclegan/latest_net.pth
python app.py            # development
gunicorn -w 1 app:app    # production (single worker: the model is loaded per process)
```

- The app **requires** a trained checkpoint and exits with a clear error otherwise — there is intentionally no fallback model
- The style-weight slider (0–2.0) maps directly to AdaIN's α (feature-space interpolation; values above 1 extrapolate)
- Inference is CPU-capable; CUDA is used automatically when available

## 5. Reproducibility checklist

- [ ] `pip install -r requirements.txt` (PyTorch-only stack; no TensorFlow)
- [ ] dataset in the layout above (`python data/download_data.py --dataset monet2photo --download_styles` for a starter set)
- [ ] train ≥ a few epochs; confirm all nine loss curves move in TensorBoard
- [ ] `test.py --compute_metrics` produces `results/<name>/metrics.txt`
- [ ] `demo.py` and `app.py` run against the produced `latest_net.pth`
