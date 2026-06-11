# Architecture

## 1. System overview

```
                        ┌────────────────────────────────────────────┐
                        │              AdaIN-CycleGAN                │
                        │                                            │
 content image (A) ──▶  │  G_A (AdaIN generator)  ──▶ fake_B         │ ──▶ D_A (real/fake in B?)
                        │        ▲                                   │
 style exemplar  ──▶    │  StyleEncoder (VGG19, frozen)              │
                        │        │ style features (B, 128, h, w)     │
                        │        ▼                                   │
 content image (B) ──▶  │  G_B (AdaIN generator)  ──▶ fake_A         │ ──▶ D_B (real/fake in A?)
                        └────────────────────────────────────────────┘
```

Four trainable networks plus one frozen encoder:

| Network | File | Role |
|---|---|---|
| `G_A`, `G_B` | `models/adain_cycle_gan.py` (`AdaINGenerator`) | ResNet-style encoder–decoder translators with AdaIN residual blocks |
| `D_A`, `D_B` | `models/adain_cycle_gan.py` (`Discriminator`) | 70×70 PatchGAN discriminators |
| `StyleEncoder` | `models/adain.py` | First 8 layers of pre-trained VGG19 (through `conv2_2`, 128 channels), frozen |

## 2. The AdaIN generator

Pipeline inside `AdaINGenerator`:

1. **Downsampling head** (`model_down`): 7×7 conv → two stride-2 convs (`ngf` → `4·ngf` channels, spatial /4)
2. **9 AdaIN residual blocks** (`AdaINResBlock`): each block runs `pad → conv → AdaIN → ReLU → pad → conv → AdaIN` plus a residual connection
3. **Upsampling tail** (`model_up`): two transposed convs → 7×7 conv → `tanh`

### Style injection

Each `AdaINResBlock` owns a **learned 1×1 projection** (`style_proj`) that maps the 128-channel VGG style features to the block's channel count (256). The projection is registered in `__init__`, so it is trained jointly with the generator. (An earlier version created a *random* conv inside `forward` on every call — effectively noise injection; see [04_fixes_and_changes.md](04_fixes_and_changes.md).)

The `AdaIN` layer itself is parameter-free:

```
AdaIN(c, s) = σ(s) · (c − μ(c)) / (σ(c) + ε) + μ(s)
```

Statistics are channel-wise over spatial dimensions. When a batch element has **N style exemplars** (shape `[B, N, C, H, W]`), per-exemplar statistics are computed first and then averaged over N, so the style statistics always align with the content batch dimension.

### Style weight (α) — feature-space interpolation

The user-facing style weight is applied **inside** the AdaIN layers, following the AdaIN paper:

```
output = c + α · (AdaIN(c, s) − c)
```

- `α = 0` → content statistics untouched
- `α = 1` → full style statistics
- `α > 1` → extrapolation (stronger stylization)

This replaces the earlier pixel-space blend `(1−w)·content + w·stylized`, which only cross-fades two images and does not actually modulate the transfer.

## 3. Loss formulation

All losses live in one place (`backward_G` / `backward_D_basic`), used by `optimize_parameters`:

| Loss | Definition | Weight |
|---|---|---|
| Adversarial (LSGAN) | `criterionGAN` = MSE against real/fake targets | 1 |
| Cycle consistency | `criterionCycle` = L1(`rec_A`, `real_A`), L1(`rec_B`, `real_B`) | `λ_A`, `λ_B` = 10 |
| Identity | `criterionIdt` = L1(`G_A(real_B)`, `real_B`), L1(`G_B(real_A)`, `real_A`) | `λ_idt` = 0.5 (× `λ_A/B`) |
| Style (AdaIN) | `criterionStyle` = MSE of VGG feature mean + std between `fake_B` and the exemplar | `λ_style` = 1 |

Discriminators use the standard `0.5 · (loss_real + loss_fake)` LSGAN objective.

### Well-defined cycle consistency with arbitrary styles

Naive AdaIN-CycleGAN reuses the exemplar's style features for *all four* generator calls, which makes the reconstruction target ambiguous. We instead encode three style feature sets per iteration:

```python
style_features = E(style_img)   # the exemplar
style_A        = E(real_A)      # source A's own appearance
style_B        = E(real_B)      # source B's own appearance

fake_B = G_A(real_A, style_features)   # stylize with the exemplar
rec_A  = G_B(fake_B, style_A)          # reconstruct A with A's own statistics
fake_A = G_B(real_B, style_A)
rec_B  = G_A(fake_A, style_B)          # reconstruct B with B's own statistics
```

The reverse mapping is always told *which appearance to restore*, so `||rec_A − real_A||₁` is a coherent objective even though the forward style is arbitrary. The exemplar's influence on `fake_B` is enforced separately by the style loss.

## 4. Training step (`optimize_parameters`)

One unified path per iteration:

1. `forward()` — compute all fakes and reconstructions
2. Freeze `D_A`, `D_B` → `backward_G()` → clip generator grads (max-norm 1.0) → `optimizer_G.step()`
3. Unfreeze discriminators → `backward_D_A()`, `backward_D_B()` → clip → `optimizer_D.step()`

## 5. Inference paths

- **CLI** (`demo.py`): loads `netG_A` weights from a checkpoint, encodes the style image once, runs `G_A(content, style, α)`
- **Web app** (`app.py`): same pipeline behind Flask; checkpoint path from `CHECKPOINT_PATH` env var (default `./checkpoints/adain_cyclegan/latest_net.pth`); the style-weight slider maps directly to α
- **Evaluation** (`test.py`): batch inference + PSNR/SSIM/FID, see [05_evaluation_and_deployment.md](05_evaluation_and_deployment.md)

## 6. Data pipeline

`data/datasets.py` → `UnalignedStyleDataset` yields per item:

- `A`: random domain-A image, `B`: random domain-B image (unpaired)
- `style`: `n_style_images` exemplars stacked to `[N, 3, H, W]` (batched → `[B, N, 3, H, W]`)

Images are resized to `load_size` (286), random-cropped to `crop_size` (256), flipped, and normalized to `[−1, 1]`.
