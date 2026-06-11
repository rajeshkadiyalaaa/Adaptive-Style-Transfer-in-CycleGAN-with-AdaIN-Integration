# Fixes & Changes — Report Findings F1–F10

Each finding from the verification report ([verification_report.md](verification_report.md)) mapped to the exact change that resolves it.

## F1 — Fatal crash: undefined loss criteria

**Was:** `optimize_parameters()` referenced `self.criterionGAN`, `self.criterionCycle`, `self.criterionStyle`; none existed → `AttributeError` on the first training iteration.

**Fix (`models/adain_cycle_gan.py`):**
- Added a `GANLoss` module (LSGAN/MSE with auto-built target tensors) and instantiated `self.criterionGAN = GANLoss()`, `self.criterionCycle = nn.L1Loss()`, `self.criterionIdt = nn.L1Loss()` in `__init__`
- Implemented `criterionStyle(generated, style_features)` as a method: MSE between channel-wise mean/std of VGG features of the generated image and the exemplar

## F2 — Fatal crash: missing scipy import

**Was:** `FID.calculate_frechet_distance` called `scipy.linalg.sqrtm()` with no import.

**Fix (`utils/metrics.py`):** added `import scipy.linalg`. (`scipy` was already in `requirements.txt`.)

## F3 — Logic bug: AdaIN batch-dimension mishandling

**Was:** with multi-exemplar style input `[B, N, C, H, W]` flattened to `[B·N, C, H, W]`, the style statistics were then computed using the **content** batch size `B` → wrong `view()` / shape mismatch for `N > 1`. Additionally, channel mismatches were "fixed" by creating a **new randomly-initialized 1×1 conv inside `forward` on every call** — untrained, different each step, pure noise injection.

**Fix (`models/adain.py`):**
- Style statistics are computed per exemplar at batch size `B·N`, then averaged over `N` back to `[B, C, 1, 1]`, always aligned with the content batch
- The random conv is gone. Each `AdaINResBlock` now owns a learned `style_proj = nn.Conv2d(style_nc=128, dim, 1)` registered in `__init__`, trained jointly with the generator. `AdaIN` asserts the channels match.

## F4 — Scope bug in `StyleEncoder.forward`

**Was:** `B` and `N` were used in a condition outside the branch that defined them → `NameError`/`UnboundLocalError` for 4D inputs.

**Fix (`models/adain.py`):** a `multi_style` flag is captured before reshaping; `B`, `N` are only referenced when it is true.

## F5 — Dead code: identity loss never used

**Was:** `--lambda_identity` (default 0.5) was parsed but no identity term was computed anywhere.

**Fix (`models/adain_cycle_gan.py`, `train.py`):** `backward_G` now computes `idt_A = G_A(real_B, style_B)` and `idt_B = G_B(real_A, style_A)` (style-free variants when AdaIN is off), weights them by `λ_idt · λ_B/A`, includes both in `loss_G`, and `train.py` logs `idt_A`/`idt_B` to console and TensorBoard.

## F6 — Inconsistency: two divergent loss paths

**Was:** `backward_G` (functional, runnable) and `optimize_parameters` (criterion-based, broken) implemented different losses; only one could ever be in effect, and they disagreed.

**Fix (`models/adain_cycle_gan.py`):** single source of truth. All loss math lives in `backward_G` / `backward_D_basic` (using the criteria from F1). `optimize_parameters` only orchestrates: forward → freeze Ds → `backward_G` → clip → step G → unfreeze → `backward_D_A/B` → clip → step D.

## F7 — Wrong math: pixel-space style-weight blending

**Was:** `app.py` implemented the style slider as `(1−w)·content + w·stylized` on output pixels — a cross-fade, not style modulation.

**Fix (`models/adain.py`, `app.py`, `demo.py`):** `AdaIN.forward` accepts `alpha` and computes `c + α·(AdaIN(c,s) − c)` in **feature space**, exactly as in the AdaIN paper (with `α > 1` allowed as extrapolation). The α parameter is threaded through `AdaINResBlock` and `AdaINGenerator`; the web slider and the new `demo.py --style_weight` flag map straight to it.

## F8 — Architectural flaw: ill-defined cycle consistency

**Was:** the reverse generators were conditioned on the *exemplar's* style features while being asked to reconstruct the source image — the cycle target was ambiguous for arbitrary styles.

**Fix (`models/adain_cycle_gan.py`):** `forward()` also encodes the source images' own style features (`style_A = E(real_A)`, `style_B = E(real_B)`) and conditions every reverse mapping on them:

```
fake_B = G_A(real_A, style_exemplar)    rec_A = G_B(fake_B, style_A)
fake_A = G_B(real_B, style_A)           rec_B = G_A(fake_A, style_B)
```

The reconstruction objective is now well-posed; the exemplar's influence on `fake_B` is enforced by the separate style loss. The rationale is documented in [01_research.md](01_research.md) §2.

## F9 — False claim: "AdaIN requires paired data"

**Was:** README justified the hybrid by claiming AdaIN alone needs paired training data — incorrect (AdaIN trains on unpaired content/style sets).

**Fix (`README.md`):** rewrote "Why Combine Them?" with the accurate trade-off: AdaIN alone lacks adversarial/cycle constraints (domain realism, content guarantees); the hybrid adds those while keeping exemplar-level control. Also fixed the misattributed MUNIT citation.

## F10 — Deployment gap: app unrelated to the custom model

**Was:** `app.py` loaded a TensorFlow-Hub (Magenta) SavedModel from `models/`; the custom PyTorch code was never executed by the demo.

**Fix (`app.py`, `requirements.txt`):** the app is now pure PyTorch — it builds `AdaINGenerator` + `StyleEncoder`, loads `netG_A` weights from `CHECKPOINT_PATH` (default `./checkpoints/adain_cyclegan/latest_net.pth`), and fails fast with an actionable error if no checkpoint exists. `tensorflow` / `tensorflow-hub` were removed from `requirements.txt`, and the legacy SavedModel files (~90 MB under `models/`) were deleted during repository cleanup.

## Extra — latent autograd bug found during verification (not in the report)

**Was:** `AdaINResBlock.forward` sliced its conv block as `conv_block[3:6]` for the second convolution, but index 3 is the **in-place ReLU** module. Combined with the explicit `F.relu(out, True)` call, the activation ran twice in-place on the same tensor, raising `RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation` on the first backward pass. The report never reached this because training crashed earlier at F1.

**Fix (`models/adain.py`):** second slice corrected to `conv_block[4:7]` (skip the already-applied ReLU) and the explicit ReLU made non-in-place. Caught by the smoke test below.

---

## Verification

A structural smoke test (CPU, random VGG weights) confirmed after the fixes:

- `AdaIN.forward` with 4D and 5D style inputs and α ∈ {0, 0.5, 1, 1.5}; α = 0 returns content unchanged
- `StyleEncoder` handles 4D and 5D inputs (F4)
- `AdaINResBlock.style_proj` is registered and trainable
- A full `AdaINStyleCycleGAN.optimize_parameters()` step runs end-to-end with `B=2, N=3` style exemplars and produces finite, non-zero values for all nine losses — including `idt_A/idt_B` (F5) and `style` (F1)

All modified files pass `py_compile`.

---

## Files changed

| File | Change summary |
|---|---|
| `models/adain.py` | F3, F4; learned style projection; α interpolation |
| `models/adain_cycle_gan.py` | F1, F5, F6, F8; `GANLoss`; α pass-through in generator |
| `utils/metrics.py` | F2 |
| `app.py` | F7, F10 (rewritten on PyTorch) |
| `demo.py` | `--style_weight` flag (README parity) |
| `train.py` | identity-loss logging |
| `README.md` | F9; checkpoint requirement; MUNIT citation |
| `requirements.txt` | removed TensorFlow deps |
| `document/` | this documentation set (new) |
