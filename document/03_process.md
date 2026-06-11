# Build & Repair Process

This is the step-by-step record of how the project was brought from "looks polished but cannot run" (per the verification report in [verification_report.md](verification_report.md)) to an internally consistent implementation.

## Step 0 — Audit

**Goal:** reproduce every finding in the verification report before touching code.

Read all model, pipeline, config, and data files. Confirmed all ten findings (F1–F10):

- `optimize_parameters()` crashed immediately: `self.criterionGAN` / `criterionCycle` / `criterionStyle` never defined (F1)
- `utils/metrics.py` used `scipy.linalg.sqrtm` without importing scipy (F2)
- `AdaIN.forward` computed style statistics with the *content* batch size after flattening `[B, N, ...]` style features to `[B·N, ...]` — a shape mismatch whenever `N > 1` (F3); it also created a **new random 1×1 conv on every forward pass** for channel matching, injecting untrained noise into the style path
- `StyleEncoder.forward` referenced `B`/`N` outside the branch that defined them (F4)
- `--lambda_identity` existed as an option but no identity loss was ever computed (F5)
- Two divergent loss paths: `backward_G` (working, F.mse/l1-based) vs `optimize_parameters` (broken, criterion-based) — they disagreed on identity and style handling (F6)
- `app.py` blended pixels for the style weight (F7) and used a TensorFlow-Hub Magenta model entirely unrelated to the PyTorch codebase (F10)
- Cycle consistency was undefined for arbitrary styles: the reverse generator was conditioned on the *exemplar's* style while being asked to reconstruct the source (F8)
- README claimed "AdaIN alone requires paired training data" — false (F9)

**Verify:** every finding traced to a specific file/line before planning fixes.

## Step 1 — Repair the AdaIN core (`models/adain.py`)

1. **`AdaIN.forward`**: track `multi_style` explicitly; compute style mean/std per exemplar at batch size `B·N`, then average statistics over `N` back to `[B, C, 1, 1]`. Now style statistics always align with the content batch. (F3)
2. **Channel matching**: removed the per-call random conv. Channel projection is the *generator's* job — each `AdaINResBlock` now owns a learned `style_proj = Conv2d(128, dim, 1)` registered at construction time, so it is part of `netG_*.parameters()` and trained by the existing optimizer. `AdaIN` itself asserts channels already match.
3. **`StyleEncoder.forward`**: replaced the broken `B`/`N` scope check with a `multi_style` flag captured before reshaping. (F4)
4. **Alpha interpolation**: `AdaIN.forward(..., alpha)` implements `c + α·(AdaIN(c,s) − c)`, threaded through `AdaINResBlock.forward` and `AdaINGenerator.forward`. This is the feature-space style weight used by the app/demo. (foundation for F7)

**Verify:** smoke test — forward pass with `[B, N, C, H, W]` style input and `α ∈ {0, 0.5, 1}` produces correct shapes.

## Step 2 — Repair the training model (`models/adain_cycle_gan.py`)

1. **Defined the criteria** in `__init__`: `criterionGAN` (a small `GANLoss` LSGAN module that builds target tensors), `criterionCycle` (L1), `criterionIdt` (L1). (F1)
2. **`criterionStyle`** implemented as a method: VGG-encode the generated image, match channel-wise mean/std against the exemplar's features (with multi-exemplar averaging via a shared `_feature_stats` helper).
3. **Single loss path** (F6): `optimize_parameters` now only orchestrates — `forward()` → freeze Ds → `backward_G()` → clip → step → unfreeze Ds → `backward_D_A/B()` → clip → step. All loss math lives in `backward_G` / `backward_D_basic`, which use the criteria.
4. **Identity loss** (F5): `backward_G` computes `idt_A = G_A(real_B, style_B)` and `idt_B = G_B(real_A, style_A)` when `λ_idt > 0` and includes both terms in `loss_G`; `train.py` now logs them.
5. **Well-defined cycle** (F8): `forward()` additionally encodes `style_A = E(real_A)` and `style_B = E(real_B)`, and conditions every reverse mapping on the source's own statistics (see [02_architecture.md](02_architecture.md) §3).

**Verify:** one full `optimize_parameters()` step on random tensors runs without error and produces finite losses.

## Step 3 — Fix metrics (`utils/metrics.py`)

Added `import scipy.linalg` (F2). `scipy` was already pinned in `requirements.txt`.

## Step 4 — Reconnect the deployment (`app.py`)

Rewrote the Flask app on the custom PyTorch stack (F10):

- Loads `AdaINGenerator` (`netG_A` weights) + frozen `StyleEncoder` from a checkpoint (`CHECKPOINT_PATH` env var, default `./checkpoints/adain_cyclegan/latest_net.pth`); fails fast with a clear message if missing — no silent fallback to an unrelated model
- The style-weight slider maps to AdaIN's α (feature-space interpolation), replacing pixel blending (F7)
- Routes and request handling (upload/webcam/base64) kept unchanged

Also added the `--style_weight` flag to `demo.py` (README already documented it, but it didn't exist) and removed the now-unused `tensorflow` / `tensorflow-hub` dependencies from `requirements.txt`, ending the two-framework split flagged by the report.

## Step 5 — Correct the documentation (`README.md`)

- Fixed the false claim that AdaIN requires paired data (F9); the "Why combine?" rationale now states the honest trade-off (domain realism + exemplar control vs. AdaIN-alone simplicity)
- Web-app section now documents the checkpoint requirement and the feature-space style weight
- Fixed the misattributed MUNIT reference (Huang et al., ECCV 2018)

## Step 6 — Verify

Installed CPU PyTorch in a throwaway venv and ran a structural smoke test (VGG weights randomized to avoid the download):

- AdaIN forward: 4D/5D style inputs × α ∈ {0, 0.5, 1, 1.5}; α = 0 is an exact identity
- StyleEncoder: 4D and 5D inputs
- Full model: one `optimize_parameters()` step with `B=2`, `N=3` style exemplars → all nine losses finite and active

The first run **exposed a latent bug the report missed**: `AdaINResBlock` sliced its conv block as `[3:6]` for the second conv, but index 3 is the in-place ReLU — the activation ran twice in-place and broke autograd (`RuntimeError: ... modified by an inplace operation`). Fixed the slice to `[4:7]` and made the explicit ReLU non-in-place; the report never reached this crash because training died earlier at F1. After the fix, all smoke tests and `py_compile` checks pass.

## Step 7 — Write this documentation

Created the `document/` folder: research background, architecture, this process log, the finding-by-finding fix map, and the evaluation/deployment guide.

## Step 8 — Repository cleanup & organization

- **Deleted** the legacy TensorFlow SavedModel (`models/saved_model.pb` + `models/variables/`, ~90 MB) — unused by any code after the `app.py` rewrite
- **Deleted** the stale `docs/` folder — it described the pre-fix behavior (and `docs/images/` only contained a README for diagrams that were never created); `document/` is the single canonical documentation set
- **Moved** the verification report into [verification_report.md](verification_report.md) and the CycleGAN paper notes into [06_cyclegan_notes.md](06_cyclegan_notes.md)
- **Moved** the sample photos from `static/images/` (unreferenced by the web UI, which works on uploads) into `examples/content/` and `examples/styles/` for use with `demo.py`; removed the empty `static/` folder
- **Rewrote** the root `README.md` to match the actual implementation

## Known remaining limitations

- **No trained checkpoint ships with the repo.** Training on the bundled `datasets/Project_dataset` (or e.g. `monet2photo` + a style set) is required before `app.py`/`demo.py` produce meaningful output. The legacy TF SavedModel files that previously sat under `models/` have been deleted during cleanup.
- `train.py` constructs `ImagePool` buffers but the discriminator update consumes current fakes directly (pre-existing; harmless but the pools are dead code).
- FID computed over LPIPS/AlexNet features is non-standard; scores are not comparable with Inception-based FID from the literature.
