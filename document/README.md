# Project Documentation

Documentation for **Adaptive Style Transfer in CycleGAN with AdaIN Integration**, written from the perspective of building (and repairing) this project end to end.

| File | Contents |
|---|---|
| [01_research.md](01_research.md) | Research background: CycleGAN, AdaIN, why combine them, and the design tension that has to be resolved |
| [02_architecture.md](02_architecture.md) | System architecture: networks, data flow, loss functions, and how cycle consistency is kept well-defined |
| [03_process.md](03_process.md) | Step-by-step build/repair process: what was done, in what order, and why |
| [04_fixes_and_changes.md](04_fixes_and_changes.md) | Verification report findings (F1–F10) mapped to the exact code changes that resolve them |
| [05_evaluation_and_deployment.md](05_evaluation_and_deployment.md) | How to train, evaluate (PSNR/SSIM/FID), and deploy the web app |
| [06_cyclegan_notes.md](06_cyclegan_notes.md) | Working notes on the CycleGAN paper (architecture, losses, comparison to other GANs) |
| [verification_report.md](verification_report.md) | The original third-party verification report whose findings (F1–F10) drove the repair work |

## Quick orientation

- **Model code**: `models/adain.py` (AdaIN layer, VGG19 style encoder, AdaIN residual block), `models/adain_cycle_gan.py` (generators, discriminators, full training model)
- **Pipelines**: `train.py` (training), `test.py` (evaluation), `demo.py` (CLI inference), `app.py` (Flask web app)
- **Data**: `data/datasets.py` (unpaired A/B + style exemplar datasets)
- **Known limitation**: no pre-trained checkpoint ships with the repository. The web app and demo require a checkpoint produced by `train.py`.
