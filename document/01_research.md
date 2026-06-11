# Research Background

## 1. The two building blocks

### CycleGAN (Zhu et al., ICCV 2017)

CycleGAN learns a mapping between two image domains **A** and **B** without paired examples. It uses:

- Two generators: `G_A: A → B` and `G_B: B → A`
- Two PatchGAN discriminators: `D_A` (judges domain-B realism) and `D_B` (judges domain-A realism)
- **Adversarial loss** so generated images look like the target domain
- **Cycle-consistency loss** `||G_B(G_A(a)) − a||₁` so content survives the round trip
- **Identity loss** `||G_A(b) − b||₁` so generators preserve color/composition when given an image already in their target domain

Limitation: a trained CycleGAN maps to **one fixed target domain**. There is no per-image style control — "photo → Monet" always produces the same flavor of Monet.

### AdaIN (Huang & Belongie, ICCV 2017)

Adaptive Instance Normalization performs **arbitrary** style transfer in a single forward pass:

```
AdaIN(c, s) = σ(s) · (c − μ(c)) / σ(c) + μ(s)
```

The content features `c` are normalized, then re-scaled and re-shifted with the channel-wise statistics of the style features `s` (computed with a fixed VGG encoder). Because the style enters only through its feature statistics, **any** style image can be used at inference time with no retraining.

Important: AdaIN is trained on **unpaired** data (any content set + any style set). It does *not* require paired training data — an earlier version of this project's README claimed otherwise, which was incorrect and has been fixed.

### MUNIT (Huang et al., ECCV 2018)

MUNIT is the closest precedent for this project: it decomposes images into a domain-invariant content code and a domain-specific style code, and injects style via AdaIN inside the decoder, trained with adversarial + reconstruction losses on unpaired data. It demonstrates that AdaIN-style modulation inside a GAN translation framework is a sound design.

## 2. Why combine them — and the honest trade-off

Neither block alone gives what we want:

| | Domain realism (adversarial) | Per-exemplar style control | Content preservation guarantee |
|---|---|---|---|
| CycleGAN | yes | no | yes (cycle loss) |
| AdaIN | no | yes | weak (content loss only) |
| **This project** | yes | yes | yes |

The combination gives a translator to a target *domain* (e.g. paintings) whose exact appearance is steered per-image by a style exemplar, while adversarial training keeps outputs realistic and cycle consistency protects structure.

### The design tension

There is a real conceptual problem with naively bolting AdaIN onto CycleGAN: **cycle consistency is ill-defined for arbitrary styles**. If `G_A` stylizes `a` with exemplar `s`, what statistics should `G_B` use to map back? Using the *same* exemplar `s` for the reverse direction (as the original code did) is wrong — the reverse generator would be asked to reconstruct `a` while being told to match the appearance of `s`.

### Our resolution

We condition each reverse mapping on the **source image's own style statistics**:

- Forward: `fake_B = G_A(real_A, style(s))` — stylize with the exemplar
- Backward: `rec_A = G_B(fake_B, style(real_A))` — reconstruct A using A's own VGG statistics

This makes the cycle target unambiguous: the reverse generator always knows which appearance it must restore. The exemplar style is constrained separately by an AdaIN-style loss (mean/std matching in VGG feature space) on `fake_B`. Section 3 of [02_architecture.md](02_architecture.md) covers the loss formulation in detail.

The remaining honest caveat: for *pure* arbitrary style transfer with no domain prior, AdaIN alone is simpler and sufficient. This project is justified when you want **both** a domain prior (adversarial realism in a target domain) **and** exemplar control — the MUNIT-style middle ground.

## 3. Reference papers

The `Research/` folder contains the three primary sources:

1. `CycleGAN.pdf` — Zhu, Park, Isola, Efros. *Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks.* ICCV 2017. [arXiv:1703.10593](https://arxiv.org/abs/1703.10593)
2. `arbitrary_style_transfer_Adaln.pdf` — Huang, Belongie. *Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization.* ICCV 2017. [arXiv:1703.06868](https://arxiv.org/abs/1703.06868)
3. `DomainTransfer.pdf` — domain transfer background; see also MUNIT: Huang, Liu, Belongie, Kautz. ECCV 2018. [arXiv:1804.04732](https://arxiv.org/abs/1804.04732)
