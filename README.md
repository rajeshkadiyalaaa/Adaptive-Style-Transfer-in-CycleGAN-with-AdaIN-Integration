# 🎨 Adaptive Style Transfer in CycleGAN with AdaIN Integration

[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A cutting-edge implementation that combines **CycleGAN's** unpaired image translation with **Adaptive Instance Normalization (AdaIN)** for flexible, user-defined style transfer without retraining.

## ✨ Key Features

- **🎭 Arbitrary Style Transfer**: Apply any style to any content image without model retraining
- **📸 Content Preservation**: Maintains structural integrity while transferring style
- **🎨 User-Defined Stylization**: Define custom styles on-the-fly
- **🌐 Interactive Web Interface**: Easy-to-use Flask web application for non-technical users
- **📊 Improved Metrics**: Better PSNR, SSIM, and FID scores compared to standard CycleGAN
- **⚡ Real-time Processing**: Optimized inference with controllable style weight
- **🎛️ Style Weight Control**: Adjustable influence of style on output images

## 🏗️ Architecture Overview

This project integrates AdaIN layers into CycleGAN's architecture to enable:

1. **Dynamic Style Adaptation**: Content features are dynamically aligned with style statistics
2. **Cycle Consistency**: Maintains content integrity through bidirectional translation
3. **Adversarial Training**: Ensures realistic output images
4. **Flexible Inference**: Apply any style without retraining

### Core Components

```
AdaIN-CycleGAN = CycleGAN + AdaIN Integration
│
├─ Two Generators (A→B, B→A)
├─ Two Discriminators
├─ Style Encoder (VGG19-based)
└─ AdaIN Layers (for style modulation)
```

## 📁 Project Structure

```
.
├── config/                      # Configuration and hyperparameters
│   ├── base_options.py         # Base configuration
│   └── train_options.py        # Training-specific options
├── data/                        # Data handling
│   ├── datasets.py             # Dataset implementations
│   └── download_data.py        # Dataset download utilities
├── models/                      # Model implementations
│   ├── adain.py                # AdaIN layer implementation
│   └── adain_cycle_gan.py      # Main model architecture
├── utils/                       # Utility functions
│   ├── image_pool.py           # Image buffer for training
│   └── metrics.py              # PSNR, SSIM, FID metrics
├── templates/                   # HTML templates for web interface
├── static/                      # CSS, JavaScript assets
├── checkpoints/                 # Saved model checkpoints
├── datasets/                    # Training and test datasets
├── docs/                        # Comprehensive documentation
│   ├── overview.md             # Project overview
│   ├── architecture.md         # Architecture details
│   ├── technical_details.md    # Implementation details
│   ├── workflow.md             # System workflow
│   └── usage_guide.md          # Detailed usage instructions
├── app.py                       # Flask web application
├── train.py                     # Training script
├── test.py                      # Testing and evaluation
├── demo.py                      # Command-line demo
├── CycleGan.md                  # Technical documentation
└── requirements.txt             # Python dependencies
```

## 🚀 Quick Start

### Prerequisites

- **Python 3.8+**
- **CUDA-compatible GPU** (recommended for training)
- **GPU Memory**: 4GB+ for training, 2GB+ for inference

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/rajeshkadiyalaaa/Adaptive-Style-Transfer-in-CycleGAN-with-AdaIN-Integration.git
   cd Adaptive-Style-Transfer-in-CycleGAN-with-AdaIN-Integration
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download datasets** (optional):
   ```bash
   python data/download_data.py --dataset monet2photo --download_styles
   ```

## 💻 Usage

### 🌐 Web Interface (Recommended for Beginners)

The easiest way to use style transfer:

```bash
python app.py
```

Then open your browser to `http://localhost:5000` and:
1. Upload a content image or capture one with your webcam
2. Upload a style reference image
3. Adjust the style weight slider (0-2.0)
4. Click "Generate" to create your styled image

### 🖥️ Command-Line Usage

For scripting and batch processing:

```bash
python demo.py \
  --content path/to/content/image.jpg \
  --style path/to/style/image.jpg \
  --output path/to/output/result.jpg \
  --style_weight 1.0
```

**Parameters**:
- `--content`: Path to content image
- `--style`: Path to style reference image
- `--output`: Output image path
- `--style_weight`: Style influence (0.0-2.0, default: 1.0)

### 🏋️ Training Your Own Model

1. **Prepare your dataset** with the following structure:
   ```
   datasets/your_dataset/
   ├── trainA/          # Content domain A training images
   ├── trainB/          # Content domain B training images
   ├── testA/           # Content domain A test images
   ├── testB/           # Content domain B test images
   └── style/           # Style reference images
   ```

2. **Start training**:
   ```bash
   python train.py \
     --dataroot ./datasets/your_dataset \
     --name my_experiment \
     --batch_size 4 \
     --n_epochs 100 \
     --n_epochs_decay 100
   ```

3. **Monitor training with TensorBoard**:
   ```bash
   tensorboard --logdir ./checkpoints/my_experiment/logs
   ```

### 📊 Evaluation

Evaluate model performance on test data:

```bash
python test.py \
  --dataroot ./datasets/your_dataset \
  --name my_experiment \
  --results_dir ./results
```

This generates evaluation metrics (PSNR, SSIM, FID).

## 📋 Training Options

Common hyperparameters:

| Option | Default | Description |
|--------|---------|-------------|
| `--batch_size` | 8 | Batch size for training |
| `--n_epochs` | 100 | Epochs with initial learning rate |
| `--n_epochs_decay` | 100 | Epochs for learning rate decay |
| `--lr` | 0.0002 | Initial learning rate |
| `--lambda_A` | 10.0 | Cycle consistency weight (A→B→A) |
| `--lambda_B` | 10.0 | Cycle consistency weight (B→A→B) |
| `--lambda_style` | 1.0 | Style loss weight |
| `--lambda_identity` | 0.5 | Identity loss weight |

For complete options, run:
```bash
python train.py --help
```

## 🔬 Technical Approach

### CycleGAN Architecture
- Two paired generators for bidirectional translation
- Two discriminators for adversarial training
- Cycle consistency loss for content preservation

### AdaIN Integration
- **Adaptive Instance Normalization**: Aligns content feature statistics with style statistics
- **Formula**: `AdaIN(C, S) = σ(S) * (C - μ(C)) / σ(C) + μ(S)`
  - `C`: Content features
  - `S`: Style features
  - Enables arbitrary style transfer without retraining

### Loss Functions
1. **Adversarial Loss**: Ensures realistic generated images
2. **Cycle Consistency Loss**: Maintains content integrity (A→B→A)
3. **Identity Loss**: Preserves color when input is already in target domain
4. **Style Loss**: Ensures style fidelity through AdaIN

## 📈 Results & Performance

### Improvements over Standard CycleGAN
- **PSNR**: Higher values indicating better reconstruction quality
- **SSIM**: Higher structural similarity preservation
- **FID**: Lower Fréchet Inception Distance indicating better style transfer

### Advantages
- ✅ Works with any style without retraining
- ✅ Better content preservation
- ✅ User control over style intensity
- ✅ Real-time inference capability

## 🎯 Applications

- **🖼️ Digital Art**: Create artworks in various artistic styles
- **👗 Fashion Design**: Visualize clothing in different styles
- **🎬 Entertainment**: Apply visual effects to films and games
- **🌐 Virtual Reality**: Transform VR environments with custom styles
- **📷 Photography**: Enhance and stylize photographs with filters

## 📚 Documentation

For more detailed information, see:
- [**Project Overview**](docs/overview.md) - High-level introduction
- [**Architecture**](docs/architecture.md) - System components and data flow
- [**Technical Details**](docs/technical_details.md) - Implementation specifics
- [**Workflow**](docs/workflow.md) - Step-by-step processing pipeline
- [**Usage Guide**](docs/usage_guide.md) - Detailed usage instructions
- [**CycleGAN Technical Paper Notes**](CycleGan.md) - Research background

## 🔗 References

1. **CycleGAN**: J. Zhu, T. Park, P. Isola, and A. A. Efros. "Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks." ICCV 2017. [arXiv:1703.10593](https://arxiv.org/abs/1703.10593)

2. **AdaIN**: X. Huang and S. Belongie. "Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization." ICCV 2017. [arXiv:1703.06868](https://arxiv.org/abs/1703.06868)

3. **MUNIT**: H. Lee, H. Tseng, J. Huang, M. Singh, and M. Yang. "Diverse Image-to-Image Translation via Disentangled Representations." ECCV 2021.

## 🛠️ System Requirements

| Component | Requirement |
|-----------|-------------|
| **Python** | 3.8 or higher |
| **GPU** | CUDA-compatible (recommended) |
| **GPU Memory** | 4GB+ for training, 2GB+ for inference |
| **RAM** | 8GB minimum recommended |
| **Storage** | 500MB+ for models and datasets |

## 📝 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to:
- Report bugs and issues
- Suggest improvements
- Submit pull requests with enhancements
- Improve documentation

## 📧 Support

For issues, questions, or suggestions, please open an [Issue](https://github.com/rajeshkadiyalaaa/Adaptive-Style-Transfer-in-CycleGAN-with-AdaIN-Integration/issues) on GitHub.

---

## 🎓 Key Concepts

### What is CycleGAN?
CycleGAN performs **unpaired image-to-image translation** between two domains without requiring paired examples, using cycle consistency loss to preserve content.

### What is AdaIN?
Adaptive Instance Normalization (**AdaIN**) aligns the mean and variance of content features with those of style features, enabling **arbitrary style transfer**.

### Why Combine Them?
- **CycleGAN** alone is limited to fixed domain pairs
- **AdaIN alone** requires paired training data
- **Combined**: Flexible, unpaired style transfer with arbitrary styles

---

**Made with ❤️ for the computer vision community**
