# Adaptive Style Transfer in CycleGAN with AdaIN Integration

This project implements the research on enhancing CycleGAN with Adaptive Instance Normalization (AdaIN) for user-defined style transfer. The implementation enables dynamic, arbitrary style transfer without requiring retraining for each new style.

## Overview

Traditional CycleGAN models are limited to translating images between fixed domains. Our approach integrates AdaIN layers into the CycleGAN architecture, enabling flexible style adaptation while maintaining content integrity. This allows users to apply arbitrary styles to their images without training separate models for each style.

## Features

- **Arbitrary Style Transfer**: Apply any style to any content image without retraining
- **Content Preservation**: Maintains the structural integrity of content while transferring style
- **User-Defined Stylization**: Enables custom, on-the-fly style definition
- **Interactive Web Interface**: Web application for easy style transfer without coding knowledge
- **Improved Metrics**: Better PSNR, SSIM, and FID values compared to standard CycleGAN
- **Real-time Processing**: Optimized for efficient style transfer with controllable style weight

## Project Structure

```
.
├── config/                 # Configuration files and parameter settings
├── data/                   # Data handling utilities
│   ├── datasets.py         # Dataset implementations
│   └── download_data.py    # Scripts to download datasets
├── models/                 # Model implementations
│   ├── adain.py            # AdaIN layer implementation
│   ├── adain_cycle_gan.py  # CycleGAN with AdaIN integration
│   └── saved_model.pb      # Pre-trained TensorFlow model
├── utils/                  # Utility functions
│   ├── image_pool.py       # Image buffer for training
│   └── metrics.py          # PSNR, SSIM, FID metrics
├── templates/              # HTML templates for web interface
├── static/                 # Static assets for web interface (CSS, JS)
├── checkpoints/            # Model checkpoints saved during training
├── datasets/               # Training and testing datasets
├── docs/                   # Documentation and additional resources
├── app.py                  # Flask web application for style transfer
├── train.py                # Training script
├── test.py                 # Testing and evaluation script
├── demo.py                 # Demo application for command-line usage
├── CycleGan.md             # Detailed technical documentation on the approach
└── requirements.txt        # Required Python packages
```

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/adaptive-style-transfer.git
cd adaptive-style-transfer
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Web Application

The easiest way to use the style transfer is through the web interface:

1. Start the Flask application:
```bash
python app.py
```

2. Open your browser and navigate to `http://localhost:5000`

3. Upload a content image (or take one using your webcam) and a style image

4. Adjust the style weight slider if desired

5. Click "Generate" to create your styled image

### Command-line Style Transfer

To apply a style to content images using the command line:

```bash
python demo.py --content path/to/content/image --style path/to/style/image --output path/to/output --style_weight 1.0
```

### Training

To train the model with your own datasets:

```bash
python train.py --dataroot ./datasets/your_dataset --name your_experiment_name --batch_size 4 --n_epochs 100 --n_epochs_decay 100
```

Additional training options can be found in `config/train_options.py`.

### Evaluation

To evaluate model performance:

```bash
python test.py --dataroot ./datasets/test_dataset --name your_experiment_name --results_dir ./results
```

## Technical Approach

Our implementation combines CycleGAN's unpaired image translation capabilities with AdaIN for flexible style transfer:

1. **CycleGAN Architecture**: Two generators and discriminators with cycle consistency loss
2. **AdaIN Integration**: Dynamic alignment of content features with style statistics
3. **Style Weight Control**: Adjustable influence of style on the output image
4. **Improved Training**: Enhanced generator architecture with residual blocks and skip connections

For more technical details, refer to `CycleGan.md` in this repository.

## Results

Our implementation demonstrates significant improvements over the standard CycleGAN:

- Higher PSNR and SSIM values indicating better content preservation
- Lower FID values indicating better style transfer quality
- Greater flexibility in style application
- User control over style intensity

## Applications

This technology has applications in multiple domains:

- **Digital Art**: Create artworks in various styles
- **Fashion Design**: Visualize clothing in different styles
- **Entertainment**: Apply visual styles to film and game content
- **Virtual Reality**: Transform VR environments with custom styles
- **Photography**: Enhance and stylize photographs with artistic filters

## System Requirements

- Python 3.8 or higher
- CUDA-compatible GPU (recommended for training)
- 4GB+ GPU memory for training
- 2GB+ RAM for inference

## References

1. J. Zhu, T. Park, P. Isola, and A. A. Efros, "Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks," in 2017 IEEE International Conference on Computer Vision (ICCV), 2017.
2. X. Huang and S. Belongie, "Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization," in Proceedings of the IEEE International Conference on Computer Vision, 2017.
3. H. Lee, H. Tseng, J. Huang, M. Singh, and M. Yang, "Diverse Image-to-Image Translation via Disentangled Representations," in European Conference on Computer Vision, 2021.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 