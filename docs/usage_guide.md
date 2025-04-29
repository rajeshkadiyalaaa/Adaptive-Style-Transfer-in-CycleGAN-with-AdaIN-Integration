# Usage Guide

This document provides detailed instructions for installing, training, testing, and using the AdaIN-CycleGAN model.

## Installation

### Requirements

- Python 3.7 or later
- PyTorch 1.9 or later
- CUDA-capable GPU (recommended for training)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/adain-cyclegan.git
   cd adain-cyclegan
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Download a pre-trained model (optional):
   ```bash
   # Will be available in future releases
   ```

## Dataset Preparation

AdaIN-CycleGAN requires datasets for both content domains (A and B) and style references. The expected directory structure is:

```
datasets/
└── your_dataset_name/
    ├── trainA/
    │   └── (training images from domain A)
    ├── trainB/
    │   └── (training images from domain B)
    ├── testA/
    │   └── (test images from domain A)
    ├── testB/
    │   └── (test images from domain B)
    └── style/
        └── (style reference images)
```

### Downloading Example Datasets

You can download example datasets using the provided script:

```bash
python data/download_data.py --dataset monet2photo --download_styles
```

Available datasets include:
- `monet2photo`
- `vangogh2photo`
- `ukiyoe2photo`
- `cezanne2photo`
- `apple2orange`
- `summer2winter_yosemite`

### Creating Your Own Dataset

To use your own images:

1. Create the required directory structure:
   ```bash
   python data/download_data.py --dataset your_dataset_name --create_structure
   ```

2. Place your images in the appropriate folders:
   - Content domain A images in `trainA/` and `testA/`
   - Content domain B images in `trainB/` and `testB/`
   - Style reference images in `style/`

## Training

### Basic Training

To train the model with default parameters:

```bash
python train.py --dataroot ./datasets/your_dataset_name --name your_experiment_name --use_adain
```

### Training Options

Common training options include:

| Option | Description |
|--------|-------------|
| `--dataroot` | Path to the dataset directory |
| `--name` | Name of the experiment |
| `--use_adain` | Enable AdaIN layers for style transfer |
| `--n_epochs` | Number of epochs with initial learning rate |
| `--n_epochs_decay` | Number of epochs to linearly decay learning rate |
| `--batch_size` | Input batch size |
| `--lambda_style` | Weight for style loss |
| `--n_style_images` | Number of style images to use per batch |
| `--continue_train` | Continue training from the latest checkpoint |

For a complete list of options, see:

```bash
python train.py --help
```

### Training Tips

1. **Start Small**: Begin with a smaller dataset and fewer epochs to ensure everything works correctly.
2. **Monitoring**: Use TensorBoard to monitor training progress:
   ```bash
   tensorboard --logdir=./checkpoints/your_experiment_name/logs
   ```
3. **Checkpoint Frequency**: Adjust `--save_epoch_freq` to control how often checkpoints are saved.
4. **Early Stopping**: Monitor validation metrics to prevent overfitting.

## Testing

### Basic Testing

To test the model on your test dataset:

```bash
python test.py --dataroot ./datasets/your_dataset_name --name your_experiment_name --use_adain
```

### Style Transfer Testing

For arbitrary style transfer using specific content and style images:

```bash
python test.py --content_image path/to/content.jpg --style_image path/to/style.jpg --name your_experiment_name --output_path ./results/output.jpg --use_adain
```

### Evaluation Metrics

To compute evaluation metrics:

```bash
python test.py --dataroot ./datasets/your_dataset_name --name your_experiment_name --compute_metrics --use_adain
```

## Demo Application

Use the demo script for quick style transfer between individual images:

```bash
python demo.py --content path/to/content/image --style path/to/style/image --output path/to/output/image --model ./checkpoints/your_experiment_name/latest_net.pth
```

## Model Configuration

### Key Configuration Files

- `config/base_options.py`: Common options for both training and testing
- `config/train_options.py`: Training-specific options
- `config/test_options.py`: Testing-specific options

### Custom Configuration

To customize the model architecture or training procedure, modify:

- `models/adain_cycle_gan.py`: Main model implementation
- `models/adain.py`: AdaIN layer and related components

## Troubleshooting

### Common Issues

1. **Out of Memory Errors**:
   - Reduce `batch_size`
   - Use a smaller `load_size` and `crop_size`
   - Use CPU mode with `--gpu_ids -1` (much slower)

2. **Training Instability**:
   - Reduce learning rate (`--lr`)
   - Adjust loss weights (`--lambda_A`, `--lambda_B`, `--lambda_style`)
   - Enable gradient clipping by keeping default settings

3. **Poor Style Transfer**:
   - Ensure style images are diverse and representative
   - Adjust `--lambda_style` to control style influence
   - Try different style sampling methods

4. **Missing Modules**:
   - Verify all dependencies are installed
   - Check the PyTorch and CUDA versions

### Support

For additional support:
- Check the GitHub repository issues
- Refer to the paper references
- Consult the original CycleGAN and AdaIN papers

## Advanced Usage

### Multi-GPU Training

To use multiple GPUs:

```bash
python train.py --dataroot ./datasets/your_dataset_name --name your_experiment_name --gpu_ids 0,1,2,3 --use_adain
```

### Custom Style Sampling

Control how style images are sampled:

```bash
python train.py --dataroot ./datasets/your_dataset_name --name your_experiment_name --style_sample_method random --n_style_images 5 --use_adain
```

### Fine-Tuning

Fine-tune a pre-trained model:

```bash
python train.py --dataroot ./datasets/your_dataset_name --name your_experiment_name --continue_train --epoch_count [last_epoch+1] --use_adain
``` 