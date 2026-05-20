#!/usr/bin/env python3
import torch
import os
import numpy as np
from torch.utils.data import DataLoader
import torchvision.utils as vutils
from PIL import Image

from config.test_options import TestOptions
from data.datasets import UnalignedStyleDataset, SingleStyleDataset
from models.adain_cycle_gan import AdaINStyleCycleGAN
from utils.metrics import calculate_psnr, calculate_ssim, FID


def save_images(visuals, result_dir, image_path, opt):
    """Save images to the disk.
    
    Args:
        visuals (dict): Dictionary of images to save
        result_dir (str): Directory to save results
        image_path (str): Path of the input image
        opt: Command line options
    """
    short_path = os.path.basename(image_path)
    name = os.path.splitext(short_path)[0]
    
    for label, im_data in visuals.items():
        # Denormalize the image
        im = im_data.detach().cpu().float().numpy()
        im = (im + 1) / 2.0  # Convert from [-1, 1] to [0, 1]
        im = np.transpose(im[0], (1, 2, 0))  # CHW to HWC
        im = np.clip(im, 0, 1)
        im = (im * 255).astype(np.uint8)
        
        # Save the image
        im = Image.fromarray(im)
        image_name = f'{name}_{label}.png'
        save_path = os.path.join(result_dir, image_name)
        im.save(save_path)


def test_model(opt):
    """Test the model with performance optimizations.
    
    Args:
        opt: Command line options
    """
    # Create the results directory
    result_dir = os.path.join(opt.results_dir, opt.name)
    os.makedirs(result_dir, exist_ok=True)
    
    # Create the dataset and dataloader
    if opt.content_image is not None and opt.style_image is not None:
        # Use single image for arbitrary style transfer
        dataset = SingleStyleDataset(opt)
    else:
        # Use standard CycleGAN test dataset
        dataset = UnalignedStyleDataset(opt)
    
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=opt.num_threads,
        pin_memory=torch.cuda.is_available()
    )
    
    # Create the model
    model = AdaINStyleCycleGAN(opt)
    model.eval()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and not hasattr(opt, 'cpu_mode') else 'cpu')
    
    # Load the trained model
    checkpoint_path = os.path.join(opt.checkpoints_dir, opt.name, 'latest_net.pth')
    if not os.path.exists(checkpoint_path):
        print(f'Checkpoint not found at {checkpoint_path}')
        return
    
    print(f'Loading the model from {checkpoint_path}')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.netG_A.load_state_dict(checkpoint['netG_A'])
    model.netG_B.load_state_dict(checkpoint['netG_B'])
    
    # Initialize metric calculators if needed
    metrics = None
    fid_calculator = None
    outputs_for_fid = []
    refs_for_fid = []
    
    if opt.compute_metrics:
        metrics = {
            'psnr': [],
            'ssim': []
        }
        if hasattr(opt, 'fid') and opt.fid:
            metrics['fid'] = []
            fid_calculator = FID(opt.gpu_ids[0] if opt.gpu_ids else 'cpu')
    
    # Testing loop with torch.inference_mode() for faster inference
    with torch.inference_mode():
        for i, data in enumerate(dataloader):
            print(f'Processing image {i+1}/{len(dataloader)}')
            
            if isinstance(dataset, SingleStyleDataset):
                # For single image style transfer
                content = data['content'].to(model.device)
                style = data['style'].to(model.device) if 'style' in data else None
                
                # Extract style features if using AdaIN
                if style is not None and model.use_adain:
                    style_features = model.style_encoder(style)
                    
                    # Generate stylized output
                    output = model.netG_A(content, style_features)
                    
                    # Save the results
                    visuals = {'content': content, 'style': style, 'output': output}
                    save_images(visuals, result_dir, data['content_path'][0], opt)
                    
                    # Compute metrics if needed
                    if metrics and opt.reference_dir:
                        # Load reference image
                        ref_path = os.path.join(opt.reference_dir, os.path.basename(data['content_path'][0]))
                        if os.path.exists(ref_path):
                            ref_img = Image.open(ref_path).convert('RGB')
                            transform = dataset.transform
                            ref_tensor = transform(ref_img).unsqueeze(0).to(model.device)
                            
                            # Compute metrics
                            metrics['psnr'].append(calculate_psnr(output, ref_tensor))
                            metrics['ssim'].append(calculate_ssim(output, ref_tensor))
                            if 'fid' in metrics:
                                # Collect outputs for FID calculation
                                outputs_for_fid.append(output.detach().cpu())
                                refs_for_fid.append(ref_tensor.detach().cpu())
            else:
                # For standard CycleGAN testing
                model.set_input(data)
                
                # Forward pass
                model.forward()
                
                # Save the results
                visuals = {
                    'real_A': model.real_A,
                    'fake_B': model.fake_B,
                    'rec_A': model.rec_A,
                    'real_B': model.real_B,
                    'fake_A': model.fake_A,
                    'rec_B': model.rec_B
                }
                
                if hasattr(data, 'A_paths'):
                    img_path = data['A_paths'][0]
                elif hasattr(data, 'path'):
                    img_path = data['path'][0]
                else:
                    img_path = f'test_image_{i}.png'
                
                save_images(visuals, result_dir, img_path, opt)
                
                # Compute metrics if needed
                if metrics:
                    metrics['psnr'].append(calculate_psnr(model.rec_A, model.real_A))
                    metrics['ssim'].append(calculate_ssim(model.rec_A, model.real_A))
                    metrics['psnr'].append(calculate_psnr(model.rec_B, model.real_B))
                    metrics['ssim'].append(calculate_ssim(model.rec_B, model.real_B))
    
    # Compute FID if collected
    if metrics and 'fid' in metrics and len(outputs_for_fid) > 0:
        outputs = torch.cat(outputs_for_fid, dim=0)
        refs = torch.cat(refs_for_fid, dim=0)
        metrics['fid'] = fid_calculator.calculate_fid(refs, outputs)
    
    # Print metrics results
    if metrics:
        print('===== Metrics Results =====')
        for metric_name, values in metrics.items():
            if metric_name != 'fid':
                if isinstance(values, list) and len(values) > 0:
                    print(f'Average {metric_name.upper()}: {np.mean(values):.4f}')
            else:
                print(f'FID: {values:.4f}')
        
        # Save metrics to file
        with open(os.path.join(result_dir, 'metrics.txt'), 'w') as f:
            for metric_name, values in metrics.items():
                if metric_name != 'fid':
                    if isinstance(values, list) and len(values) > 0:
                        f.write(f'Average {metric_name.upper()}: {np.mean(values):.4f}\n')
                else:
                    f.write(f'FID: {values:.4f}\n')


def main():
    """Main testing function."""
    # Parse command-line options
    opt = TestOptions().parse()
    
    # Enable cuDNN benchmarking for faster inference
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    
    # Test the model
    test_model(opt)


if __name__ == '__main__':
    main()
