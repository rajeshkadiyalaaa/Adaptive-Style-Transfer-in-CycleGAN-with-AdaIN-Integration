#!/usr/bin/env python3
import argparse
import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from models.adain_cycle_gan import AdaINStyleCycleGAN
from models.adain import StyleEncoder


def get_transform(size=256):
    """Get transformation for image preprocessing.
    
    Args:
        size (int): Image size after transformation
        
    Returns:
        torchvision.transforms.Compose: Composition of image transforms
    """
    return transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])


def load_image(image_path, transform=None, max_size=None):
    """Load and transform an image.
    
    Args:
        image_path (str): Path to the image
        transform: Optional transformation to apply
        max_size (int, optional): Maximum size of the image
        
    Returns:
        torch.Tensor: Transformed image tensor
    """
    image = Image.open(image_path).convert('RGB')
    
    # Resize if max_size is specified
    if max_size is not None:
        if image.size[0] > image.size[1]:
            aspect_ratio = image.size[0] / image.size[1]
            size = (int(max_size * aspect_ratio), max_size)
        else:
            aspect_ratio = image.size[1] / image.size[0]
            size = (max_size, int(max_size * aspect_ratio))
        image = image.resize(size, Image.BICUBIC)
    
    # Apply transform if provided
    if transform is not None:
        image = transform(image).unsqueeze(0)
    
    return image


def tensor_to_image(tensor):
    """Convert tensor to PIL Image.
    
    Args:
        tensor (torch.Tensor): Image tensor of shape (1, C, H, W)
        
    Returns:
        PIL.Image: PIL image
    """
    # Denormalize
    tensor = tensor.clone().detach().cpu()
    tensor = (tensor + 1) / 2.0
    tensor = tensor.clamp_(0, 1)
    
    # Convert to numpy array
    image = tensor.numpy()
    image = image.squeeze(0)
    image = np.transpose(image, (1, 2, 0))
    image = (image * 255).astype(np.uint8)
    
    return Image.fromarray(image)


def style_transfer(content_path, style_path, output_path, model_path, device='cpu'):
    """Perform style transfer with AdaIN-CycleGAN.
    
    Args:
        content_path (str): Path to content image
        style_path (str): Path to style image
        output_path (str): Path to save the result image
        model_path (str): Path to the model checkpoint
        device (str): Device to use ('cpu' or 'cuda')
    """
    # Create options object
    class Opt:
        def __init__(self):
            self.name = 'adain_cyclegan'
            self.use_adain = True
            self.input_nc = 3
            self.output_nc = 3
            self.ngf = 64
            self.ndf = 64
            self.netG = 'adain_resnet_9blocks'
            self.gpu_ids = [0] if device == 'cuda' and torch.cuda.is_available() else []
    
    opt = Opt()
    
    # Set up the device
    device = torch.device('cuda' if device == 'cuda' and torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load the model
    model = AdaINStyleCycleGAN(opt)
    
    # Load weights
    if os.path.exists(model_path):
        print(f'Loading model from {model_path}')
        checkpoint = torch.load(model_path, map_location=device)
        model.netG_A.load_state_dict(checkpoint['netG_A'])
    else:
        print(f'Model file not found: {model_path}')
        return
    
    # Set model to evaluation mode
    model.eval()
    
    # Load and preprocess the content and style images
    transform = get_transform()
    content_img = load_image(content_path, transform)
    style_img = load_image(style_path, transform)
    
    # Move to device
    content_img = content_img.to(device)
    style_img = style_img.to(device)
    
    # Extract style features
    style_features = model.style_encoder(style_img)
    
    # Perform style transfer
    with torch.no_grad():
        output = model.netG_A(content_img, style_features)
    
    # Save the result
    result_img = tensor_to_image(output)
    result_img.save(output_path)
    print(f'Result saved to {output_path}')
    
    # Display the images
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(Image.open(content_path))
    plt.title('Content Image')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(Image.open(style_path))
    plt.title('Style Image')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(result_img)
    plt.title('Output Image')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()


def main():
    """Main function for the demo script."""
    parser = argparse.ArgumentParser(description='AdaIN-CycleGAN Style Transfer Demo')
    parser.add_argument('--content', type=str, required=True, help='path to content image')
    parser.add_argument('--style', type=str, required=True, help='path to style image')
    parser.add_argument('--output', type=str, default='output.jpg', help='path to save the result')
    parser.add_argument('--model', type=str, default='./checkpoints/adain_cyclegan/latest_net.pth', help='path to model checkpoint')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'], help='device to use')
    
    args = parser.parse_args()
    
    # Check if files exist
    if not os.path.exists(args.content):
        print(f'Content image not found: {args.content}')
        return
    
    if not os.path.exists(args.style):
        print(f'Style image not found: {args.style}')
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    
    # Perform style transfer
    style_transfer(args.content, args.style, args.output, args.model, args.device)


if __name__ == '__main__':
    main() 