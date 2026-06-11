import torch
import numpy as np
import scipy.linalg
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import lpips
import torch.nn as nn


def tensor_to_numpy(tensor):
    """Convert torch tensor to numpy array.
    
    Args:
        tensor (torch.Tensor): Input tensor of shape (N, C, H, W)
        
    Returns:
        numpy.ndarray: Output numpy array of shape (N, H, W, C)
    """
    # Denormalize if tensor values are in [-1, 1]
    tensor = (tensor + 1) / 2.0
    
    # Convert to numpy array and transpose dimensions
    img = tensor.cpu().detach().numpy()
    img = np.transpose(img, (0, 2, 3, 1))
    
    # Clip values to [0, 1]
    img = np.clip(img, 0, 1)
    return img


def calculate_psnr(img1, img2):
    """Calculate PSNR between two images.
    
    Args:
        img1 (torch.Tensor): First image of shape (N, C, H, W)
        img2 (torch.Tensor): Second image of shape (N, C, H, W)
        
    Returns:
        float: Mean PSNR value
    """
    # Convert to numpy arrays
    img1_np = tensor_to_numpy(img1)
    img2_np = tensor_to_numpy(img2)
    
    # Calculate PSNR for each image pair
    psnr_values = []
    for i in range(img1_np.shape[0]):
        psnr_values.append(psnr(img1_np[i], img2_np[i], data_range=1.0))
    
    return np.mean(psnr_values)


def calculate_ssim(img1, img2):
    """Calculate SSIM between two images.
    
    Args:
        img1 (torch.Tensor): First image of shape (N, C, H, W)
        img2 (torch.Tensor): Second image of shape (N, C, H, W)
        
    Returns:
        float: Mean SSIM value
    """
    # Convert to numpy arrays
    img1_np = tensor_to_numpy(img1)
    img2_np = tensor_to_numpy(img2)
    
    # Calculate SSIM for each image pair
    ssim_values = []
    for i in range(img1_np.shape[0]):
        # Calculate SSIM per channel and then average
        ssim_vals = []
        for c in range(img1_np.shape[3]):
            ssim_vals.append(ssim(img1_np[i, :, :, c], img2_np[i, :, :, c], data_range=1.0))
        ssim_values.append(np.mean(ssim_vals))
    
    return np.mean(ssim_values)


class FID:
    """Frechet Inception Distance (FID) metric calculator.
    
    FID measures the distance between feature vectors calculated for real
    and generated images.
    """
    
    def __init__(self, device='cpu'):
        """Initialize the FID calculator.
        
        Args:
            device (str): Device to perform calculations on ('cpu' or 'cuda')
        """
        self.device = device
        
        # Load LPIPS model for feature extraction
        self.model = lpips.LPIPS(net='alex').to(device)
        self.model.eval()
    
    def calculate_statistics(self, images):
        """Calculate mean and covariance statistics of features.
        
        Args:
            images (torch.Tensor): Batch of images of shape (N, C, H, W)
            
        Returns:
            tuple: Mean and covariance matrices of features
        """
        features = []
        
        # Extract features in smaller batches to avoid memory issues
        batch_size = 16
        with torch.no_grad():
            for i in range(0, images.size(0), batch_size):
                batch = images[i:i+batch_size].to(self.device)
                # Extract intermediate features (before normalization)
                feat = self.model.net(batch)
                features.append(feat)
        
        # Concatenate all features
        features = torch.cat(features, dim=0)
        
        # Reshape features to vectors
        features = features.view(features.size(0), -1)
        
        # Calculate mean and covariance
        mu = torch.mean(features, dim=0)
        cov = torch.matmul((features - mu).T, (features - mu)) / (features.size(0) - 1)
        
        return mu, cov
    
    def calculate_frechet_distance(self, mu1, cov1, mu2, cov2, eps=1e-6):
        """Calculate Frechet Distance between two sets of statistics.
        
        Args:
            mu1 (torch.Tensor): Mean of features from first distribution
            cov1 (torch.Tensor): Covariance of features from first distribution
            mu2 (torch.Tensor): Mean of features from second distribution
            cov2 (torch.Tensor): Covariance of features from second distribution
            eps (float): Small constant for numerical stability
            
        Returns:
            float: Frechet distance
        """
        # Convert to numpy for easier computation
        mu1, cov1 = mu1.cpu().numpy(), cov1.cpu().numpy()
        mu2, cov2 = mu2.cpu().numpy(), cov2.cpu().numpy()
        
        diff = mu1 - mu2
        
        # Calculate sqrt of product of covariances
        covmean, _ = scipy.linalg.sqrtm(cov1.dot(cov2), disp=False)
        if not np.isfinite(covmean).all():
            offset = np.eye(cov1.shape[0]) * eps
            covmean = scipy.linalg.sqrtm((cov1 + offset).dot(cov2 + offset))
        
        # Check and correct imaginary components
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        
        # Calculate FID
        fid = diff.dot(diff) + np.trace(cov1) + np.trace(cov2) - 2 * np.trace(covmean)
        return fid
    
    def calculate_fid(self, real_images, generated_images):
        """Calculate FID between real and generated images.
        
        Args:
            real_images (torch.Tensor): Real images of shape (N, C, H, W)
            generated_images (torch.Tensor): Generated images of shape (N, C, H, W)
            
        Returns:
            float: FID score (lower is better)
        """
        # Calculate statistics for real and generated images
        mu_real, cov_real = self.calculate_statistics(real_images)
        mu_gen, cov_gen = self.calculate_statistics(generated_images)
        
        # Calculate FID
        fid = self.calculate_frechet_distance(mu_real, cov_real, mu_gen, cov_gen)
        
        return fid


class PerceptualLoss(nn.Module):
    """Perceptual loss using LPIPS for style transfer evaluation.
    
    This loss measures perceptual similarity between images using 
    the same feature representations as in FID.
    """
    
    def __init__(self, device='cpu'):
        """Initialize the perceptual loss.
        
        Args:
            device (str): Device to perform calculations on ('cpu' or 'cuda')
        """
        super(PerceptualLoss, self).__init__()
        self.loss_fn = lpips.LPIPS(net='alex').to(device)
    
    def forward(self, x, y):
        """Calculate perceptual loss between x and y.
        
        Args:
            x (torch.Tensor): First image batch of shape (N, C, H, W)
            y (torch.Tensor): Second image batch of shape (N, C, H, W)
            
        Returns:
            torch.Tensor: Perceptual loss value
        """
        return self.loss_fn(x, y) 