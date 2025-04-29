import os
import random
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
from abc import ABC, abstractmethod
import numpy as np
import torch


def get_transform(opt, grayscale=False, convert=True, normalize=True):
    """Return a composition of image transforms.
    
    Args:
        opt: Command line options
        grayscale (bool): Whether to convert image to grayscale
        convert (bool): Whether to convert PIL Image to Tensor
        normalize (bool): Whether to normalize the tensor
        
    Returns:
        transform (torchvision.transforms.Compose): Composition of image transforms
    """
    transform_list = []
    if grayscale:
        transform_list.append(transforms.Grayscale(1))
    
    if 'resize' in opt.preprocess:
        transform_list.append(transforms.Resize(opt.load_size, Image.BICUBIC))
    
    if 'crop' in opt.preprocess:
        transform_list.append(transforms.RandomCrop(opt.crop_size))
    
    if not opt.no_flip:
        transform_list.append(transforms.RandomHorizontalFlip())
    
    if convert:
        transform_list.append(transforms.ToTensor())
    
    if normalize:
        transform_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    
    return transforms.Compose(transform_list)


class BaseDataset(data.Dataset, ABC):
    """Abstract base class for datasets.
    
    To create a subclass, you need to implement the following:
    -- <__init__>: Initialize the class, first call BaseDataset.__init__(self, opt).
    -- <__len__>: Return the size of dataset.
    -- <__getitem__>: Get a data point.
    """

    def __init__(self, opt):
        """Initialize the BaseDataset class.
        
        Args:
            opt: Command line options
        """
        self.opt = opt
        self.root = opt.dataroot

    @abstractmethod
    def __len__(self):
        """Return the total number of images."""
        return 0

    @abstractmethod
    def __getitem__(self, index):
        """Return a data point and its metadata information.
        
        Args:
            index (int): Index of the data point
            
        Returns:
            dict: A dictionary containing the data point
        """
        pass


class UnalignedStyleDataset(BaseDataset):
    """Dataset for CycleGAN with style images.
    
    This dataset loads unaligned/unpaired datasets for CycleGAN
    and also includes style images for AdaIN style transfer.
    """

    def __init__(self, opt):
        """Initialize the UnalignedStyleDataset class.
        
        Args:
            opt: Command line options
        """
        BaseDataset.__init__(self, opt)
        
        # Get directories for A, B, and style images
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')
        
        # Style directory is optional
        self.use_style = opt.use_adain if hasattr(opt, 'use_adain') else False
        self.dir_style = None
        if self.use_style:
            self.dir_style = os.path.join(opt.dataroot, 'style') if not hasattr(opt, 'style_dir') else opt.style_dir
        
        # Load paths for each domain
        self.A_paths = sorted(self._make_dataset(self.dir_A, opt.max_dataset_size))
        self.B_paths = sorted(self._make_dataset(self.dir_B, opt.max_dataset_size))
        
        # Load paths for style images if applicable
        self.style_paths = []
        if self.use_style and self.dir_style and os.path.exists(self.dir_style):
            self.style_paths = sorted(self._make_dataset(self.dir_style, opt.max_dataset_size))
        
        # Set the sizes
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
        self.style_size = len(self.style_paths)
        
        # Get transforms
        self.transform_A = get_transform(opt)
        self.transform_B = get_transform(opt)
        self.transform_style = get_transform(opt)
        
        # Style sampling method
        self.style_sample_method = opt.style_sample_method if hasattr(opt, 'style_sample_method') else 'random'
        self.n_style_images = opt.n_style_images if hasattr(opt, 'n_style_images') else 1
    
    def _make_dataset(self, dir, max_dataset_size=float("inf")):
        """Create a dataset from the directory.
        
        Args:
            dir (str): Directory path
            max_dataset_size (int): Maximum dataset size
            
        Returns:
            list: List of image paths
        """
        images = []
        assert os.path.isdir(dir), '%s is not a valid directory' % dir
        
        # Include common image extensions
        valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
        
        for root, _, fnames in sorted(os.walk(dir)):
            for fname in sorted(fnames):
                ext = os.path.splitext(fname)[1].lower()
                if ext in valid_extensions:
                    path = os.path.join(root, fname)
                    images.append(path)
        
        return images[:min(max_dataset_size, len(images))]
    
    def __getitem__(self, index):
        """Return a data point and its metadata information.
        
        Args:
            index (int): Index of the data point
            
        Returns:
            dict: A dictionary containing the data point
        """
        # Get A and B images
        A_path = self.A_paths[index % self.A_size]
        if self.opt.serial_batches:
            # In serial mode, we ensure that all batches follow the same order
            index_B = index % self.B_size
        else:
            # In random mode, we randomize the index for domain B
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        
        # Load and transform A and B images
        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')
        A = self.transform_A(A_img)
        B = self.transform_B(B_img)
        
        # Create result dictionary
        result = {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}
        
        # Handle style images if applicable
        if self.use_style and self.style_size > 0:
            if self.style_sample_method == 'random':
                # Randomly sample style images
                style_indices = random.sample(range(self.style_size), min(self.n_style_images, self.style_size))
            else:
                # Sample from the same class or use other methods
                # This would require additional class information
                style_indices = [index % self.style_size]
            
            style_imgs = []
            style_paths = []
            for style_idx in style_indices:
                style_path = self.style_paths[style_idx]
                style_img = Image.open(style_path).convert('RGB')
                style = self.transform_style(style_img)
                style_imgs.append(style)
                style_paths.append(style_path)
            
            # If we have multiple style images, stack them into a batch
            if len(style_imgs) > 1:
                style = torch.stack(style_imgs)
            else:
                style = style_imgs[0]
            
            result['style'] = style
            result['style_paths'] = style_paths
        
        return result
    
    def __len__(self):
        """Return the total number of images."""
        return max(self.A_size, self.B_size)


class SingleStyleDataset(BaseDataset):
    """Dataset for testing style transfer with single images."""
    
    def __init__(self, opt):
        """Initialize the SingleStyleDataset class.
        
        Args:
            opt: Command line options
        """
        BaseDataset.__init__(self, opt)
        
        # Get content and style image paths
        self.content_image = opt.content_image
        self.style_image = opt.style_image
        
        # Transformations
        self.transform = get_transform(opt)
    
    def __getitem__(self, _):
        """Return the content and style images.
        
        Returns:
            dict: A dictionary containing the content and style images
        """
        # Load content image
        content_img = Image.open(self.content_image).convert('RGB')
        content = self.transform(content_img)
        
        # Load style image if available
        style = None
        if self.style_image:
            style_img = Image.open(self.style_image).convert('RGB')
            style = self.transform(style_img)
        
        return {'content': content, 'style': style, 
                'content_path': self.content_image, 'style_path': self.style_image}
    
    def __len__(self):
        """Return the dataset size (1 for single image)."""
        return 1 