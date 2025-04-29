import argparse
import os
from pathlib import Path
import torch


class BaseOptions:
    """Base configuration class for both training and testing options.
    
    This class defines options used during both training and test time.
    It also implements several helper functions such as parsing, printing, and saving the options.
    """

    def __init__(self):
        """Initialize the BaseOptions class."""
        self.initialized = False
        self.isTrain = True

    def initialize(self, parser):
        """Define the common options that are used in both training and testing."""
        # Basic parameters
        parser.add_argument('--name', type=str, default='adain_cyclegan', help='name of the experiment')
        parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        
        # Data parameters
        parser.add_argument('--dataroot', type=str, required=True, help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
        parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        parser.add_argument('--load_size', type=int, default=286, help='scale images to this size')
        parser.add_argument('--crop_size', type=int, default=256, help='then crop to this size')
        parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels: 3 for RGB and 1 for grayscale')
        parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels: 3 for RGB and 1 for grayscale')
        parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order, otherwise takes them randomly')
        parser.add_argument('--batch_size', type=int, default=8, help='input batch size')
        parser.add_argument('--num_threads', default=4, type=int, help='# threads for loading data')
        parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset')
        parser.add_argument('--preprocess', type=str, default='resize_and_crop', help='scaling and cropping of images at load time [resize_and_crop | crop | scale_width | scale_width_and_crop | none]')
        parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data augmentation')
        
        # Network parameters
        parser.add_argument('--netG', type=str, default='adain_resnet_9blocks', help='specify generator architecture [adain_resnet_9blocks | resnet_9blocks | unet_256 | unet_128]')
        parser.add_argument('--netD', type=str, default='basic', help='specify discriminator architecture [basic | n_layers | pixel]')
        parser.add_argument('--n_layers_D', type=int, default=3, help='only used if netD==n_layers')
        parser.add_argument('--ngf', type=int, default=64, help='number of filters in the first conv layer')
        parser.add_argument('--ndf', type=int, default=64, help='number of filters in the first conv layer of discriminator')
        parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization [instance | batch | none]')
        parser.add_argument('--init_type', type=str, default='normal', help='network initialization [normal | xavier | kaiming | orthogonal]')
        parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')
        
        # AdaIN parameters
        parser.add_argument('--use_adain', action='store_true', help='use AdaIN in generator')
        parser.add_argument('--style_dim', type=int, default=512, help='dimension of style code')
        parser.add_argument('--style_encoder', type=str, default='vgg19', help='style encoder network [vgg19 | vgg16]')
        
        # Additional parameters
        parser.add_argument('--verbose', action='store_true', help='if specified, print more debugging information')
        parser.add_argument('--suffix', default='', type=str, help='customized suffix: opt.name = opt.name + suffix')
        
        self.initialized = True
        return parser

    def gather_options(self):
        """Initialize and collect the options from command-line arguments."""
        if not self.initialized:
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # Get the basic options
        opt, _ = parser.parse_known_args()

        # Save and return the parser
        self.parser = parser
        return parser.parse_args()

    def print_options(self, opt):
        """Print and save options
        
        It will print both current options and default values(if different).
        It will save options into a text file / [checkpoints_dir] / opt.txt
        """
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = f'\t[default: {default}]'
            message += f'{k:>25}: {str(v):<30}{comment}\n'
        message += '----------------- End -------------------'
        print(message)

        # Save to file
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        os.makedirs(expr_dir, exist_ok=True)
        with open(os.path.join(expr_dir, f'{opt.phase}_opt.txt'), 'w') as f:
            f.write(message)
            f.write('\n')

    def parse(self):
        """Parse options and set up GPU device."""
        opt = self.gather_options()
        opt.isTrain = self.isTrain   # train or test

        # Process opt.suffix
        if opt.suffix:
            suffix = ('_' + opt.suffix.format(**vars(opt))) if opt.suffix != '' else ''
            opt.name = opt.name + suffix

        self.print_options(opt)

        # Set up GPU device
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        
        # Check if CUDA is available and handle GPU device selection
        if len(opt.gpu_ids) > 0 and torch.cuda.is_available():
            try:
                torch.cuda.set_device(opt.gpu_ids[0])
            except Exception as e:
                print(f"Warning: Could not set CUDA device: {e}")
                print("Falling back to CPU")
                opt.gpu_ids = [-1]
        else:
            print("CUDA is not available. Using CPU")
            opt.gpu_ids = [-1]

        self.opt = opt
        return self.opt 