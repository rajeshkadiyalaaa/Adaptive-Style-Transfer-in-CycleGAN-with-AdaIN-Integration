from .base_options import BaseOptions


class TestOptions(BaseOptions):
    """Test configuration options.
    
    This class includes test-specific options.
    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        
        # Testing parameters
        parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        parser.add_argument('--eval', action='store_true', help='use eval mode during test time.')
        parser.add_argument('--num_test', type=int, default=float("inf"), help='how many test images to run')
        
        # Style transfer parameters
        parser.add_argument('--style_image', type=str, default=None, help='path to style image for arbitrary style transfer')
        parser.add_argument('--content_image', type=str, default=None, help='path to content image for arbitrary style transfer')
        parser.add_argument('--output_path', type=str, default='output.jpg', help='save the result image to this path')
        
        # For evaluation
        parser.add_argument('--compute_metrics', action='store_true', help='compute evaluation metrics (PSNR, SSIM, FID)')
        parser.add_argument('--reference_dir', type=str, default=None, help='path to reference images for metric calculation')
        
        # AdaIN specific
        parser.add_argument('--style_interpolation', type=str, default=None, help='comma separated style interpolation weights, e.g. 0.3,0.7')
        parser.add_argument('--multiple_styles', action='store_true', help='use multiple style images for transfer')
        parser.add_argument('--style_dirs', type=str, default=None, help='comma separated paths to directories of style images')
        
        self.isTrain = False
        return parser 