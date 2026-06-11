import torch
import torch.nn as nn
import itertools
import functools
from .adain import AdaIN, StyleEncoder, AdaINResBlock
import torch.nn.functional as F


class AdaINGenerator(nn.Module):
    """Generator network with AdaIN for arbitrary style transfer.
    
    This generator extends the standard CycleGAN generator with AdaIN layers
    to enable arbitrary style transfer without retraining.
    """
    
    def __init__(self, input_nc, output_nc, ngf=64, n_blocks=9, norm_layer=nn.InstanceNorm2d, 
                 use_dropout=False, padding_type='reflect', use_adain=True):
        """Initialize the AdaIN-enhanced generator.
        
        Args:
            input_nc (int): Number of input channels
            output_nc (int): Number of output channels
            ngf (int): Number of filters in the first conv layer
            n_blocks (int): Number of ResNet blocks
            norm_layer: Normalization layer
            use_dropout (bool): Whether to use dropout in the generator
            padding_type (str): Type of padding
            use_adain (bool): Whether to use AdaIN layers
        """
        super(AdaINGenerator, self).__init__()
        self.use_adain = use_adain
        
        # Initial convolution block
        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=True),
                 norm_layer(ngf),
                 nn.ReLU(True)]
        
        # Downsampling
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=True),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]
        
        # ResNet blocks
        mult = 2 ** n_downsampling
        self.res_blocks = nn.ModuleList()
        for i in range(n_blocks):
            self.res_blocks.append(
                AdaINResBlock(ngf * mult, use_adain=use_adain, padding_type=padding_type, norm_layer=norm_layer)
            )
        
        # Store the model before ResNet blocks
        self.model_down = nn.Sequential(*model)
        
        # Upsampling
        model = []
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2,
                                          padding=1, output_padding=1, bias=True),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        
        # Final output layer
        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
                  nn.Tanh()]
        
        # Store the model after ResNet blocks
        self.model_up = nn.Sequential(*model)
    
    def forward(self, x, style_feat=None, alpha=1.0):
        """Forward pass of the generator.
        
        Args:
            x (torch.Tensor): Input image tensor
            style_feat (torch.Tensor, optional): Style feature tensor
                                              Required if use_adain is True
            alpha (float): Feature-space style interpolation weight
                           (0.0 = content only, 1.0 = full style).
        
        Returns:
            torch.Tensor: Output image tensor
        """
        x = self.model_down(x)
        
        # Apply AdaIN ResNet blocks
        for res_block in self.res_blocks:
            x = res_block(x, style_feat, alpha) if self.use_adain else res_block(x)
        
        x = self.model_up(x)
        return x


class GANLoss(nn.Module):
    """LSGAN objective that builds target tensors matching the prediction shape."""
    
    def __init__(self):
        super(GANLoss, self).__init__()
        self.loss = nn.MSELoss()
    
    def forward(self, prediction, target_is_real):
        """Compute the LSGAN loss.
        
        Args:
            prediction (torch.Tensor): Discriminator output.
            target_is_real (bool): Whether the target label is real or fake.
        
        Returns:
            torch.Tensor: Loss value.
        """
        if target_is_real:
            target = torch.ones_like(prediction)
        else:
            target = torch.zeros_like(prediction)
        return self.loss(prediction, target)


class Discriminator(nn.Module):
    """Discriminator network for the CycleGAN.
    
    This discriminator uses a PatchGAN architecture.
    """
    
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.InstanceNorm2d):
        """Initialize the discriminator.
        
        Args:
            input_nc (int): Number of input channels
            ndf (int): Number of filters in the first conv layer
            n_layers (int): Number of convolutional layers
            norm_layer: Normalization layer
        """
        super(Discriminator, self).__init__()
        
        # No need to use bias as BatchNorm2d has affine parameters
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True)
        ]
        
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=2, padding=1, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]
        
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=1, padding=1, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=4, stride=1, padding=1)
        ]
        
        self.model = nn.Sequential(*sequence)
    
    def forward(self, input):
        """Forward pass of the discriminator.
        
        Args:
            input (torch.Tensor): Input image tensor
        
        Returns:
            torch.Tensor: Output prediction map
        """
        return self.model(input)


class AdaINStyleCycleGAN(nn.Module):
    """Complete AdaIN-enhanced CycleGAN model.
    
    This model integrates AdaIN layers into the CycleGAN architecture
    for user-defined style transfer.
    """
    
    def __init__(self, opt):
        """Initialize the AdaINStyleCycleGAN.
        
        Args:
            opt: Command line options
        """
        super(AdaINStyleCycleGAN, self).__init__()
        self.opt = opt
        
        # Set device based on GPU availability and opt.gpu_ids
        if opt.gpu_ids and opt.gpu_ids[0] >= 0 and torch.cuda.is_available():
            self.device = torch.device(f'cuda:{opt.gpu_ids[0]}')
        else:
            self.device = torch.device('cpu')
            
        self.use_adain = opt.use_adain if hasattr(opt, 'use_adain') else True
        
        # Specify input and output channels
        self.input_nc = opt.input_nc
        self.output_nc = opt.output_nc
        
        # Define networks
        # Generator A -> B
        self.netG_A = AdaINGenerator(opt.input_nc, opt.output_nc, opt.ngf, 
                                    norm_layer=nn.InstanceNorm2d, 
                                    use_dropout=False, use_adain=self.use_adain)
        
        # Generator B -> A
        self.netG_B = AdaINGenerator(opt.output_nc, opt.input_nc, opt.ngf, 
                                    norm_layer=nn.InstanceNorm2d, 
                                    use_dropout=False, use_adain=self.use_adain)
        
        # Discriminator A
        self.netD_A = Discriminator(opt.output_nc, opt.ndf, n_layers=3, 
                                   norm_layer=nn.InstanceNorm2d)
        
        # Discriminator B
        self.netD_B = Discriminator(opt.input_nc, opt.ndf, n_layers=3, 
                                   norm_layer=nn.InstanceNorm2d)
        
        # Style Encoder for AdaIN
        if self.use_adain:
            self.style_encoder = StyleEncoder()
        
        # Loss criteria
        self.criterionGAN = GANLoss()
        self.criterionCycle = nn.L1Loss()
        self.criterionIdt = nn.L1Loss()
        
        # Move networks to the specified device
        self.netG_A.to(self.device)
        self.netG_B.to(self.device)
        self.netD_A.to(self.device)
        self.netD_B.to(self.device)
        if self.use_adain:
            self.style_encoder.to(self.device)
        
        # Initialize weights
        self.init_weights()
        
    def init_weights(self):
        """Initialize network weights."""
        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif classname.find('BatchNorm2d') != -1:
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0.0)
        
        self.netG_A.apply(init_func)
        self.netG_B.apply(init_func)
        self.netD_A.apply(init_func)
        self.netD_B.apply(init_func)
    
    def set_input(self, input_data):
        """Set input data.
        
        Args:
            input_data (dict): Input data with keys 'A', 'B', and optionally 'style'
        """
        self.real_A = input_data['A'].to(self.device)
        self.real_B = input_data['B'].to(self.device)
        if 'style' in input_data and self.use_adain:
            self.style_img = input_data['style'].to(self.device)
    
    def forward(self):
        """Forward pass to compute the generated images.
        
        With AdaIN enabled, cycle consistency is made well-defined by
        conditioning each reverse mapping on the style statistics of the
        original source image: A -> B is stylized with the style exemplar,
        and B -> A reconstructs A using A's own style features. Without this,
        the cycle target would be ambiguous for arbitrary styles.
        """
        if self.use_adain and hasattr(self, 'style_img'):
            # Extract style features from the exemplar and from both sources
            self.style_features = self.style_encoder(self.style_img)
            self.style_A = self.style_encoder(self.real_A)
            self.style_B = self.style_encoder(self.real_B)
            
            # Generate fake images with style transfer
            self.fake_B = self.netG_A(self.real_A, self.style_features)
            self.fake_A = self.netG_B(self.real_B, self.style_A)
            
            # Reconstruct original images using the source images' own styles
            self.rec_A = self.netG_B(self.fake_B, self.style_A)
            self.rec_B = self.netG_A(self.fake_A, self.style_B)
        else:
            # Standard CycleGAN forward pass without style transfer
            self.fake_B = self.netG_A(self.real_A)
            self.fake_A = self.netG_B(self.real_B)
            self.rec_A = self.netG_B(self.fake_B)
            self.rec_B = self.netG_A(self.fake_A)
    
    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator.
        
        Args:
            netD (nn.Module): The discriminator network
            real (torch.Tensor): Real images
            fake (torch.Tensor): Fake images
            
        Returns:
            torch.Tensor: Discriminator loss
        """
        # Real
        loss_D_real = self.criterionGAN(netD(real), True)
        
        # Fake
        loss_D_fake = self.criterionGAN(netD(fake.detach()), False)
        
        # Combined loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D
    
    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A."""
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, self.fake_B)
    
    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B."""
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, self.fake_A)
    
    @staticmethod
    def _feature_stats(feat):
        """Compute per-channel mean and std of feature maps.
        
        Args:
            feat (torch.Tensor): Features of shape (B, C, H, W) or (B, N, C, H, W).
                                 For 5D inputs, statistics are averaged over N.
        
        Returns:
            tuple: (mean, std) tensors of shape (B, C)
        """
        if feat.dim() == 5:
            B, N, C = feat.shape[:3]
            flat = feat.view(B * N, C, -1)
            mean = flat.mean(dim=2).view(B, N, C).mean(dim=1)
            std = flat.std(dim=2).view(B, N, C).mean(dim=1)
        else:
            flat = feat.view(feat.size(0), feat.size(1), -1)
            mean = flat.mean(dim=2)
            std = flat.std(dim=2)
        return mean, std
    
    def criterionStyle(self, generated, style_features):
        """AdaIN-style loss: match mean/std of VGG features of the generated
        image to those of the style exemplar.
        
        Args:
            generated (torch.Tensor): Generated images of shape (B, C, H, W)
            style_features (torch.Tensor): Pre-computed style features
        
        Returns:
            torch.Tensor: Style loss value
        """
        gen_mean, gen_std = self._feature_stats(self.style_encoder(generated))
        style_mean, style_std = self._feature_stats(style_features.detach())
        return F.mse_loss(gen_mean, style_mean) + F.mse_loss(gen_std, style_std)
    
    def backward_G(self):
        """Calculate the loss for generators G_A and G_B."""
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        lambda_identity = self.opt.lambda_identity
        lambda_style = self.opt.lambda_style if hasattr(self.opt, 'lambda_style') else 0
        use_style = self.use_adain and hasattr(self, 'style_img')
        
        # Identity loss: feeding a target-domain image to a generator should
        # leave it (approximately) unchanged
        if lambda_identity > 0:
            if use_style:
                self.idt_A = self.netG_A(self.real_B, self.style_B)
                self.idt_B = self.netG_B(self.real_A, self.style_A)
            else:
                self.idt_A = self.netG_A(self.real_B)
                self.idt_B = self.netG_B(self.real_A)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_identity
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_identity
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0
        
        # GAN loss D_A(G_A(A))
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
        
        # GAN loss D_B(G_B(B))
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)
        
        # Forward cycle loss
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        
        # Backward cycle loss
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
        
        # Style loss (if using AdaIN)
        self.loss_style = 0
        if use_style and lambda_style > 0:
            self.loss_style = self.criterionStyle(self.fake_B, self.style_features) * lambda_style
        
        # Combined loss
        self.loss_G = (self.loss_G_A + self.loss_G_B
                       + self.loss_cycle_A + self.loss_cycle_B
                       + self.loss_idt_A + self.loss_idt_B
                       + self.loss_style)
        self.loss_G.backward()
    
    def optimize_parameters(self):
        """Optimize network parameters: one generator step, one discriminator step."""
        # Forward pass
        self.forward()
        
        # Update generators (discriminators frozen)
        self.set_requires_grad([self.netD_A, self.netD_B], False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        torch.nn.utils.clip_grad_norm_(self.netG_A.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(self.netG_B.parameters(), max_norm=1.0)
        self.optimizer_G.step()
        
        # Update discriminators
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()
        self.backward_D_A()
        self.backward_D_B()
        torch.nn.utils.clip_grad_norm_(self.netD_A.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(self.netD_B.parameters(), max_norm=1.0)
        self.optimizer_D.step()
    
    def set_requires_grad(self, nets, requires_grad=False):
        """Set requires_grad for all the networks.
        
        Args:
            nets (list): A list of networks
            requires_grad (bool): Whether requires_grad or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad 