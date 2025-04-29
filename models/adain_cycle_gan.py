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
    
    def forward(self, x, style_feat=None):
        """Forward pass of the generator.
        
        Args:
            x (torch.Tensor): Input image tensor
            style_feat (torch.Tensor, optional): Style feature tensor
                                              Required if use_adain is True
        
        Returns:
            torch.Tensor: Output image tensor
        """
        x = self.model_down(x)
        
        # Apply AdaIN ResNet blocks
        for res_block in self.res_blocks:
            x = res_block(x, style_feat) if self.use_adain else res_block(x)
        
        x = self.model_up(x)
        return x


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
        """Forward pass to compute the generated images."""
        if self.use_adain and hasattr(self, 'style_img'):
            # Extract style features
            self.style_features = self.style_encoder(self.style_img)
            
            # Generate fake images with style transfer
            self.fake_B = self.netG_A(self.real_A, self.style_features)
            self.fake_A = self.netG_B(self.real_B, self.style_features)
            
            # Reconstruct original images
            self.rec_A = self.netG_B(self.fake_B, self.style_features)
            self.rec_B = self.netG_A(self.fake_A, self.style_features)
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
        pred_real = netD(real)
        loss_D_real = F.mse_loss(pred_real, torch.ones_like(pred_real))
        
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = F.mse_loss(pred_fake, torch.zeros_like(pred_fake))
        
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
    
    def backward_G(self):
        """Calculate the loss for generators G_A and G_B."""
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        lambda_identity = self.opt.lambda_identity
        lambda_style = self.opt.lambda_style if hasattr(self.opt, 'lambda_style') else 0
        
        # GAN loss D_A(G_A(A))
        self.loss_G_A = F.mse_loss(self.netD_A(self.fake_B), torch.ones_like(self.netD_A(self.fake_B)))
        
        # GAN loss D_B(G_B(B))
        self.loss_G_B = F.mse_loss(self.netD_B(self.fake_A), torch.ones_like(self.netD_B(self.fake_A)))
        
        # Forward cycle loss
        self.loss_cycle_A = lambda_A * F.l1_loss(self.rec_A, self.real_A)
        
        # Backward cycle loss
        self.loss_cycle_B = lambda_B * F.l1_loss(self.rec_B, self.real_B)
        
        # Style loss (if using AdaIN)
        self.loss_style = 0
        if self.use_adain and lambda_style > 0 and hasattr(self, 'style_img'):
            # Extract features from fake_B and style image
            fake_B_features = self.style_encoder(self.fake_B)
            style_features = self.style_features.detach()  # Don't backprop through style features
            
            # Calculate mean and std for style loss
            fake_mean = fake_B_features.view(fake_B_features.size(0), fake_B_features.size(1), -1).mean(dim=2)
            fake_std = fake_B_features.view(fake_B_features.size(0), fake_B_features.size(1), -1).std(dim=2)
            style_mean = style_features.view(style_features.size(0), style_features.size(1), -1).mean(dim=2)
            style_std = style_features.view(style_features.size(0), style_features.size(1), -1).std(dim=2)
            
            # Mean and std loss
            self.loss_style = lambda_style * (F.mse_loss(fake_mean, style_mean) + F.mse_loss(fake_std, style_std))
        
        # Combined loss
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_style
        self.loss_G.backward()
    
    def optimize_parameters(self):
        """Optimize network parameters."""
        # Forward pass
        self.forward()
        
        # Set G_A and G_B's gradients to zero
        self.optimizer_G.zero_grad()
        
        # Calculate GAN and cycle losses
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)
        
        # Forward cycle loss
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * self.opt.lambda_A
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * self.opt.lambda_B
        
        # Style loss if using AdaIN
        if self.use_adain:
            self.loss_style = self.criterionStyle(self.fake_B, self.style_features) * self.opt.lambda_style
        else:
            self.loss_style = 0
        
        # Combined generator loss
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_style
        
        # Backward pass
        self.loss_G.backward()
        
        # Clip gradients for stability
        torch.nn.utils.clip_grad_norm_(self.netG_A.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(self.netG_B.parameters(), max_norm=1.0)
        
        # Update generator parameters
        self.optimizer_G.step()
        
        # Set D_A and D_B's gradients to zero
        self.optimizer_D.zero_grad()
        
        # Calculate discriminator losses
        self.loss_D_A = self.criterionGAN(self.netD_A(self.real_B), True) + \
                       self.criterionGAN(self.netD_A(self.fake_B.detach()), False)
        self.loss_D_B = self.criterionGAN(self.netD_B(self.real_A), True) + \
                       self.criterionGAN(self.netD_B(self.fake_A.detach()), False)
        
        # Combined discriminator loss
        self.loss_D = (self.loss_D_A + self.loss_D_B) * 0.5
        
        # Backward pass
        self.loss_D.backward()
        
        # Clip gradients for stability
        torch.nn.utils.clip_grad_norm_(self.netD_A.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(self.netD_B.parameters(), max_norm=1.0)
        
        # Update discriminator parameters
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