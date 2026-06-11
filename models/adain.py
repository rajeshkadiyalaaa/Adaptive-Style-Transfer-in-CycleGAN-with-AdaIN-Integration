import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaIN(nn.Module):
    """Adaptive Instance Normalization layer.
    
    This layer applies instance normalization to content features, then adapts their 
    mean and variance to match those of the style features.
    
    Reference:
    Huang, X., & Belongie, S. (2017). Arbitrary Style Transfer in Real-time with 
    Adaptive Instance Normalization. ICCV 2017.
    """
    
    def __init__(self, epsilon=1e-5):
        """Initialize the AdaIN layer.
        
        Args:
            epsilon (float): Small constant for numerical stability.
        """
        super(AdaIN, self).__init__()
        self.epsilon = epsilon
        
    def forward(self, content_feat, style_feat, alpha=1.0):
        """Forward pass of AdaIN.
        
        Args:
            content_feat (torch.Tensor): Content feature tensor of shape (B, C, H, W)
            style_feat (torch.Tensor): Style feature tensor of shape (B, C, H', W')
                                      or (B, N, C, H', W') where N is the number of
                                      style images per batch element. Channel count
                                      must match the content features.
            alpha (float): Style interpolation weight applied in feature space.
                           1.0 applies the full style statistics, 0.0 returns the
                           content features unchanged, values > 1.0 extrapolate.
        
        Returns:
            torch.Tensor: The normalized and modulated content feature tensor.
        """
        multi_style = style_feat.dim() == 5
        if multi_style:  # [B, N, C, H, W]
            B, N, C, H, W = style_feat.shape
            style_feat = style_feat.view(B * N, C, H, W)
        
        assert content_feat.size(1) == style_feat.size(1), (
            f'Channel mismatch between content ({content_feat.size(1)}) and '
            f'style ({style_feat.size(1)}) features. Project style features '
            f'before applying AdaIN.'
        )
        
        batch_size, channel_size = content_feat.size(0), content_feat.size(1)
        
        # Calculate content features statistics (mean and std)
        content_view = content_feat.view(batch_size, channel_size, -1)
        content_mean = content_view.mean(dim=2).view(batch_size, channel_size, 1, 1)
        content_std = content_view.std(dim=2).view(batch_size, channel_size, 1, 1) + self.epsilon
        content_feat_normalized = (content_feat - content_mean) / content_std
        
        # Calculate style statistics per style image, then average the statistics
        # over the N exemplars of each batch element so they align with the
        # content batch dimension.
        style_view = style_feat.view(style_feat.size(0), channel_size, -1)
        style_mean = style_view.mean(dim=2)
        style_std = style_view.std(dim=2)
        if multi_style:
            style_mean = style_mean.view(B, N, channel_size).mean(dim=1)
            style_std = style_std.view(B, N, channel_size).mean(dim=1)
        style_mean = style_mean.view(-1, channel_size, 1, 1)
        style_std = style_std.view(-1, channel_size, 1, 1) + self.epsilon
        
        # Adapt content features to style features
        adapted_features = style_std * content_feat_normalized + style_mean
        
        # Feature-space interpolation between content and stylized features
        if alpha != 1.0:
            adapted_features = content_feat + alpha * (adapted_features - content_feat)
        
        return adapted_features


class StyleEncoder(nn.Module):
    """Style encoder to extract style features.
    
    This module uses a pre-trained VGG-19 network to extract style features
    from style images, which are then used by the AdaIN layers.
    """
    
    def __init__(self, vgg_path=None):
        """Initialize style encoder.
        
        Args:
            vgg_path (str, optional): Path to pre-trained VGG-19 weights.
        """
        super(StyleEncoder, self).__init__()
        
        from torchvision import models
        vgg = models.vgg19(pretrained=True if vgg_path is None else False)
        if vgg_path is not None:
            vgg.load_state_dict(torch.load(vgg_path))
        
        # We use the first few layers of VGG-19 for style extraction
        self.encoder = nn.Sequential(
            *list(vgg.features.children())[:8]  # Extract features before ReLU activation
        )
        
        # Freeze the encoder parameters
        for param in self.encoder.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        """Extract style features from input image.
        
        Args:
            x (torch.Tensor): Input style image tensor of shape (B, N, 3, H, W)
                             where B is batch size and N is number of style images per batch
        
        Returns:
            torch.Tensor: Style features.
        """
        # Reshape input to handle multiple style images per batch
        multi_style = x.dim() == 5
        if multi_style:  # [B, N, C, H, W]
            B, N, C, H, W = x.shape
            x = x.view(B * N, C, H, W)
        
        # Extract features
        features = self.encoder(x)
        
        # Reshape back if we had multiple style images
        if multi_style:
            features = features.view(B, N, *features.shape[1:])
        
        return features


class AdaINResBlock(nn.Module):
    """Residual block with AdaIN for arbitrary style transfer.
    
    This residual block replaces instance normalization with AdaIN layers to
    enable arbitrary style transfer in CycleGAN generator networks.
    """
    
    def __init__(self, dim, use_adain=True, padding_type='reflect', norm_layer=nn.InstanceNorm2d,
                 style_nc=128):
        """Initialize AdaIN Residual Block.
        
        Args:
            dim (int): Number of channels in the input and output.
            use_adain (bool): Whether to use AdaIN layers.
            padding_type (str): Type of padding ('reflect', 'replicate', 'zero').
            norm_layer: Normalization layer.
            style_nc (int): Number of channels of the incoming style features
                            (128 for the VGG19-based StyleEncoder).
        """
        super(AdaINResBlock, self).__init__()
        self.use_adain = use_adain
        self.conv_block = self._build_conv_block(dim, padding_type, norm_layer)
        self.adain1 = AdaIN() if use_adain else None
        self.adain2 = AdaIN() if use_adain else None
        # Learned projection that maps style features to this block's channel
        # count. Registered here so it is trained with the generator (instead
        # of being recreated with random weights on every forward pass).
        self.style_proj = nn.Conv2d(style_nc, dim, kernel_size=1) if use_adain else None
    
    def _project_style(self, style_feat):
        """Project style features to this block's channel count."""
        if style_feat.dim() == 5:  # [B, N, C, H, W]
            B, N = style_feat.shape[:2]
            projected = self.style_proj(style_feat.flatten(0, 1))
            return projected.view(B, N, *projected.shape[1:])
        return self.style_proj(style_feat)
    
    def _build_conv_block(self, dim, padding_type, norm_layer):
        """Build the convolutional block.
        
        Args:
            dim (int): Number of channels.
            padding_type (str): Type of padding.
            norm_layer: Normalization layer.
            
        Returns:
            nn.Sequential: Convolutional block.
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError(f'padding type {padding_type} is not implemented')
        
        conv_block += [
            nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=True),
            norm_layer(dim) if not self.use_adain else nn.Identity(),
            nn.ReLU(True)
        ]
        
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError(f'padding type {padding_type} is not implemented')
        
        conv_block += [
            nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=True),
            norm_layer(dim) if not self.use_adain else nn.Identity()
        ]
        
        return nn.Sequential(*conv_block)
    
    def forward(self, x, style_feat=None, alpha=1.0):
        """Forward pass of AdaIN residual block.
        
        Args:
            x (torch.Tensor): Input feature tensor.
            style_feat (torch.Tensor, optional): Style feature tensor.
                                              Required if use_adain is True.
            alpha (float): Feature-space style interpolation weight.
        
        Returns:
            torch.Tensor: Output feature tensor with style transfer applied.
        """
        if self.use_adain:
            assert style_feat is not None, "Style features must be provided when use_adain is True"
            
            style_feat = self._project_style(style_feat)
            
            # Apply first convolution and AdaIN
            out = self.conv_block[0:3](x)  # Padding, Conv, Identity
            out = self.adain1(out, style_feat, alpha)
            out = F.relu(out)
            
            # Apply second convolution and AdaIN (index 3 is the ReLU module,
            # already applied above, so skip it)
            out = self.conv_block[4:7](out)  # Padding, Conv, Identity
            out = self.adain2(out, style_feat, alpha)
        else:
            out = self.conv_block(x)
        
        # Add residual connection
        return x + out 