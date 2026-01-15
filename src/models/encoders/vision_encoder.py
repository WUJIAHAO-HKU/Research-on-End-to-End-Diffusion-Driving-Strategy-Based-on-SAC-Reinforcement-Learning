"""
Vision Encoder for RGB and Depth Images

Supports multiple architectures:
- ResNet (18, 34, 50)
- EfficientNet (b0-b7)
- Vision Transformer (ViT)
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Tuple, Optional
import timm


class ResNetEncoder(nn.Module):
    """
    ResNet-based vision encoder for RGB-D images.
    
    Features:
    - Pretrained on ImageNet
    - Optional fine-tuning
    - Separate or fused RGB-D processing
    """
    
    def __init__(
        self,
        backbone: str = "resnet18",
        pretrained: bool = True,
        freeze_backbone: bool = False,
        input_channels: int = 4,  # RGB (3) + Depth (1)
        feature_dim: int = 512,
        spatial_features: bool = False,
    ):
        super().__init__()
        
        self.backbone_name = backbone
        self.feature_dim = feature_dim
        self.spatial_features = spatial_features
        
        # Load pretrained ResNet
        if backbone == "resnet18":
            resnet = models.resnet18(pretrained=pretrained)
            self.base_feature_dim = 512
        elif backbone == "resnet34":
            resnet = models.resnet34(pretrained=pretrained)
            self.base_feature_dim = 512
        elif backbone == "resnet50":
            resnet = models.resnet50(pretrained=pretrained)
            self.base_feature_dim = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Modify first conv layer for 4 channels (RGB-D)
        if input_channels != 3:
            original_conv = resnet.conv1
            self.conv1 = nn.Conv2d(
                input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
            # Initialize new channels with pretrained weights (average for depth)
            with torch.no_grad():
                self.conv1.weight[:, :3, :, :] = original_conv.weight
                if input_channels > 3:
                    # Initialize depth channel as average of RGB
                    self.conv1.weight[:, 3:, :, :] = original_conv.weight.mean(dim=1, keepdim=True)
        else:
            self.conv1 = resnet.conv1
        
        # Extract feature extractor (remove FC layer)
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.parameters():
                param.requires_grad = False
        
        # Projection head to desired feature dimension
        if spatial_features:
            # Keep spatial dimensions for attention mechanisms
            self.projection = nn.Sequential(
                nn.Conv2d(self.base_feature_dim, feature_dim, 1),
                nn.ReLU(),
            )
        else:
            # Global pooling to feature vector
            self.projection = nn.Sequential(
                nn.Linear(self.base_feature_dim, feature_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
            )
        
        self.output_dim = feature_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: (B, C, H, W) image tensor
            
        Returns:
            features: (B, feature_dim) or (B, feature_dim, H', W') if spatial_features=True
        """
        # Input shape: (B, 4, H, W) for RGB-D
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)  # (B, base_feature_dim, H/32, W/32)
        
        if self.spatial_features:
            # Return spatial features for attention
            x = self.projection(x)  # (B, feature_dim, H', W')
        else:
            # Global average pooling
            x = self.avgpool(x)  # (B, base_feature_dim, 1, 1)
            x = torch.flatten(x, 1)  # (B, base_feature_dim)
            x = self.projection(x)  # (B, feature_dim)
        
        return x


class EfficientNetEncoder(nn.Module):
    """
    EfficientNet-based vision encoder.
    
    More efficient than ResNet for similar accuracy.
    """
    
    def __init__(
        self,
        variant: str = "efficientnet_b0",
        pretrained: bool = True,
        freeze_backbone: bool = False,
        input_channels: int = 4,
        feature_dim: int = 512,
    ):
        super().__init__()
        
        # Load EfficientNet using timm
        self.backbone = timm.create_model(
            variant,
            pretrained=pretrained,
            in_chans=input_channels,
            num_classes=0,  # Remove classification head
            global_pool='avg'
        )
        
        # Get output dimension
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_channels, 224, 224)
            base_feature_dim = self.backbone(dummy_input).shape[1]
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Projection head
        self.projection = nn.Sequential(
            nn.Linear(base_feature_dim, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        
        self.output_dim = feature_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: (B, C, H, W) image tensor
            
        Returns:
            features: (B, feature_dim)
        """
        x = self.backbone(x)
        x = self.projection(x)
        return x


class ViTEncoder(nn.Module):
    """
    Vision Transformer (ViT) encoder.
    
    Good for capturing global context.
    """
    
    def __init__(
        self,
        variant: str = "vit_small_patch16_224",
        pretrained: bool = True,
        freeze_backbone: bool = False,
        input_channels: int = 4,
        feature_dim: int = 512,
    ):
        super().__init__()
        
        # Load ViT using timm
        self.backbone = timm.create_model(
            variant,
            pretrained=pretrained,
            in_chans=input_channels,
            num_classes=0,
            global_pool='token'
        )
        
        # Get output dimension
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_channels, 224, 224)
            base_feature_dim = self.backbone(dummy_input).shape[1]
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Projection head
        self.projection = nn.Sequential(
            nn.Linear(base_feature_dim, feature_dim),
            nn.ReLU(),
            nn.LayerNorm(feature_dim),
        )
        
        self.output_dim = feature_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: (B, C, H, W) image tensor
            
        Returns:
            features: (B, feature_dim)
        """
        x = self.backbone(x)
        x = self.projection(x)
        return x


class VisionEncoder(nn.Module):
    """
    Unified vision encoder interface.
    
    Automatically handles:
    - RGB-D fusion
    - Data augmentation (during training)
    - Normalization
    """
    
    def __init__(
        self,
        encoder_type: str = "resnet",
        backbone: str = "resnet18",
        pretrained: bool = True,
        freeze_backbone: bool = False,
        input_channels: int = 4,
        image_size: Tuple[int, int] = (240, 320),
        feature_dim: int = 512,
        augmentation: bool = True,
        spatial_features: bool = False,
    ):
        super().__init__()
        
        self.encoder_type = encoder_type
        self.image_size = image_size
        self.augmentation = augmentation and self.training
        
        # Create encoder
        if encoder_type == "resnet":
            self.encoder = ResNetEncoder(
                backbone=backbone,
                pretrained=pretrained,
                freeze_backbone=freeze_backbone,
                input_channels=input_channels,
                feature_dim=feature_dim,
                spatial_features=spatial_features,
            )
        elif encoder_type == "efficientnet":
            self.encoder = EfficientNetEncoder(
                variant=backbone,
                pretrained=pretrained,
                freeze_backbone=freeze_backbone,
                input_channels=input_channels,
                feature_dim=feature_dim,
            )
        elif encoder_type == "vit":
            self.encoder = ViTEncoder(
                variant=backbone,
                pretrained=pretrained,
                freeze_backbone=freeze_backbone,
                input_channels=input_channels,
                feature_dim=feature_dim,
            )
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")
        
        self.output_dim = self.encoder.output_dim
        
        # Normalization (ImageNet stats)
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406, 0.5]))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225, 0.25]))
        
        # Data augmentation
        if augmentation:
            self.aug_color_jitter = nn.Sequential(
                # Implemented as learnable params for differentiability
            )
    
    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize image to zero mean and unit variance"""
        # Assume input is [0, 1]
        mean = self.mean.view(1, -1, 1, 1)
        std = self.std.view(1, -1, 1, 1)
        return (x - mean) / std
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with automatic preprocessing.
        
        Args:
            x: (B, C, H, W) image tensor, expected in [0, 1] range
            
        Returns:
            features: (B, feature_dim) or (B, feature_dim, H', W')
        """
        # Normalize
        x = self.normalize(x)
        
        # Apply augmentation during training
        if self.training and self.augmentation:
            # Random crop (if image is larger than expected)
            # Color jitter, etc.
            pass
        
        # Encode
        features = self.encoder(x)
        
        return features


def create_vision_encoder(config: dict) -> VisionEncoder:
    """
    Factory function to create vision encoder from config.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Initialized vision encoder
    """
    return VisionEncoder(
        encoder_type=config.get('type', 'resnet'),
        backbone=config.get('backbone', 'resnet18'),
        pretrained=config.get('pretrained', True),
        freeze_backbone=config.get('freeze_backbone', False),
        input_channels=config.get('input_channels', 4),
        image_size=tuple(config.get('image_size', [240, 320])),
        feature_dim=config.get('feature_dim', 512),
        augmentation=config.get('augmentation', {}).get('enabled', True),
        spatial_features=config.get('spatial_features', False),
    )
