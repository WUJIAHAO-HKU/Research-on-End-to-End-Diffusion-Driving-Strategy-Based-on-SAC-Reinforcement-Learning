"""
Multi-Modal Fusion Encoder

Fuses visual, LiDAR, and proprioceptive features using:
- Concatenation + MLP
- Cross-attention
- FiLM (Feature-wise Linear Modulation)
- Gated fusion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import math


class CrossAttentionFusion(nn.Module):
    """
    Cross-attention based multi-modal fusion.
    
    Allows each modality to attend to other modalities.
    """
    
    def __init__(
        self,
        vision_dim: int = 512,
        lidar_dim: int = 256,
        proprio_dim: int = 8,
        hidden_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.vision_dim = vision_dim
        self.lidar_dim = lidar_dim
        self.proprio_dim = proprio_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Project all modalities to same dimension
        self.vision_proj = nn.Linear(vision_dim, hidden_dim)
        self.lidar_proj = nn.Linear(lidar_dim, hidden_dim)
        self.proprio_proj = nn.Linear(proprio_dim, hidden_dim)
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        # Layer norm
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # Feedforward
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )
        
        self.output_dim = hidden_dim
    
    def forward(
        self,
        vision_features: torch.Tensor,
        lidar_features: torch.Tensor,
        proprio_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass with cross-attention.
        
        Args:
            vision_features: (B, vision_dim)
            lidar_features: (B, lidar_dim)
            proprio_features: (B, proprio_dim)
            
        Returns:
            fused_features: (B, hidden_dim)
        """
        B = vision_features.size(0)
        
        # Project to same dimension
        vision_proj = self.vision_proj(vision_features)  # (B, hidden_dim)
        lidar_proj = self.lidar_proj(lidar_features)    # (B, hidden_dim)
        proprio_proj = self.proprio_proj(proprio_features)  # (B, hidden_dim)
        
        # Stack as sequence: (B, 3, hidden_dim)
        x = torch.stack([vision_proj, lidar_proj, proprio_proj], dim=1)
        
        # Self-attention across modalities
        attn_out, _ = self.attention(x, x, x)  # (B, 3, hidden_dim)
        x = self.norm1(x + attn_out)
        
        # Feedforward
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        
        # Average pooling across modalities
        fused = x.mean(dim=1)  # (B, hidden_dim)
        
        return fused


class FiLMFusion(nn.Module):
    """
    FiLM (Feature-wise Linear Modulation) based fusion.
    
    Use one modality to modulate another via affine transformation.
    """
    
    def __init__(
        self,
        vision_dim: int = 512,
        lidar_dim: int = 256,
        proprio_dim: int = 8,
        hidden_dim: int = 512,
    ):
        super().__init__()
        
        self.vision_dim = vision_dim
        self.lidar_dim = lidar_dim
        self.proprio_dim = proprio_dim
        self.hidden_dim = hidden_dim
        
        # Base feature extraction
        self.vision_base = nn.Sequential(
            nn.Linear(vision_dim, hidden_dim),
            nn.ReLU(),
        )
        
        self.lidar_base = nn.Sequential(
            nn.Linear(lidar_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # FiLM parameters from proprioception
        self.film_gamma = nn.Linear(proprio_dim, hidden_dim * 2)
        self.film_beta = nn.Linear(proprio_dim, hidden_dim * 2)
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        
        self.output_dim = hidden_dim
    
    def forward(
        self,
        vision_features: torch.Tensor,
        lidar_features: torch.Tensor,
        proprio_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass with FiLM.
        
        Args:
            vision_features: (B, vision_dim)
            lidar_features: (B, lidar_dim)
            proprio_features: (B, proprio_dim)
            
        Returns:
            fused_features: (B, hidden_dim)
        """
        # Extract base features
        vision_base = self.vision_base(vision_features)  # (B, hidden_dim)
        lidar_base = self.lidar_base(lidar_features)      # (B, hidden_dim)
        
        # Concatenate
        combined = torch.cat([vision_base, lidar_base], dim=1)  # (B, hidden_dim * 2)
        
        # FiLM modulation
        gamma = self.film_gamma(proprio_features)  # (B, hidden_dim * 2)
        beta = self.film_beta(proprio_features)    # (B, hidden_dim * 2)
        
        modulated = gamma * combined + beta
        
        # Output projection
        fused = self.output_proj(modulated)
        
        return fused


class GatedFusion(nn.Module):
    """
    Gated fusion with learned importance weights.
    
    Dynamically weights modalities based on input.
    """
    
    def __init__(
        self,
        vision_dim: int = 512,
        lidar_dim: int = 256,
        proprio_dim: int = 8,
        hidden_dim: int = 512,
    ):
        super().__init__()
        
        self.vision_dim = vision_dim
        self.lidar_dim = lidar_dim
        self.proprio_dim = proprio_dim
        self.hidden_dim = hidden_dim
        
        # Project to common dimension
        self.vision_proj = nn.Sequential(
            nn.Linear(vision_dim, hidden_dim),
            nn.ReLU(),
        )
        
        self.lidar_proj = nn.Sequential(
            nn.Linear(lidar_dim, hidden_dim),
            nn.ReLU(),
        )
        
        self.proprio_proj = nn.Sequential(
            nn.Linear(proprio_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Gating network
        total_dim = vision_dim + lidar_dim + proprio_dim
        self.gate_network = nn.Sequential(
            nn.Linear(total_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),  # 3 modalities
            nn.Softmax(dim=1),
        )
        
        self.output_dim = hidden_dim
    
    def forward(
        self,
        vision_features: torch.Tensor,
        lidar_features: torch.Tensor,
        proprio_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass with gated fusion.
        
        Args:
            vision_features: (B, vision_dim)
            lidar_features: (B, lidar_dim)
            proprio_features: (B, proprio_dim)
            
        Returns:
            fused_features: (B, hidden_dim)
        """
        # Project to common dimension
        vision_proj = self.vision_proj(vision_features)  # (B, hidden_dim)
        lidar_proj = self.lidar_proj(lidar_features)      # (B, hidden_dim)
        proprio_proj = self.proprio_proj(proprio_features)  # (B, hidden_dim)
        
        # Compute gates
        all_features = torch.cat([vision_features, lidar_features, proprio_features], dim=1)
        gates = self.gate_network(all_features)  # (B, 3)
        
        # Weighted sum
        fused = (
            gates[:, 0:1] * vision_proj +
            gates[:, 1:2] * lidar_proj +
            gates[:, 2:3] * proprio_proj
        )
        
        return fused


class ConcatMLPFusion(nn.Module):
    """
    Simple concatenation + MLP fusion.
    
    Fast and effective baseline.
    """
    
    def __init__(
        self,
        vision_dim: int = 512,
        lidar_dim: int = 256,
        proprio_dim: int = 8,
        hidden_dim: int = 512,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.vision_dim = vision_dim
        self.lidar_dim = lidar_dim
        self.proprio_dim = proprio_dim
        self.hidden_dim = hidden_dim
        
        # MLP layers
        total_dim = vision_dim + lidar_dim + proprio_dim
        layers = []
        in_dim = total_dim
        
        for _ in range(num_layers):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_dim = hidden_dim
        
        self.mlp = nn.Sequential(*layers)
        self.output_dim = hidden_dim
    
    def forward(
        self,
        vision_features: torch.Tensor,
        lidar_features: torch.Tensor,
        proprio_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass with concatenation.
        
        Args:
            vision_features: (B, vision_dim)
            lidar_features: (B, lidar_dim)
            proprio_features: (B, proprio_dim)
            
        Returns:
            fused_features: (B, hidden_dim)
        """
        # Concatenate all features
        combined = torch.cat([vision_features, lidar_features, proprio_features], dim=1)
        
        # MLP
        fused = self.mlp(combined)
        
        return fused


class FusionEncoder(nn.Module):
    """
    Unified multi-modal fusion encoder.
    
    Handles missing modalities gracefully.
    """
    
    def __init__(
        self,
        fusion_type: str = "cross_attention",
        vision_dim: int = 512,
        lidar_dim: int = 256,
        proprio_dim: int = 8,
        hidden_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.fusion_type = fusion_type
        
        # Create fusion module
        if fusion_type == "cross_attention":
            self.fusion = CrossAttentionFusion(
                vision_dim=vision_dim,
                lidar_dim=lidar_dim,
                proprio_dim=proprio_dim,
                hidden_dim=hidden_dim,
                dropout=dropout,
            )
        elif fusion_type == "film":
            self.fusion = FiLMFusion(
                vision_dim=vision_dim,
                lidar_dim=lidar_dim,
                proprio_dim=proprio_dim,
                hidden_dim=hidden_dim,
            )
        elif fusion_type == "gated":
            self.fusion = GatedFusion(
                vision_dim=vision_dim,
                lidar_dim=lidar_dim,
                proprio_dim=proprio_dim,
                hidden_dim=hidden_dim,
            )
        elif fusion_type == "concat":
            self.fusion = ConcatMLPFusion(
                vision_dim=vision_dim,
                lidar_dim=lidar_dim,
                proprio_dim=proprio_dim,
                hidden_dim=hidden_dim,
                dropout=dropout,
            )
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")
        
        self.output_dim = self.fusion.output_dim
        
        # Default values for missing modalities
        self.register_buffer('default_vision', torch.zeros(1, vision_dim))
        self.register_buffer('default_lidar', torch.zeros(1, lidar_dim))
        self.register_buffer('default_proprio', torch.zeros(1, proprio_dim))
    
    def forward(
        self,
        features: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Forward pass with automatic handling of missing modalities.
        
        Args:
            features: Dictionary containing:
                - 'vision': (B, vision_dim) or None
                - 'lidar': (B, lidar_dim) or None
                - 'proprio': (B, proprio_dim) or None
                
        Returns:
            fused_features: (B, hidden_dim)
        """
        # Get batch size
        B = None
        for v in features.values():
            if v is not None:
                B = v.size(0)
                break
        
        if B is None:
            raise ValueError("All modalities are None")
        
        # Get features with defaults
        vision_features = features.get('vision', self.default_vision.expand(B, -1))
        lidar_features = features.get('lidar', self.default_lidar.expand(B, -1))
        proprio_features = features.get('proprio', self.default_proprio.expand(B, -1))
        
        # Fuse
        fused = self.fusion(vision_features, lidar_features, proprio_features)
        
        return fused


class HierarchicalFusion(nn.Module):
    """
    Hierarchical fusion: fuse pairwise then globally.
    
    More robust to missing modalities.
    """
    
    def __init__(
        self,
        vision_dim: int = 512,
        lidar_dim: int = 256,
        proprio_dim: int = 8,
        hidden_dim: int = 512,
    ):
        super().__init__()
        
        # Pairwise fusion
        self.vision_lidar_fusion = nn.Sequential(
            nn.Linear(vision_dim + lidar_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        
        self.vision_proprio_fusion = nn.Sequential(
            nn.Linear(vision_dim + proprio_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        
        self.lidar_proprio_fusion = nn.Sequential(
            nn.Linear(lidar_dim + proprio_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        
        # Global fusion
        self.global_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        self.output_dim = hidden_dim
    
    def forward(
        self,
        vision_features: torch.Tensor,
        lidar_features: torch.Tensor,
        proprio_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass with hierarchical fusion.
        
        Args:
            vision_features: (B, vision_dim)
            lidar_features: (B, lidar_dim)
            proprio_features: (B, proprio_dim)
            
        Returns:
            fused_features: (B, hidden_dim)
        """
        # Pairwise fusion
        vl_fused = self.vision_lidar_fusion(
            torch.cat([vision_features, lidar_features], dim=1)
        )
        vp_fused = self.vision_proprio_fusion(
            torch.cat([vision_features, proprio_features], dim=1)
        )
        lp_fused = self.lidar_proprio_fusion(
            torch.cat([lidar_features, proprio_features], dim=1)
        )
        
        # Global fusion
        all_fused = torch.cat([vl_fused, vp_fused, lp_fused], dim=1)
        fused = self.global_fusion(all_fused)
        
        return fused


def create_fusion_encoder(config: dict) -> FusionEncoder:
    """Factory function to create fusion encoder from config"""
    return FusionEncoder(
        fusion_type=config.get('type', 'cross_attention'),
        vision_dim=config.get('vision_dim', 512),
        lidar_dim=config.get('lidar_dim', 256),
        proprio_dim=config.get('proprio_dim', 8),
        hidden_dim=config.get('hidden_dim', 512),
        dropout=config.get('dropout', 0.1),
    )
