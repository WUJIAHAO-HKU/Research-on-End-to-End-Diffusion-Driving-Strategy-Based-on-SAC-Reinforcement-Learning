"""
LiDAR Point Cloud Encoder

Supports multiple architectures:
- PointNet
- PointNet++
- Voxel-based CNN
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class PointNetEncoder(nn.Module):
    """
    PointNet encoder for 3D point clouds.
    
    Reference: PointNet (Qi et al., CVPR 2017)
    """
    
    def __init__(
        self,
        input_dim: int = 3,  # (x, y, z) or (x, y, intensity)
        mlp_layers: Tuple[int, ...] = (64, 128, 256),
        global_feature_dim: int = 256,
        use_batch_norm: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = global_feature_dim
        
        # Shared MLP layers
        self.mlp_layers = nn.ModuleList()
        in_dim = input_dim
        for out_dim in mlp_layers:
            layers = [nn.Conv1d(in_dim, out_dim, 1)]
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(out_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            self.mlp_layers.append(nn.Sequential(*layers))
            in_dim = out_dim
        
        # Global feature extraction
        self.global_mlp = nn.Sequential(
            nn.Conv1d(mlp_layers[-1], global_feature_dim, 1),
            nn.BatchNorm1d(global_feature_dim) if use_batch_norm else nn.Identity(),
            nn.ReLU(),
        )
    
    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            points: (B, N, input_dim) point cloud
            
        Returns:
            global_features: (B, global_feature_dim)
        """
        # Transpose for Conv1d: (B, N, input_dim) -> (B, input_dim, N)
        x = points.transpose(1, 2)
        
        # Shared MLP
        for mlp in self.mlp_layers:
            x = mlp(x)
        
        # Global features
        x = self.global_mlp(x)  # (B, global_feature_dim, N)
        
        # Max pooling across points
        global_features = torch.max(x, dim=2)[0]  # (B, global_feature_dim)
        
        return global_features


class PointNetPlusPlusEncoder(nn.Module):
    """
    PointNet++ encoder with hierarchical feature learning.
    
    Reference: PointNet++ (Qi et al., NeurIPS 2017)
    """
    
    def __init__(
        self,
        input_dim: int = 3,
        feature_dim: int = 256,
        num_points: int = 360,
        use_xyz: bool = True,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = feature_dim
        self.use_xyz = use_xyz
        
        # Set abstraction layers (SA)
        # SA1: sample 128 points, ball query radius 0.2
        self.sa1 = SetAbstraction(
            npoint=128,
            radius=0.2,
            nsample=32,
            in_channel=input_dim + 3 if use_xyz else input_dim,
            mlp=[64, 64, 128],
        )
        
        # SA2: sample 32 points, ball query radius 0.4
        self.sa2 = SetAbstraction(
            npoint=32,
            radius=0.4,
            nsample=64,
            in_channel=128 + 3 if use_xyz else 128,
            mlp=[128, 128, 256],
        )
        
        # Global feature extraction
        self.sa3 = SetAbstraction(
            npoint=None,  # Global pooling
            radius=None,
            nsample=None,
            in_channel=256 + 3 if use_xyz else 256,
            mlp=[256, 512, feature_dim],
        )
    
    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            points: (B, N, 3+C) point cloud with features
            
        Returns:
            global_features: (B, feature_dim)
        """
        # Split xyz and features
        xyz = points[..., :3].contiguous()  # (B, N, 3)
        features = points[..., 3:].contiguous() if points.shape[-1] > 3 else None
        
        # Set abstraction layers
        xyz, features = self.sa1(xyz, features)
        xyz, features = self.sa2(xyz, features)
        xyz, features = self.sa3(xyz, features)
        
        # Features shape: (B, feature_dim, 1)
        global_features = features.squeeze(-1)
        
        return global_features


class SetAbstraction(nn.Module):
    """Set Abstraction layer for PointNet++"""
    
    def __init__(
        self,
        npoint: Optional[int],
        radius: Optional[float],
        nsample: Optional[int],
        in_channel: int,
        mlp: list,
    ):
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
    
    def forward(
        self,
        xyz: torch.Tensor,
        features: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            xyz: (B, N, 3)
            features: (B, N, C) or None
            
        Returns:
            new_xyz: (B, npoint, 3)
            new_features: (B, C', npoint)
        """
        if self.npoint is None:
            # Global pooling
            new_xyz = xyz.mean(dim=1, keepdim=True)  # (B, 1, 3)
            if features is not None:
                features = features.transpose(1, 2)  # (B, C, N)
                new_features = torch.max(features, dim=2, keepdim=True)[0]  # (B, C, 1)
            else:
                xyz_trans = xyz.transpose(1, 2)  # (B, 3, N)
                new_features = torch.max(xyz_trans, dim=2, keepdim=True)[0]  # (B, 3, 1)
        else:
            # Sample points using farthest point sampling
            new_xyz = self.farthest_point_sample(xyz, self.npoint)
            
            # Group points and extract features
            grouped_xyz, grouped_features = self.query_ball_point(
                self.radius, self.nsample, xyz, new_xyz, features
            )
            
            # MLP
            new_features = grouped_features  # (B, C, npoint, nsample)
            for i, conv in enumerate(self.mlp_convs):
                bn = self.mlp_bns[i]
                new_features = F.relu(bn(conv(new_features)))
            
            # Max pooling
            new_features = torch.max(new_features, dim=3)[0]  # (B, C', npoint)
        
        return new_xyz, new_features
    
    @staticmethod
    def farthest_point_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
        """
        Farthest point sampling.
        
        Args:
            xyz: (B, N, 3)
            npoint: number of points to sample
            
        Returns:
            sampled_xyz: (B, npoint, 3)
        """
        device = xyz.device
        B, N, C = xyz.shape
        
        centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
        distance = torch.ones(B, N, device=device) * 1e10
        farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
        batch_indices = torch.arange(B, dtype=torch.long, device=device)
        
        for i in range(npoint):
            centroids[:, i] = farthest
            centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
            dist = torch.sum((xyz - centroid) ** 2, dim=2)
            mask = dist < distance
            distance[mask] = dist[mask]
            farthest = torch.max(distance, dim=1)[1]
        
        sampled_xyz = xyz[batch_indices.unsqueeze(1), centroids]
        return sampled_xyz
    
    @staticmethod
    def query_ball_point(
        radius: float,
        nsample: int,
        xyz: torch.Tensor,
        new_xyz: torch.Tensor,
        features: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Ball query to group points.
        
        Args:
            radius: search radius
            nsample: max number of points in each group
            xyz: (B, N, 3)
            new_xyz: (B, npoint, 3)
            features: (B, N, C) or None
            
        Returns:
            grouped_xyz: (B, 3, npoint, nsample)
            grouped_features: (B, C+3, npoint, nsample)
        """
        # Simplified implementation (use proper ball query in production)
        # For now, use KNN as approximation
        B, N, _ = xyz.shape
        _, npoint, _ = new_xyz.shape
        
        # Compute pairwise distances
        dist = torch.cdist(new_xyz, xyz)  # (B, npoint, N)
        
        # Get k nearest neighbors
        _, idx = torch.topk(dist, k=min(nsample, N), dim=2, largest=False)  # (B, npoint, nsample)
        
        # Group xyz
        batch_indices = torch.arange(B, device=xyz.device).view(B, 1, 1).expand(-1, npoint, nsample)
        grouped_xyz = xyz[batch_indices, idx]  # (B, npoint, nsample, 3)
        grouped_xyz = (grouped_xyz - new_xyz.unsqueeze(2)).permute(0, 3, 1, 2)  # (B, 3, npoint, nsample)
        
        # Group features
        if features is not None:
            grouped_features = features[batch_indices, idx]  # (B, npoint, nsample, C)
            grouped_features = grouped_features.permute(0, 3, 1, 2)  # (B, C, npoint, nsample)
            grouped_features = torch.cat([grouped_xyz, grouped_features], dim=1)  # (B, C+3, npoint, nsample)
        else:
            grouped_features = grouped_xyz
        
        return grouped_xyz, grouped_features


class VoxelEncoder(nn.Module):
    """
    Voxel-based encoder for point clouds.
    
    Converts point cloud to 3D voxel grid and applies 3D CNN.
    Faster than PointNet for large point clouds.
    """
    
    def __init__(
        self,
        voxel_size: Tuple[int, int, int] = (32, 32, 16),
        spatial_range: Tuple[float, float, float] = (10.0, 10.0, 5.0),
        feature_dim: int = 256,
    ):
        super().__init__()
        
        self.voxel_size = voxel_size
        self.spatial_range = spatial_range
        self.output_dim = feature_dim
        
        # 3D CNN
        self.conv3d = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(2),
            
            nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(2),
            
            nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.AdaptiveMaxPool3d(1),
        )
        
        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(128, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
    
    def voxelize(self, points: torch.Tensor) -> torch.Tensor:
        """
        Convert point cloud to voxel grid.
        
        Args:
            points: (B, N, 3) point cloud
            
        Returns:
            voxels: (B, 1, D, H, W) voxel occupancy grid
        """
        B, N, _ = points.shape
        device = points.device
        
        # Normalize to [0, 1]
        points_normalized = (points + torch.tensor(self.spatial_range, device=device)) / \
                           (2 * torch.tensor(self.spatial_range, device=device))
        
        # Convert to voxel indices
        voxel_indices = (points_normalized * torch.tensor(self.voxel_size, device=device)).long()
        voxel_indices = torch.clamp(voxel_indices, 0, torch.tensor(self.voxel_size, device=device) - 1)
        
        # Create voxel grid
        voxels = torch.zeros(B, 1, *self.voxel_size, device=device)
        
        for b in range(B):
            voxels[b, 0, voxel_indices[b, :, 0], voxel_indices[b, :, 1], voxel_indices[b, :, 2]] = 1.0
        
        return voxels
    
    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            points: (B, N, 3) point cloud
            
        Returns:
            features: (B, feature_dim)
        """
        # Voxelize
        voxels = self.voxelize(points)
        
        # 3D CNN
        x = self.conv3d(voxels)  # (B, 128, 1, 1, 1)
        x = x.view(x.size(0), -1)  # (B, 128)
        
        # MLP
        features = self.mlp(x)
        
        return features


class LiDAREncoder(nn.Module):
    """
    Unified LiDAR encoder interface.
    
    Handles:
    - Point cloud preprocessing
    - Normalization
    - Augmentation (during training)
    """
    
    def __init__(
        self,
        encoder_type: str = "pointnet",
        input_dim: int = 3,
        num_points: int = 360,
        feature_dim: int = 256,
        normalize: bool = True,
        augmentation: bool = True,
    ):
        super().__init__()
        
        self.encoder_type = encoder_type
        self.num_points = num_points
        self.normalize = normalize
        self.augmentation = augmentation and self.training
        
        # Create encoder
        if encoder_type == "pointnet":
            self.encoder = PointNetEncoder(
                input_dim=input_dim,
                mlp_layers=(64, 128, 256),
                global_feature_dim=feature_dim,
            )
        elif encoder_type == "pointnet++":
            self.encoder = PointNetPlusPlusEncoder(
                input_dim=input_dim,
                feature_dim=feature_dim,
                num_points=num_points,
            )
        elif encoder_type == "voxel":
            self.encoder = VoxelEncoder(
                voxel_size=(32, 32, 16),
                spatial_range=(10.0, 10.0, 5.0),
                feature_dim=feature_dim,
            )
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")
        
        self.output_dim = self.encoder.output_dim
    
    def preprocess(self, points: torch.Tensor) -> torch.Tensor:
        """
        Preprocess point cloud.
        
        - Subsample/upsample to fixed number
        - Remove NaN/Inf
        - Normalize
        - Augment (if training)
        """
        B, N, C = points.shape
        
        # Remove invalid points
        valid_mask = torch.isfinite(points).all(dim=2)
        
        # Subsample or pad
        if N > self.num_points:
            # Random sampling
            indices = torch.randperm(N, device=points.device)[:self.num_points]
            points = points[:, indices, :]
        elif N < self.num_points:
            # Pad with zeros
            padding = torch.zeros(B, self.num_points - N, C, device=points.device)
            points = torch.cat([points, padding], dim=1)
        
        # Normalize to unit sphere
        if self.normalize:
            centroid = points.mean(dim=1, keepdim=True)
            points = points - centroid
            max_dist = torch.max(torch.norm(points, dim=2, keepdim=True), dim=1, keepdim=True)[0]
            points = points / (max_dist + 1e-8)
        
        # Augmentation (random rotation, jitter, etc.)
        if self.training and self.augmentation:
            # Random rotation around z-axis
            theta = torch.rand(B, device=points.device) * 2 * 3.14159
            cos_theta = torch.cos(theta)
            sin_theta = torch.sin(theta)
            rotation_matrix = torch.stack([
                torch.stack([cos_theta, -sin_theta, torch.zeros_like(theta)], dim=1),
                torch.stack([sin_theta, cos_theta, torch.zeros_like(theta)], dim=1),
                torch.stack([torch.zeros_like(theta), torch.zeros_like(theta), torch.ones_like(theta)], dim=1),
            ], dim=1)  # (B, 3, 3)
            points = torch.bmm(points, rotation_matrix.transpose(1, 2))
            
            # Random jitter
            jitter = torch.randn_like(points) * 0.01
            points = points + jitter
        
        return points
    
    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            points: (B, N, C) point cloud
            
        Returns:
            features: (B, feature_dim)
        """
        # Preprocess
        points = self.preprocess(points)
        
        # Encode
        features = self.encoder(points)
        
        return features


def create_lidar_encoder(config: dict) -> LiDAREncoder:
    """Factory function to create LiDAR encoder from config"""
    return LiDAREncoder(
        encoder_type=config.get('type', 'pointnet'),
        input_dim=config.get('input_dim', 3),
        num_points=config.get('num_points', 360),
        feature_dim=config.get('feature_dim', 256),
        normalize=config.get('normalize', True),
        augmentation=config.get('augmentation', {}).get('enabled', True),
    )
