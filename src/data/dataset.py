"""
PyTorch Dataset for Imitation Learning

Loads expert demonstrations for behavior cloning.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import h5py
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pickle


class DemonstrationDataset(Dataset):
    """
    Dataset for loading expert demonstrations.
    
    Supports:
    - Multiple trajectory formats (HDF5, pickle, zarr)
    - Data augmentation
    - Trajectory chunking
    """
    
    def __init__(
        self,
        data_path: Path,
        observation_keys: List[str] = None,
        action_dim: int = 3,
        chunk_size: int = 16,
        augmentation: bool = True,
        normalize: bool = True,
    ):
        self.data_path = Path(data_path)
        self.observation_keys = observation_keys or ['vision', 'lidar', 'proprio']
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.augmentation = augmentation
        self.normalize = normalize
        
        # Load data
        self.trajectories = self._load_trajectories()
        
        # Build index
        self.trajectory_indices = self._build_index()
        
        # Compute normalization statistics
        if self.normalize:
            self.obs_mean, self.obs_std = self._compute_normalization_stats()
            self.action_mean, self.action_std = self._compute_action_stats()
    
    def _load_trajectories(self) -> List[Dict]:
        """Load all trajectories from disk"""
        trajectories = []
        
        if self.data_path.is_file():
            # Single file
            if self.data_path.suffix == '.hdf5' or self.data_path.suffix == '.h5':
                trajectories = self._load_hdf5(self.data_path)
            elif self.data_path.suffix == '.pkl':
                with open(self.data_path, 'rb') as f:
                    trajectories = pickle.load(f)
        else:
            # Directory of files
            for file_path in sorted(self.data_path.glob('*.pkl')):
                with open(file_path, 'rb') as f:
                    traj = pickle.load(f)
                    trajectories.append(traj)
        
        print(f"Loaded {len(trajectories)} trajectories from {self.data_path}")
        return trajectories
    
    def _load_hdf5(self, path: Path) -> List[Dict]:
        """Load trajectories from HDF5 file"""
        trajectories = []
        
        with h5py.File(path, 'r') as f:
            num_trajs = len(f.keys())
            
            for traj_idx in range(num_trajs):
                traj_group = f[f'traj_{traj_idx}']
                
                traj = {
                    'observations': {},
                    'actions': traj_group['actions'][:],
                    'rewards': traj_group['rewards'][:],
                    'dones': traj_group['dones'][:],
                }
                
                # Load observations
                obs_group = traj_group['observations']
                for key in self.observation_keys:
                    if key in obs_group:
                        traj['observations'][key] = obs_group[key][:]
                
                trajectories.append(traj)
        
        return trajectories
    
    def _build_index(self) -> List[Tuple[int, int]]:
        """
        Build index of (trajectory_idx, start_timestep) pairs.
        
        Allows sampling trajectory chunks efficiently.
        """
        indices = []
        
        for traj_idx, traj in enumerate(self.trajectories):
            traj_len = len(traj['actions'])
            
            # Create chunks
            for start_t in range(0, traj_len - self.chunk_size + 1, self.chunk_size // 2):
                indices.append((traj_idx, start_t))
        
        return indices
    
    def _compute_normalization_stats(self) -> Tuple[Dict, Dict]:
        """Compute mean and std for observations"""
        # Accumulate statistics
        obs_sums = {key: 0 for key in self.observation_keys}
        obs_sq_sums = {key: 0 for key in self.observation_keys}
        count = 0
        
        for traj in self.trajectories:
            traj_len = len(traj['actions'])
            count += traj_len
            
            for key in self.observation_keys:
                obs = traj['observations'][key]
                obs_sums[key] += obs.sum(axis=0)
                obs_sq_sums[key] += (obs ** 2).sum(axis=0)
        
        # Compute mean and std
        obs_mean = {key: obs_sums[key] / count for key in self.observation_keys}
        obs_std = {
            key: np.sqrt(obs_sq_sums[key] / count - obs_mean[key] ** 2)
            for key in self.observation_keys
        }
        
        # Avoid division by zero
        for key in self.observation_keys:
            obs_std[key] = np.maximum(obs_std[key], 1e-6)
        
        return obs_mean, obs_std
    
    def _compute_action_stats(self) -> Tuple[np.ndarray, np.ndarray]:
        """Compute mean and std for actions"""
        all_actions = np.concatenate([traj['actions'] for traj in self.trajectories], axis=0)
        
        action_mean = all_actions.mean(axis=0)
        action_std = all_actions.std(axis=0)
        action_std = np.maximum(action_std, 1e-6)
        
        return action_mean, action_std
    
    def __len__(self):
        return len(self.trajectory_indices)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a trajectory chunk.
        
        Returns:
            batch: Dictionary containing:
                - observations: Dict of (chunk_size, ...) tensors
                - actions: (chunk_size, action_dim) tensor
                - rewards: (chunk_size,) tensor (optional)
        """
        traj_idx, start_t = self.trajectory_indices[idx]
        traj = self.trajectories[traj_idx]
        
        end_t = min(start_t + self.chunk_size, len(traj['actions']))
        
        # Extract chunk
        observations = {}
        for key in self.observation_keys:
            obs = traj['observations'][key][start_t:end_t]
            
            # Normalize
            if self.normalize:
                obs = (obs - self.obs_mean[key]) / self.obs_std[key]
            
            observations[key] = torch.from_numpy(obs).float()
        
        actions = traj['actions'][start_t:end_t]
        
        # Normalize actions
        if self.normalize:
            actions = (actions - self.action_mean) / self.action_std
        
        actions = torch.from_numpy(actions).float()
        
        # Pad if necessary
        if end_t - start_t < self.chunk_size:
            pad_len = self.chunk_size - (end_t - start_t)
            
            for key in observations:
                pad_shape = (pad_len,) + observations[key].shape[1:]
                pad = torch.zeros(pad_shape, dtype=observations[key].dtype)
                observations[key] = torch.cat([observations[key], pad], dim=0)
            
            action_pad = torch.zeros((pad_len, self.action_dim), dtype=actions.dtype)
            actions = torch.cat([actions, action_pad], dim=0)
        
        # Apply augmentation
        if self.augmentation and self.training:
            observations = self._augment_observations(observations)
        
        return {
            'observations': observations,
            'actions': actions,
        }
    
    def _augment_observations(self, observations: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Apply data augmentation.
        
        - Vision: color jitter, random crop, gaussian noise
        - LiDAR: rotation, dropout, noise
        """
        # Vision augmentation
        if 'vision' in observations:
            # Random color jitter
            brightness = 0.2
            contrast = 0.2
            saturation = 0.2
            
            obs = observations['vision']
            if np.random.random() < 0.5:
                # Brightness
                factor = 1.0 + np.random.uniform(-brightness, brightness)
                obs = torch.clamp(obs * factor, 0, 1)
            
            if np.random.random() < 0.5:
                # Contrast
                factor = 1.0 + np.random.uniform(-contrast, contrast)
                mean = obs.mean(dim=[2, 3], keepdim=True)
                obs = torch.clamp((obs - mean) * factor + mean, 0, 1)
            
            # Gaussian noise
            if np.random.random() < 0.3:
                noise = torch.randn_like(obs) * 0.01
                obs = torch.clamp(obs + noise, 0, 1)
            
            observations['vision'] = obs
        
        # LiDAR augmentation
        if 'lidar' in observations:
            obs = observations['lidar']
            
            # Random rotation (around z-axis)
            if np.random.random() < 0.5:
                angle = np.random.uniform(-np.pi / 6, np.pi / 6)
                cos_a = np.cos(angle)
                sin_a = np.sin(angle)
                
                # Assuming points are (T, N, 3) where last dim is (x, y, z)
                x = obs[..., 0] * cos_a - obs[..., 1] * sin_a
                y = obs[..., 0] * sin_a + obs[..., 1] * cos_a
                obs = torch.stack([x, y, obs[..., 2]], dim=-1)
            
            # Point dropout
            if np.random.random() < 0.3:
                dropout_rate = 0.1
                mask = torch.rand(obs.shape[:-1]) > dropout_rate
                obs = obs * mask.unsqueeze(-1).float()
            
            # Gaussian noise
            if np.random.random() < 0.3:
                noise = torch.randn_like(obs) * 0.01
                obs = obs + noise
            
            observations['lidar'] = obs
        
        return observations
    
    def get_stats(self) -> Dict:
        """Get normalization statistics for denormalization"""
        return {
            'obs_mean': self.obs_mean,
            'obs_std': self.obs_std,
            'action_mean': self.action_mean,
            'action_std': self.action_std,
        }


def create_dataloader(
    data_path: Path,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    **dataset_kwargs
) -> DataLoader:
    """
    Create DataLoader for demonstration dataset.
    
    Args:
        data_path: Path to demonstration data
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        **dataset_kwargs: Additional arguments for DemonstrationDataset
        
    Returns:
        DataLoader instance
    """
    dataset = DemonstrationDataset(data_path, **dataset_kwargs)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    
    return dataloader
