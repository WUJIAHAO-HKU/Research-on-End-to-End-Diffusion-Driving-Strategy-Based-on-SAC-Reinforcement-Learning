"""
Checkpoint Manager

Handles saving and loading of model checkpoints with automatic cleanup.
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Optional, List, Any
import json
import shutil
from datetime import datetime
import numpy as np


class CheckpointManager:
    """
    Manages model checkpoints with automatic cleanup.
    
    Features:
    - Save best K checkpoints
    - Periodic checkpointing
    - Automatic cleanup of old checkpoints
    - Resume training from checkpoint
    """
    
    def __init__(
        self,
        save_dir: Path,
        max_to_keep: int = 5,
        keep_every_n_hours: Optional[float] = None,
        metric_name: str = "eval/return",
        mode: str = "max",
    ):
        """
        Args:
            save_dir: Directory to save checkpoints
            max_to_keep: Maximum number of best checkpoints to keep
            keep_every_n_hours: Keep checkpoint every N hours regardless of metric
            metric_name: Metric name for best checkpoint selection
            mode: 'max' or 'min' for best checkpoint selection
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_to_keep = max_to_keep
        self.keep_every_n_hours = keep_every_n_hours
        self.metric_name = metric_name
        self.mode = mode
        
        # Track checkpoints
        self.checkpoints = []  # List of (metric_value, path, timestamp)
        self.last_periodic_save = datetime.now()
        
        # Load existing checkpoints
        self._load_checkpoint_registry()
    
    def save(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        step: int,
        metric_value: float,
        additional_state: Optional[Dict[str, Any]] = None,
        force: bool = False,
    ) -> Optional[Path]:
        """
        Save checkpoint.
        
        Args:
            model: Model to save
            optimizer: Optimizer to save
            step: Training step
            metric_value: Current metric value
            additional_state: Additional state to save
            force: Force save even if not best
            
        Returns:
            Path to saved checkpoint or None
        """
        # Check if should save
        should_save = force
        is_best = False
        is_periodic = False
        
        # Check if best
        if not should_save:
            is_best = self._is_best(metric_value)
            should_save = is_best
        
        # Check periodic save
        if not should_save and self.keep_every_n_hours is not None:
            hours_since_last = (datetime.now() - self.last_periodic_save).total_seconds() / 3600
            if hours_since_last >= self.keep_every_n_hours:
                should_save = True
                is_periodic = True
                self.last_periodic_save = datetime.now()
        
        if not should_save:
            return None
        
        # Create checkpoint
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_name = f"checkpoint_step_{step}_{timestamp}.pt"
        checkpoint_path = self.save_dir / checkpoint_name
        
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'step': step,
            'metric_value': metric_value,
            'metric_name': self.metric_name,
            'timestamp': timestamp,
        }
        
        if additional_state is not None:
            checkpoint.update(additional_state)
        
        # Save checkpoint
        torch.save(checkpoint, checkpoint_path)
        
        # Update registry
        self.checkpoints.append((metric_value, checkpoint_path, timestamp, is_periodic))
        
        # Cleanup old checkpoints
        if not is_periodic:
            self._cleanup_checkpoints()
        
        # Save registry
        self._save_checkpoint_registry()
        
        # Save best checkpoint link
        if is_best:
            best_path = self.save_dir / "best_checkpoint.pt"
            shutil.copy(checkpoint_path, best_path)
        
        return checkpoint_path
    
    def load(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        checkpoint_path: Optional[Path] = None,
        load_best: bool = True,
        device: str = "cpu",
    ) -> Dict[str, Any]:
        """
        Load checkpoint.
        
        Args:
            model: Model to load state into
            optimizer: Optimizer to load state into (optional)
            checkpoint_path: Path to specific checkpoint (optional)
            load_best: Load best checkpoint if path not specified
            device: Device to load checkpoint to
            
        Returns:
            Checkpoint dictionary
        """
        # Determine checkpoint path
        if checkpoint_path is None:
            if load_best:
                checkpoint_path = self.save_dir / "best_checkpoint.pt"
            else:
                # Load latest
                if not self.checkpoints:
                    raise ValueError("No checkpoints found")
                checkpoint_path = self.checkpoints[-1][1]
        
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        return checkpoint
    
    def _is_best(self, metric_value: float) -> bool:
        """Check if current metric is best"""
        if not self.checkpoints:
            return True
        
        best_metric = self._get_best_metric()
        
        if self.mode == "max":
            return metric_value > best_metric
        else:
            return metric_value < best_metric
    
    def _get_best_metric(self) -> float:
        """Get best metric value"""
        if not self.checkpoints:
            return -np.inf if self.mode == "max" else np.inf
        
        # Filter out periodic checkpoints
        non_periodic = [cp for cp in self.checkpoints if not cp[3]]
        
        if not non_periodic:
            return -np.inf if self.mode == "max" else np.inf
        
        metrics = [cp[0] for cp in non_periodic]
        
        if self.mode == "max":
            return max(metrics)
        else:
            return min(metrics)
    
    def _cleanup_checkpoints(self):
        """Remove old checkpoints beyond max_to_keep"""
        # Separate periodic and non-periodic checkpoints
        periodic = [cp for cp in self.checkpoints if cp[3]]
        non_periodic = [cp for cp in self.checkpoints if not cp[3]]
        
        # Sort non-periodic by metric
        if self.mode == "max":
            non_periodic.sort(key=lambda x: x[0], reverse=True)
        else:
            non_periodic.sort(key=lambda x: x[0])
        
        # Keep best max_to_keep
        to_keep = non_periodic[:self.max_to_keep]
        to_remove = non_periodic[self.max_to_keep:]
        
        # Remove old checkpoints
        for _, path, _, _ in to_remove:
            if path.exists():
                path.unlink()
        
        # Update registry
        self.checkpoints = to_keep + periodic
    
    def _save_checkpoint_registry(self):
        """Save checkpoint registry to disk"""
        registry_path = self.save_dir / "checkpoint_registry.json"
        
        registry = {
            'checkpoints': [
                {
                    'metric_value': float(metric),
                    'path': str(path),
                    'timestamp': timestamp,
                    'is_periodic': is_periodic,
                }
                for metric, path, timestamp, is_periodic in self.checkpoints
            ],
            'metric_name': self.metric_name,
            'mode': self.mode,
        }
        
        with open(registry_path, 'w') as f:
            json.dump(registry, f, indent=2)
    
    def _load_checkpoint_registry(self):
        """Load checkpoint registry from disk"""
        registry_path = self.save_dir / "checkpoint_registry.json"
        
        if not registry_path.exists():
            return
        
        with open(registry_path, 'r') as f:
            registry = json.load(f)
        
        self.checkpoints = [
            (
                cp['metric_value'],
                Path(cp['path']),
                cp['timestamp'],
                cp.get('is_periodic', False)
            )
            for cp in registry['checkpoints']
            if Path(cp['path']).exists()
        ]
        
        self.metric_name = registry.get('metric_name', self.metric_name)
        self.mode = registry.get('mode', self.mode)
    
    def get_best_checkpoint_path(self) -> Optional[Path]:
        """Get path to best checkpoint"""
        best_path = self.save_dir / "best_checkpoint.pt"
        return best_path if best_path.exists() else None
    
    def get_latest_checkpoint_path(self) -> Optional[Path]:
        """Get path to latest checkpoint"""
        if not self.checkpoints:
            return None
        return self.checkpoints[-1][1]
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """List all checkpoints"""
        return [
            {
                'metric_value': metric,
                'path': str(path),
                'timestamp': timestamp,
                'is_periodic': is_periodic,
            }
            for metric, path, timestamp, is_periodic in self.checkpoints
        ]


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    **kwargs
):
    """
    Simple checkpoint save function.
    
    Args:
        path: Path to save checkpoint
        model: Model to save
        optimizer: Optimizer to save (optional)
        **kwargs: Additional state to save
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
    }
    
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    
    checkpoint.update(kwargs)
    
    torch.save(checkpoint, path)


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    Simple checkpoint load function.
    
    Args:
        path: Path to checkpoint
        model: Model to load state into
        optimizer: Optimizer to load state into (optional)
        device: Device to load to
        
    Returns:
        Checkpoint dictionary
    """
    checkpoint = torch.load(path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint
