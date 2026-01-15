"""
Unified Logger for TensorBoard and Weights & Biases

Provides consistent logging interface across different backends.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Union
import json
from datetime import datetime
import wandb
from torch.utils.tensorboard import SummaryWriter
import logging
import sys


class Logger:
    """
    Unified logger supporting multiple backends.
    
    Supports:
    - TensorBoard
    - Weights & Biases (wandb)
    - Console logging
    - JSON file logging
    """
    
    def __init__(
        self,
        log_dir: Path,
        experiment_name: str,
        config: Optional[Dict] = None,
        use_tensorboard: bool = True,
        use_wandb: bool = False,
        wandb_project: Optional[str] = None,
        wandb_entity: Optional[str] = None,
        console_level: str = "INFO",
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.experiment_name = experiment_name
        self.config = config or {}
        
        # Initialize backends
        self.use_tensorboard = use_tensorboard
        self.use_wandb = use_wandb
        
        if self.use_tensorboard:
            tensorboard_dir = self.log_dir / "tensorboard"
            tensorboard_dir.mkdir(exist_ok=True)
            self.tensorboard = SummaryWriter(log_dir=str(tensorboard_dir))
        
        if self.use_wandb:
            wandb.init(
                project=wandb_project or "sac-diffusion-driving",
                entity=wandb_entity,
                name=experiment_name,
                config=config,
                dir=str(self.log_dir),
                reinit=True,
            )
            self.wandb = wandb
        
        # Console logger
        self.console_logger = self._setup_console_logger(console_level)
        
        # JSON log file
        self.json_log_path = self.log_dir / "metrics.jsonl"
        
        # Track step
        self.global_step = 0
    
    def _setup_console_logger(self, level: str) -> logging.Logger:
        """Setup console logger"""
        logger = logging.getLogger(self.experiment_name)
        logger.setLevel(getattr(logging, level.upper()))
        
        # Console handler
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(getattr(logging, level.upper()))
        
        # Formatter
        formatter = logging.Formatter(
            '[%(asctime)s] [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        
        logger.addHandler(handler)
        
        # File handler
        file_handler = logging.FileHandler(self.log_dir / "console.log")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        return logger
    
    def log_scalar(
        self,
        tag: str,
        value: Union[float, int, torch.Tensor],
        step: Optional[int] = None,
    ):
        """Log a scalar value"""
        if step is None:
            step = self.global_step
        
        # Convert tensor to float
        if isinstance(value, torch.Tensor):
            value = value.item()
        
        # TensorBoard
        if self.use_tensorboard:
            self.tensorboard.add_scalar(tag, value, step)
        
        # Wandb
        if self.use_wandb:
            self.wandb.log({tag: value}, step=step)
        
        # JSON log
        self._log_to_json({'step': step, 'tag': tag, 'value': value})
    
    def log_scalars(
        self,
        tag_prefix: str,
        values: Dict[str, Union[float, int, torch.Tensor]],
        step: Optional[int] = None,
    ):
        """Log multiple scalar values"""
        if step is None:
            step = self.global_step
        
        # Convert tensors
        values = {
            k: v.item() if isinstance(v, torch.Tensor) else v
            for k, v in values.items()
        }
        
        # TensorBoard
        if self.use_tensorboard:
            self.tensorboard.add_scalars(tag_prefix, values, step)
        
        # Wandb
        if self.use_wandb:
            log_dict = {f"{tag_prefix}/{k}": v for k, v in values.items()}
            self.wandb.log(log_dict, step=step)
    
    def log_histogram(
        self,
        tag: str,
        values: Union[np.ndarray, torch.Tensor],
        step: Optional[int] = None,
    ):
        """Log a histogram"""
        if step is None:
            step = self.global_step
        
        # Convert to numpy
        if isinstance(values, torch.Tensor):
            values = values.detach().cpu().numpy()
        
        # TensorBoard
        if self.use_tensorboard:
            self.tensorboard.add_histogram(tag, values, step)
        
        # Wandb
        if self.use_wandb:
            self.wandb.log({tag: wandb.Histogram(values)}, step=step)
    
    def log_image(
        self,
        tag: str,
        image: Union[np.ndarray, torch.Tensor],
        step: Optional[int] = None,
        dataformats: str = 'CHW',
    ):
        """Log an image"""
        if step is None:
            step = self.global_step
        
        # TensorBoard
        if self.use_tensorboard:
            if isinstance(image, np.ndarray):
                image = torch.from_numpy(image)
            self.tensorboard.add_image(tag, image, step, dataformats=dataformats)
        
        # Wandb
        if self.use_wandb:
            if isinstance(image, torch.Tensor):
                image = image.detach().cpu().numpy()
            
            # Convert to HWC format for wandb
            if dataformats == 'CHW':
                image = image.transpose(1, 2, 0)
            
            self.wandb.log({tag: wandb.Image(image)}, step=step)
    
    def log_video(
        self,
        tag: str,
        video: Union[np.ndarray, torch.Tensor],
        step: Optional[int] = None,
        fps: int = 30,
    ):
        """
        Log a video.
        
        Args:
            tag: Name of the video
            video: Video tensor of shape (T, C, H, W) or (T, H, W, C)
            step: Global step
            fps: Frames per second
        """
        if step is None:
            step = self.global_step
        
        # TensorBoard
        if self.use_tensorboard:
            if isinstance(video, np.ndarray):
                video = torch.from_numpy(video)
            
            # Add batch dimension: (1, T, C, H, W)
            if video.ndim == 4:
                video = video.unsqueeze(0)
            
            self.tensorboard.add_video(tag, video, step, fps=fps)
        
        # Wandb
        if self.use_wandb:
            if isinstance(video, torch.Tensor):
                video = video.detach().cpu().numpy()
            
            # Convert to (T, H, W, C) format
            if video.ndim == 5:
                video = video[0]  # Remove batch dim
            if video.shape[1] == 3:  # CHW format
                video = video.transpose(0, 2, 3, 1)
            
            self.wandb.log({tag: wandb.Video(video, fps=fps)}, step=step)
    
    def log_text(
        self,
        tag: str,
        text: str,
        step: Optional[int] = None,
    ):
        """Log text"""
        if step is None:
            step = self.global_step
        
        # TensorBoard
        if self.use_tensorboard:
            self.tensorboard.add_text(tag, text, step)
        
        # Wandb
        if self.use_wandb:
            self.wandb.log({tag: wandb.Html(f"<pre>{text}</pre>")}, step=step)
    
    def log_figure(
        self,
        tag: str,
        figure,
        step: Optional[int] = None,
    ):
        """Log a matplotlib figure"""
        if step is None:
            step = self.global_step
        
        # TensorBoard
        if self.use_tensorboard:
            self.tensorboard.add_figure(tag, figure, step)
        
        # Wandb
        if self.use_wandb:
            self.wandb.log({tag: wandb.Image(figure)}, step=step)
    
    def log_hyperparameters(self, hparams: Dict[str, Any]):
        """Log hyperparameters"""
        # TensorBoard
        if self.use_tensorboard:
            # Flatten nested dicts
            flat_hparams = self._flatten_dict(hparams)
            self.tensorboard.add_hparams(flat_hparams, {})
        
        # Wandb (already logged in init)
        if self.use_wandb:
            self.wandb.config.update(hparams)
    
    def _flatten_dict(self, d: Dict, parent_key: str = '', sep: str = '/') -> Dict:
        """Flatten nested dictionary"""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
    
    def _log_to_json(self, data: Dict):
        """Append data to JSON log file"""
        with open(self.json_log_path, 'a') as f:
            json.dump(data, f)
            f.write('\n')
    
    def info(self, message: str):
        """Log info message to console"""
        self.console_logger.info(message)
    
    def warning(self, message: str):
        """Log warning message to console"""
        self.console_logger.warning(message)
    
    def error(self, message: str):
        """Log error message to console"""
        self.console_logger.error(message)
    
    def debug(self, message: str):
        """Log debug message to console"""
        self.console_logger.debug(message)
    
    def increment_step(self, n: int = 1):
        """Increment global step counter"""
        self.global_step += n
    
    def set_step(self, step: int):
        """Set global step counter"""
        self.global_step = step
    
    def close(self):
        """Close all logging backends"""
        if self.use_tensorboard:
            self.tensorboard.close()
        
        if self.use_wandb:
            self.wandb.finish()
        
        self.info("Logger closed")


class MetricTracker:
    """
    Track and compute running statistics for metrics.
    """
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.metrics = {}
    
    def update(self, name: str, value: float):
        """Update metric with new value"""
        if name not in self.metrics:
            self.metrics[name] = []
        
        self.metrics[name].append(value)
        
        # Keep only last window_size values
        if len(self.metrics[name]) > self.window_size:
            self.metrics[name].pop(0)
    
    def get_mean(self, name: str) -> float:
        """Get mean of metric"""
        if name not in self.metrics or len(self.metrics[name]) == 0:
            return 0.0
        return np.mean(self.metrics[name])
    
    def get_std(self, name: str) -> float:
        """Get standard deviation of metric"""
        if name not in self.metrics or len(self.metrics[name]) == 0:
            return 0.0
        return np.std(self.metrics[name])
    
    def get_latest(self, name: str) -> float:
        """Get latest value of metric"""
        if name not in self.metrics or len(self.metrics[name]) == 0:
            return 0.0
        return self.metrics[name][-1]
    
    def get_all_means(self) -> Dict[str, float]:
        """Get means of all metrics"""
        return {name: self.get_mean(name) for name in self.metrics.keys()}
    
    def reset(self, name: Optional[str] = None):
        """Reset metric(s)"""
        if name is None:
            self.metrics = {}
        else:
            if name in self.metrics:
                self.metrics[name] = []
