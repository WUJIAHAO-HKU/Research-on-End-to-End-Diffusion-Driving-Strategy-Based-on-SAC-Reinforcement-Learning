"""
Behavior Cloning (BC) Training Script

Pre-train diffusion policy with expert demonstrations.
"""

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import sys
from tqdm import tqdm

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.dataset import DemonstrationDataset, create_dataloader
from src.models.diffusion.ddpm import DiffusionPolicy
from src.models.encoders import create_vision_encoder, create_lidar_encoder, create_fusion_encoder
from src.utils.logger import Logger
from src.utils.checkpoint import CheckpointManager
from src.utils.visualization import TrainingVisualizer


class BehaviorCloningTrainer:
    """
    Trainer for behavior cloning with diffusion policy.
    """
    
    def __init__(self, config: DictConfig):
        self.config = config
        self.device = config.device
        
        # Create save directory
        self.save_dir = Path(config.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize logger
        self.logger = Logger(
            log_dir=self.save_dir / "logs",
            experiment_name=config.experiment_name,
            config=OmegaConf.to_container(config, resolve=True),
            use_tensorboard=config.logging.tensorboard,
            use_wandb=config.logging.wandb,
            wandb_project=config.logging.wandb_project,
        )
        
        # Build model
        self.build_model()
        
        # Build datasets
        self.build_datasets()
        
        # Checkpoint manager
        self.checkpoint_manager = CheckpointManager(
            save_dir=self.save_dir / "checkpoints",
            max_to_keep=config.training.max_checkpoints,
            metric_name="val_loss",
            mode="min",
        )
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        
        self.logger.info("Behavior Cloning Trainer initialized")
        self.logger.info(f"Total parameters: {self.count_parameters()}")
    
    def build_model(self):
        """Build diffusion policy and encoders"""
        # Vision encoder
        if self.config.model.use_vision:
            self.vision_encoder = create_vision_encoder(
                self.config.model.vision_encoder
            ).to(self.device)
            vision_dim = self.vision_encoder.output_dim
        else:
            self.vision_encoder = None
            vision_dim = 0
        
        # LiDAR encoder
        if self.config.model.use_lidar:
            self.lidar_encoder = create_lidar_encoder(
                self.config.model.lidar_encoder
            ).to(self.device)
            lidar_dim = self.lidar_encoder.output_dim
        else:
            self.lidar_encoder = None
            lidar_dim = 0
        
        # Fusion encoder
        if self.config.model.use_fusion:
            self.fusion_encoder = create_fusion_encoder(
                self.config.model.fusion_encoder
            ).to(self.device)
            condition_dim = self.fusion_encoder.output_dim
        else:
            # Simple concatenation
            condition_dim = vision_dim + lidar_dim + self.config.model.proprio_dim
        
        # Diffusion policy
        self.diffusion_policy = DiffusionPolicy(
            action_dim=self.config.model.action_dim,
            action_horizon=self.config.model.action_horizon,
            condition_dim=condition_dim,
            **self.config.model.diffusion_policy
        ).to(self.device)
        
        # Optimizer
        params = list(self.diffusion_policy.parameters())
        if self.vision_encoder is not None:
            params += list(self.vision_encoder.parameters())
        if self.lidar_encoder is not None:
            params += list(self.lidar_encoder.parameters())
        if hasattr(self, 'fusion_encoder'):
            params += list(self.fusion_encoder.parameters())
        
        self.optimizer = torch.optim.AdamW(
            params,
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.training.num_epochs,
            eta_min=self.config.training.min_learning_rate,
        )
    
    def build_datasets(self):
        """Build training and validation datasets"""
        # Training dataset
        self.train_loader = create_dataloader(
            data_path=Path(self.config.data.train_path),
            batch_size=self.config.training.batch_size,
            shuffle=True,
            num_workers=self.config.data.num_workers,
            observation_keys=self.config.data.observation_keys,
            action_dim=self.config.model.action_dim,
            chunk_size=self.config.model.action_horizon,
            augmentation=self.config.data.augmentation,
        )
        
        # Validation dataset
        self.val_loader = create_dataloader(
            data_path=Path(self.config.data.val_path),
            batch_size=self.config.training.batch_size,
            shuffle=False,
            num_workers=self.config.data.num_workers,
            observation_keys=self.config.data.observation_keys,
            action_dim=self.config.model.action_dim,
            chunk_size=self.config.model.action_horizon,
            augmentation=False,
        )
        
        self.logger.info(f"Training samples: {len(self.train_loader.dataset)}")
        self.logger.info(f"Validation samples: {len(self.val_loader.dataset)}")
    
    def encode_observations(self, observations: dict) -> torch.Tensor:
        """Encode multi-modal observations to condition vector"""
        features = {}
        
        # Vision
        if self.vision_encoder is not None and 'vision' in observations:
            features['vision'] = self.vision_encoder(observations['vision'])
        
        # LiDAR
        if self.lidar_encoder is not None and 'lidar' in observations:
            features['lidar'] = self.lidar_encoder(observations['lidar'])
        
        # Proprio
        if 'proprio' in observations:
            features['proprio'] = observations['proprio']
        
        # Fusion
        if hasattr(self, 'fusion_encoder'):
            condition = self.fusion_encoder(features)
        else:
            condition = torch.cat(list(features.values()), dim=1)
        
        return condition
    
    def train_epoch(self):
        """Train for one epoch"""
        self.diffusion_policy.train()
        if self.vision_encoder is not None:
            self.vision_encoder.train()
        if self.lidar_encoder is not None:
            self.lidar_encoder.train()
        
        epoch_loss = 0
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch}")
        
        for batch in pbar:
            observations = {
                k: v.to(self.device) for k, v in batch['observations'].items()
            }
            actions = batch['actions'].to(self.device)
            
            # Encode observations
            condition = self.encode_observations(observations)
            
            # Compute diffusion loss
            loss = self.diffusion_policy.compute_loss(actions, condition)
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            if self.config.training.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.diffusion_policy.parameters(),
                    self.config.training.grad_clip
                )
            
            self.optimizer.step()
            
            # Logging
            epoch_loss += loss.item()
            self.logger.log_scalar('train/loss', loss.item(), self.global_step)
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            self.global_step += 1
        
        avg_loss = epoch_loss / len(self.train_loader)
        return avg_loss
    
    @torch.no_grad()
    def validate(self):
        """Validate on validation set"""
        self.diffusion_policy.eval()
        if self.vision_encoder is not None:
            self.vision_encoder.eval()
        if self.lidar_encoder is not None:
            self.lidar_encoder.eval()
        
        val_loss = 0
        
        for batch in tqdm(self.val_loader, desc="Validation"):
            observations = {
                k: v.to(self.device) for k, v in batch['observations'].items()
            }
            actions = batch['actions'].to(self.device)
            
            # Encode observations
            condition = self.encode_observations(observations)
            
            # Compute loss
            loss = self.diffusion_policy.compute_loss(actions, condition)
            val_loss += loss.item()
        
        avg_val_loss = val_loss / len(self.val_loader)
        return avg_val_loss
    
    def train(self):
        """Main training loop"""
        self.logger.info("Starting training...")
        
        for epoch in range(self.config.training.num_epochs):
            self.epoch = epoch
            
            # Train
            train_loss = self.train_epoch()
            self.logger.log_scalar('epoch/train_loss', train_loss, epoch)
            
            # Validate
            val_loss = self.validate()
            self.logger.log_scalar('epoch/val_loss', val_loss, epoch)
            
            # Learning rate
            self.scheduler.step()
            self.logger.log_scalar('lr', self.scheduler.get_last_lr()[0], epoch)
            
            # Checkpoint
            self.checkpoint_manager.save(
                model=self.diffusion_policy,
                optimizer=self.optimizer,
                step=self.global_step,
                metric_value=val_loss,
                additional_state={
                    'epoch': epoch,
                    'scheduler': self.scheduler.state_dict(),
                },
            )
            
            self.logger.info(
                f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}"
            )
        
        self.logger.info("Training completed!")
        self.logger.close()
    
    def count_parameters(self):
        """Count total trainable parameters"""
        total = 0
        
        total += sum(p.numel() for p in self.diffusion_policy.parameters() if p.requires_grad)
        
        if self.vision_encoder is not None:
            total += sum(p.numel() for p in self.vision_encoder.parameters() if p.requires_grad)
        
        if self.lidar_encoder is not None:
            total += sum(p.numel() for p in self.lidar_encoder.parameters() if p.requires_grad)
        
        if hasattr(self, 'fusion_encoder'):
            total += sum(p.numel() for p in self.fusion_encoder.parameters() if p.requires_grad)
        
        return total


@hydra.main(config_path="../configs", config_name="bc_pretrain", version_base="1.2")
def main(config: DictConfig):
    """Main entry point"""
    # Print config
    print(OmegaConf.to_yaml(config))
    
    # Create trainer
    trainer = BehaviorCloningTrainer(config)
    
    # Train
    trainer.train()


if __name__ == "__main__":
    main()
