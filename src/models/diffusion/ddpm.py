"""
Diffusion Policy Core Implementation

This module implements the Denoising Diffusion Probabilistic Model (DDPM)
for action sequence generation in end-to-end driving.

Based on:
- Diffusion Policy (Chi et al., RSS 2023)
- DDPM (Ho et al., NeurIPS 2020)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
import math


class SinusoidalPositionEmbedding(nn.Module):
    """Sinusoidal time step embedding for diffusion models"""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    
    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            timesteps: (B,) tensor of timestep indices
        Returns:
            embeddings: (B, dim) tensor of embeddings
        """
        device = timesteps.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = timesteps[:, None] * embeddings[None, :]
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        return embeddings


class Conv1dBlock(nn.Module):
    """1D Convolutional block with GroupNorm and activation"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 5,
        num_groups: int = 8,
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=kernel_size // 2
        )
        self.norm = nn.GroupNorm(num_groups, out_channels)
        self.activation = nn.Mish()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x


class ConditionalUNet1D(nn.Module):
    """
    1D U-Net for denoising action sequences conditioned on observations.
    
    Architecture:
    - Encoder: Downsampling blocks
    - Bottleneck: Processing at lowest resolution
    - Decoder: Upsampling blocks with skip connections
    - Conditioning: FiLM (Feature-wise Linear Modulation)
    """
    
    def __init__(
        self,
        action_dim: int = 3,
        action_horizon: int = 8,
        down_dims: Tuple[int, ...] = (256, 512, 1024),
        kernel_size: int = 5,
        num_groups: int = 8,
        condition_dim: int = 1024,
        time_embedding_dim: int = 128,
    ):
        super().__init__()
        
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        
        # Time embedding
        self.time_embedding = nn.Sequential(
            SinusoidalPositionEmbedding(time_embedding_dim),
            nn.Linear(time_embedding_dim, time_embedding_dim * 4),
            nn.Mish(),
            nn.Linear(time_embedding_dim * 4, time_embedding_dim),
        )
        
        # Condition embedding (from observation encoder)
        self.condition_embedding = nn.Sequential(
            nn.Linear(condition_dim, condition_dim // 2),
            nn.Mish(),
            nn.Linear(condition_dim // 2, condition_dim // 2),
        )
        condition_embed_dim = condition_dim // 2
        
        # Initial convolution
        self.input_conv = Conv1dBlock(action_dim, down_dims[0], kernel_size, num_groups)
        
        # Encoder (downsampling)
        self.down_blocks = nn.ModuleList()
        in_dim = down_dims[0]
        for out_dim in down_dims:
            self.down_blocks.append(nn.ModuleList([
                Conv1dBlock(in_dim, out_dim, kernel_size, num_groups),
                Conv1dBlock(out_dim, out_dim, kernel_size, num_groups),
                nn.Conv1d(out_dim, out_dim, 3, stride=2, padding=1),  # Downsample
            ]))
            in_dim = out_dim
        
        # Bottleneck
        mid_dim = down_dims[-1]
        self.mid_block = nn.ModuleList([
            Conv1dBlock(mid_dim, mid_dim, kernel_size, num_groups),
            Conv1dBlock(mid_dim, mid_dim, kernel_size, num_groups),
        ])
        
        # Decoder (upsampling)
        up_dims = list(reversed(down_dims))
        self.up_blocks = nn.ModuleList()
        in_dim = up_dims[0]
        for i, out_dim in enumerate(up_dims):
            # Skip connection doubles input channels
            skip_in_dim = in_dim * 2 if i > 0 else in_dim
            self.up_blocks.append(nn.ModuleList([
                nn.ConvTranspose1d(skip_in_dim, out_dim, 4, stride=2, padding=1),  # Upsample
                Conv1dBlock(out_dim, out_dim, kernel_size, num_groups),
                Conv1dBlock(out_dim, out_dim, kernel_size, num_groups),
            ]))
            in_dim = out_dim
        
        # Final output convolution
        self.output_conv = nn.Conv1d(up_dims[-1], action_dim, 1)
        
        # FiLM conditioning (modulation parameters)
        total_embed_dim = time_embedding_dim + condition_embed_dim
        self.film_layers = nn.ModuleList()
        for dim in list(down_dims) + list(reversed(down_dims)):
            self.film_layers.append(nn.Linear(total_embed_dim, dim * 2))  # scale and shift
    
    def forward(
        self,
        noisy_actions: torch.Tensor,
        timesteps: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass of the U-Net denoiser.
        
        Args:
            noisy_actions: (B, action_dim, action_horizon) noisy action sequence
            timesteps: (B,) diffusion timestep indices
            condition: (B, condition_dim) observation embedding
            
        Returns:
            predicted_noise: (B, action_dim, action_horizon) predicted noise
        """
        # Embed timesteps and condition
        time_embed = self.time_embedding(timesteps)  # (B, time_embed_dim)
        cond_embed = self.condition_embedding(condition)  # (B, cond_embed_dim)
        global_embed = torch.cat([time_embed, cond_embed], dim=1)  # (B, total_embed_dim)
        
        # Initial conv
        x = self.input_conv(noisy_actions)
        
        # Encoder with skip connections
        skip_connections = []
        film_idx = 0
        for down_block in self.down_blocks:
            conv1, conv2, downsample = down_block
            x = conv1(x)
            
            # Apply FiLM conditioning
            scale, shift = self.film_layers[film_idx](global_embed).chunk(2, dim=1)
            x = x * scale[:, :, None] + shift[:, :, None]
            film_idx += 1
            
            x = conv2(x)
            skip_connections.append(x)
            x = downsample(x)
        
        # Bottleneck
        for mid_conv in self.mid_block:
            x = mid_conv(x)
        
        # Decoder with skip connections
        for i, up_block in enumerate(self.up_blocks):
            upsample, conv1, conv2 = up_block
            x = upsample(x)
            
            # Add skip connection
            if i > 0:
                skip = skip_connections.pop()
                x = torch.cat([x, skip], dim=1)
            
            x = conv1(x)
            
            # Apply FiLM conditioning
            scale, shift = self.film_layers[film_idx](global_embed).chunk(2, dim=1)
            x = x * scale[:, :, None] + shift[:, :, None]
            film_idx += 1
            
            x = conv2(x)
        
        # Final output
        predicted_noise = self.output_conv(x)
        
        return predicted_noise


class DDPMScheduler:
    """
    Noise scheduler for DDPM.
    
    Implements:
    - Beta schedule (linear, cosine, etc.)
    - Forward diffusion (adding noise)
    - Reverse diffusion (denoising)
    """
    
    def __init__(
        self,
        num_train_timesteps: int = 100,
        beta_schedule: str = "squaredcos_cap_v2",
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
    ):
        self.num_train_timesteps = num_train_timesteps
        
        # Create beta schedule
        if beta_schedule == "linear":
            self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps)
        elif beta_schedule == "squaredcos_cap_v2":
            self.betas = self._cosine_beta_schedule(num_train_timesteps)
        else:
            raise ValueError(f"Unknown beta_schedule: {beta_schedule}")
        
        # Precompute useful quantities
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Coefficients for denoising
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # Coefficients for reverse process
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
    
    def _cosine_beta_schedule(self, timesteps: int, s: float = 0.008) -> torch.Tensor:
        """Cosine schedule as proposed in https://arxiv.org/abs/2102.09672"""
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)
    
    def add_noise(
        self,
        original: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward diffusion: add noise to original data.
        
        Args:
            original: (B, ...) original data
            noise: (B, ...) noise to add
            timesteps: (B,) timestep indices
            
        Returns:
            noisy: (B, ...) noised data
        """
        sqrt_alpha_prod = self.sqrt_alphas_cumprod[timesteps]
        sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alphas_cumprod[timesteps]
        
        # Reshape for broadcasting
        while len(sqrt_alpha_prod.shape) < len(original.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
        
        noisy = sqrt_alpha_prod * original + sqrt_one_minus_alpha_prod * noise
        return noisy
    
    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
    ) -> torch.Tensor:
        """
        Reverse diffusion: denoise by one step.
        
        Args:
            model_output: Predicted noise from model
            timestep: Current timestep
            sample: Current noisy sample
            
        Returns:
            prev_sample: Denoised sample at previous timestep
        """
        t = timestep
        
        # Predicted original sample
        alpha_prod_t = self.alphas_cumprod[t]
        beta_prod_t = 1 - alpha_prod_t
        
        pred_original_sample = (
            sample - beta_prod_t ** 0.5 * model_output
        ) / alpha_prod_t ** 0.5
        
        # Clip predicted original (optional, for stability)
        pred_original_sample = torch.clamp(pred_original_sample, -1.0, 1.0)
        
        # Compute previous sample mean
        alpha_prod_t_prev = self.alphas_cumprod_prev[t]
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        
        pred_sample_direction = (1 - alpha_prod_t_prev) ** 0.5 * model_output
        prev_sample_mean = (
            alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction
        )
        
        # Add noise (except for last step)
        if t > 0:
            noise = torch.randn_like(sample)
            variance = self.posterior_variance[t] ** 0.5
            prev_sample = prev_sample_mean + variance * noise
        else:
            prev_sample = prev_sample_mean
        
        return prev_sample


class DiffusionPolicy(nn.Module):
    """
    Complete Diffusion Policy for action generation.
    
    Combines:
    - Observation encoder
    - Conditional U-Net denoiser
    - DDPM scheduler
    """
    
    def __init__(
        self,
        observation_encoder: nn.Module,
        action_dim: int = 3,
        action_horizon: int = 8,
        num_diffusion_steps: int = 20,
        **unet_kwargs,
    ):
        super().__init__()
        
        self.obs_encoder = observation_encoder
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.num_diffusion_steps = num_diffusion_steps
        
        # Denoising U-Net
        self.denoiser = ConditionalUNet1D(
            action_dim=action_dim,
            action_horizon=action_horizon,
            **unet_kwargs,
        )
        
        # Noise scheduler
        self.scheduler = DDPMScheduler(num_train_timesteps=num_diffusion_steps)
    
    def forward(
        self,
        observations: Dict[str, torch.Tensor],
        actions: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training.
        
        Args:
            observations: Multi-modal observations
            actions: Ground truth actions (for training)
            
        Returns:
            Dictionary with loss and predicted actions
        """
        # Encode observations
        obs_embedding = self.obs_encoder(observations)  # (B, condition_dim)
        
        batch_size = obs_embedding.shape[0]
        device = obs_embedding.device
        
        if self.training and actions is not None:
            # Training: predict noise
            # Reshape actions: (B, horizon, action_dim) -> (B, action_dim, horizon)
            actions = actions.transpose(1, 2)
            
            # Sample random timesteps
            timesteps = torch.randint(
                0, self.num_diffusion_steps, (batch_size,),
                device=device, dtype=torch.long
            )
            
            # Add noise to actions
            noise = torch.randn_like(actions)
            noisy_actions = self.scheduler.add_noise(actions, noise, timesteps)
            
            # Predict noise
            predicted_noise = self.denoiser(noisy_actions, timesteps, obs_embedding)
            
            # Compute loss
            loss = F.mse_loss(predicted_noise, noise)
            
            return {'loss': loss, 'predicted_noise': predicted_noise}
        else:
            # Inference: sample actions through reverse diffusion
            sampled_actions = self.sample_actions(obs_embedding)
            return {'actions': sampled_actions}
    
    @torch.no_grad()
    def sample_actions(
        self,
        obs_embedding: torch.Tensor,
        num_inference_steps: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Sample action sequence through reverse diffusion.
        
        Args:
            obs_embedding: (B, condition_dim) observation embedding
            num_inference_steps: Number of denoising steps (default: use all)
            
        Returns:
            actions: (B, horizon, action_dim) sampled actions
        """
        batch_size = obs_embedding.shape[0]
        device = obs_embedding.device
        
        if num_inference_steps is None:
            num_inference_steps = self.num_diffusion_steps
        
        # Start from pure noise
        actions = torch.randn(
            batch_size, self.action_dim, self.action_horizon,
            device=device
        )
        
        # Iterative denoising
        timesteps = torch.linspace(
            self.num_diffusion_steps - 1, 0, num_inference_steps,
            device=device, dtype=torch.long
        )
        
        for t in timesteps:
            timestep_batch = t.repeat(batch_size)
            
            # Predict noise
            predicted_noise = self.denoiser(actions, timestep_batch, obs_embedding)
            
            # Denoise one step
            actions = self.scheduler.step(predicted_noise, t.item(), actions)
        
        # Reshape: (B, action_dim, horizon) -> (B, horizon, action_dim)
        actions = actions.transpose(1, 2)
        
        return actions
    
    def compute_log_prob(
        self,
        observations: Dict[str, torch.Tensor],
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute log probability of actions (for SAC).
        
        This is an approximation for diffusion policies.
        
        Args:
            observations: Multi-modal observations
            actions: Action sequence to evaluate
            
        Returns:
            log_prob: (B,) log probability of actions
        """
        # Encode observations
        obs_embedding = self.obs_encoder(observations)
        
        # Reshape actions
        actions = actions.transpose(1, 2)  # (B, horizon, action_dim) -> (B, action_dim, horizon)
        
        # Use noise prediction error as proxy for log prob
        # Lower prediction error = higher probability
        timesteps = torch.randint(
            0, self.num_diffusion_steps, (actions.shape[0],),
            device=actions.device, dtype=torch.long
        )
        
        noise = torch.randn_like(actions)
        noisy_actions = self.scheduler.add_noise(actions, noise, timesteps)
        
        predicted_noise = self.denoiser(noisy_actions, timesteps, obs_embedding)
        
        # Compute MSE per sample
        noise_error = F.mse_loss(predicted_noise, noise, reduction='none')
        noise_error = noise_error.mean(dim=[1, 2])  # Average over action_dim and horizon
        
        # Convert to log prob (approximation)
        log_prob = -noise_error
        
        return log_prob
