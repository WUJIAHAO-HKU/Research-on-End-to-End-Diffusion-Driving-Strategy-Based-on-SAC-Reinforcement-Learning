"""
Soft Actor-Critic with Gaussian Policy

Standard SAC baseline for comparison with SAC-Diffusion.
Reference: Haarnoja et al., ICML 2018
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
from typing import Dict, Tuple, Optional
import copy


LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
EPSILON = 1e-6


class GaussianActor(nn.Module):
    """
    Stochastic Gaussian policy network.
    
    Outputs mean and log_std for Gaussian distribution.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: Tuple[int, ...] = (256, 256),
        max_action: float = 1.0,
    ):
        super().__init__()
        
        self.max_action = max_action
        
        # Shared layers
        layers = []
        in_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
            ])
            in_dim = hidden_dim
        
        self.shared = nn.Sequential(*layers)
        
        # Mean and log_std heads
        self.mean = nn.Linear(in_dim, action_dim)
        self.log_std = nn.Linear(in_dim, action_dim)
    
    def forward(
        self,
        state: torch.Tensor,
        deterministic: bool = False,
        return_log_prob: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            state: State tensor
            deterministic: Use mean action (no sampling)
            return_log_prob: Return log probability
            
        Returns:
            action: Sampled or mean action
            log_prob: Log probability (optional)
        """
        x = self.shared(state)
        
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
        std = log_std.exp()
        
        # Create distribution
        normal = Normal(mean, std)
        
        if deterministic:
            # Use mean
            action_pre_tanh = mean
        else:
            # Sample with reparameterization trick
            action_pre_tanh = normal.rsample()
        
        # Apply tanh and scale
        action = torch.tanh(action_pre_tanh) * self.max_action
        
        if return_log_prob:
            # Compute log probability with tanh correction
            log_prob = normal.log_prob(action_pre_tanh)
            
            # Apply tanh correction: log_prob - log(1 - tanh^2(x))
            log_prob = log_prob - torch.log(
                self.max_action * (1 - torch.tanh(action_pre_tanh).pow(2)) + EPSILON
            )
            log_prob = log_prob.sum(dim=1, keepdim=True)
            
            return action, log_prob
        
        return action, None
    
    def get_log_prob(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Compute log probability of given action"""
        x = self.shared(state)
        
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
        std = log_std.exp()
        
        normal = Normal(mean, std)
        
        # Inverse tanh
        action_pre_tanh = torch.atanh(action / self.max_action)
        
        log_prob = normal.log_prob(action_pre_tanh)
        log_prob = log_prob - torch.log(
            self.max_action * (1 - (action / self.max_action).pow(2)) + EPSILON
        )
        log_prob = log_prob.sum(dim=1, keepdim=True)
        
        return log_prob


class QNetwork(nn.Module):
    """Q-function network"""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: Tuple[int, ...] = (256, 256),
    ):
        super().__init__()
        
        layers = []
        in_dim = state_dim + action_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
            ])
            in_dim = hidden_dim
        
        layers.append(nn.Linear(in_dim, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], dim=1)
        return self.network(x)


class SACGaussianAgent:
    """
    SAC with Gaussian policy.
    
    Features:
    - Twin Q-networks
    - Automatic entropy tuning
    - Squashed Gaussian policy
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_action: float = 1.0,
        hidden_dims: Tuple[int, ...] = (256, 256),
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        alpha_lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        auto_entropy_tuning: bool = True,
        target_entropy: Optional[float] = None,
        device: str = "cuda",
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.gamma = gamma
        self.tau = tau
        self.device = device
        
        # Actor
        self.actor = GaussianActor(
            state_dim, action_dim, hidden_dims, max_action
        ).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        
        # Twin Critics
        self.critic1 = QNetwork(state_dim, action_dim, hidden_dims).to(device)
        self.critic1_target = copy.deepcopy(self.critic1)
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=critic_lr)
        
        self.critic2 = QNetwork(state_dim, action_dim, hidden_dims).to(device)
        self.critic2_target = copy.deepcopy(self.critic2)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=critic_lr)
        
        # Automatic entropy tuning
        self.auto_entropy_tuning = auto_entropy_tuning
        if auto_entropy_tuning:
            if target_entropy is None:
                self.target_entropy = -action_dim
            else:
                self.target_entropy = target_entropy
            
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)
            self.alpha = self.log_alpha.exp()
        else:
            self.alpha = torch.tensor(0.2, device=device)
    
    def select_action(
        self,
        state: np.ndarray,
        eval_mode: bool = False,
    ) -> np.ndarray:
        """Select action"""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action, _ = self.actor(state, deterministic=eval_mode, return_log_prob=False)
        
        return action.cpu().numpy()[0]
    
    def update(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """
        Update actor and critics.
        
        Args:
            batch: Dict with 'states', 'actions', 'rewards', 'next_states', 'dones'
            
        Returns:
            Dict with loss values
        """
        states = batch['states']
        actions = batch['actions']
        rewards = batch['rewards']
        next_states = batch['next_states']
        dones = batch['dones']
        
        # Update critics
        with torch.no_grad():
            # Sample next actions
            next_actions, next_log_probs = self.actor(next_states)
            
            # Compute target Q-values
            target_q1 = self.critic1_target(next_states, next_actions)
            target_q2 = self.critic2_target(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)
            
            # Entropy term
            target_q = target_q - self.alpha * next_log_probs
            
            # TD target
            target_q = rewards + (1 - dones) * self.gamma * target_q
        
        # Critic 1 loss
        current_q1 = self.critic1(states, actions)
        critic1_loss = F.mse_loss(current_q1, target_q)
        
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()
        
        # Critic 2 loss
        current_q2 = self.critic2(states, actions)
        critic2_loss = F.mse_loss(current_q2, target_q)
        
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()
        
        # Update actor
        new_actions, log_probs = self.actor(states)
        
        q1 = self.critic1(states, new_actions)
        q2 = self.critic2(states, new_actions)
        q = torch.min(q1, q2)
        
        actor_loss = (self.alpha * log_probs - q).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Update temperature
        alpha_loss = None
        if self.auto_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            
            self.alpha = self.log_alpha.exp()
        
        # Soft update target networks
        self._soft_update(self.critic1, self.critic1_target)
        self._soft_update(self.critic2, self.critic2_target)
        
        return {
            'critic1_loss': critic1_loss.item(),
            'critic2_loss': critic2_loss.item(),
            'actor_loss': actor_loss.item(),
            'alpha': self.alpha.item(),
            'alpha_loss': alpha_loss.item() if alpha_loss is not None else 0.0,
        }
    
    def _soft_update(self, source: nn.Module, target: nn.Module):
        """Soft update target network"""
        for param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )
    
    def save(self, path: str):
        """Save agent"""
        checkpoint = {
            'actor': self.actor.state_dict(),
            'critic1': self.critic1.state_dict(),
            'critic2': self.critic2.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic1_optimizer': self.critic1_optimizer.state_dict(),
            'critic2_optimizer': self.critic2_optimizer.state_dict(),
        }
        
        if self.auto_entropy_tuning:
            checkpoint['log_alpha'] = self.log_alpha
            checkpoint['alpha_optimizer'] = self.alpha_optimizer.state_dict()
        
        torch.save(checkpoint, path)
    
    def load(self, path: str):
        """Load agent"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic1.load_state_dict(checkpoint['critic1'])
        self.critic2.load_state_dict(checkpoint['critic2'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic1_optimizer.load_state_dict(checkpoint['critic1_optimizer'])
        self.critic2_optimizer.load_state_dict(checkpoint['critic2_optimizer'])
        
        if self.auto_entropy_tuning and 'log_alpha' in checkpoint:
            self.log_alpha = checkpoint['log_alpha']
            self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer'])
            self.alpha = self.log_alpha.exp()
        
        # Update target networks
        self.critic1_target = copy.deepcopy(self.critic1)
        self.critic2_target = copy.deepcopy(self.critic2)
