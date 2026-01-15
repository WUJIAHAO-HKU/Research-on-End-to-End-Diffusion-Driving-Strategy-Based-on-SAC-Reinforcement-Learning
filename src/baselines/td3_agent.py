"""
Twin Delayed Deep Deterministic Policy Gradient (TD3)

Baseline algorithm for comparison with SAC-Diffusion.
Reference: Fujimoto et al., ICML 2018
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional
import copy


class Actor(nn.Module):
    """Deterministic policy network"""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: Tuple[int, ...] = (256, 256),
        max_action: float = 1.0,
    ):
        super().__init__()
        
        layers = []
        in_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
            ])
            in_dim = hidden_dim
        
        layers.append(nn.Linear(in_dim, action_dim))
        layers.append(nn.Tanh())
        
        self.network = nn.Sequential(*layers)
        self.max_action = max_action
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.max_action * self.network(state)


class Critic(nn.Module):
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


class TD3Agent:
    """
    TD3 agent with:
    - Twin Q-networks
    - Delayed policy updates
    - Target policy smoothing
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_action: float = 1.0,
        hidden_dims: Tuple[int, ...] = (256, 256),
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
        policy_delay: int = 2,
        device: str = "cuda",
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        self.device = device
        
        # Actor
        self.actor = Actor(state_dim, action_dim, hidden_dims, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        
        # Twin Critics
        self.critic1 = Critic(state_dim, action_dim, hidden_dims).to(device)
        self.critic1_target = copy.deepcopy(self.critic1)
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=critic_lr)
        
        self.critic2 = Critic(state_dim, action_dim, hidden_dims).to(device)
        self.critic2_target = copy.deepcopy(self.critic2)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=critic_lr)
        
        self.total_updates = 0
    
    def select_action(
        self,
        state: np.ndarray,
        noise: float = 0.1,
        eval_mode: bool = False,
    ) -> np.ndarray:
        """Select action with optional exploration noise"""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action = self.actor(state).cpu().numpy()[0]
        
        if not eval_mode:
            noise_sample = np.random.normal(0, noise, size=self.action_dim)
            action = action + noise_sample
            action = np.clip(action, -self.max_action, self.max_action)
        
        return action
    
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
            # Target policy smoothing
            noise = torch.randn_like(actions) * self.policy_noise
            noise = noise.clamp(-self.noise_clip, self.noise_clip)
            
            next_actions = self.actor_target(next_states) + noise
            next_actions = next_actions.clamp(-self.max_action, self.max_action)
            
            # Compute target Q-values
            target_q1 = self.critic1_target(next_states, next_actions)
            target_q2 = self.critic2_target(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)
            
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
        
        # Delayed policy update
        actor_loss = None
        if self.total_updates % self.policy_delay == 0:
            # Actor loss
            actor_loss = -self.critic1(states, self.actor(states)).mean()
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # Soft update target networks
            self._soft_update(self.actor, self.actor_target)
            self._soft_update(self.critic1, self.critic1_target)
            self._soft_update(self.critic2, self.critic2_target)
        
        self.total_updates += 1
        
        return {
            'critic1_loss': critic1_loss.item(),
            'critic2_loss': critic2_loss.item(),
            'actor_loss': actor_loss.item() if actor_loss is not None else 0.0,
        }
    
    def _soft_update(self, source: nn.Module, target: nn.Module):
        """Soft update target network"""
        for param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )
    
    def save(self, path: str):
        """Save agent"""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic1': self.critic1.state_dict(),
            'critic2': self.critic2.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic1_optimizer': self.critic1_optimizer.state_dict(),
            'critic2_optimizer': self.critic2_optimizer.state_dict(),
        }, path)
    
    def load(self, path: str):
        """Load agent"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic1.load_state_dict(checkpoint['critic1'])
        self.critic2.load_state_dict(checkpoint['critic2'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic1_optimizer.load_state_dict(checkpoint['critic1_optimizer'])
        self.critic2_optimizer.load_state_dict(checkpoint['critic2_optimizer'])
        
        # Update target networks
        self.actor_target = copy.deepcopy(self.actor)
        self.critic1_target = copy.deepcopy(self.critic1)
        self.critic2_target = copy.deepcopy(self.critic2)
