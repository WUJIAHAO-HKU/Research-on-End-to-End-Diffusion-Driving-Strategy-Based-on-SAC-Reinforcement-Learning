"""
SAC Agent with Diffusion Policy as Actor

This module implements the Soft Actor-Critic (SAC) algorithm where
the actor network is a diffusion policy instead of a Gaussian policy.

Key Innovation: Maximum entropy RL with diffusion-based action sampling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import copy


class SACAgent(nn.Module):
    """
    SAC Agent with Diffusion Policy Actor and Twin Q-networks.
    
    Components:
    - Actor: Diffusion Policy (generates multi-modal actions)
    - Critic: Twin Q-networks (value estimation)
    - Alpha: Entropy temperature (auto-tuned)
    """
    
    def __init__(
        self,
        diffusion_policy: nn.Module,
        critic_network: nn.Module,
        observation_encoder_critic: nn.Module,
        action_dim: int = 3,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        auto_entropy_tuning: bool = True,
        target_entropy: Optional[float] = None,
        device: str = "cuda",
    ):
        super().__init__()
        
        self.device = torch.device(device)
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        
        # Actor (Diffusion Policy)
        self.actor = diffusion_policy.to(self.device)
        
        # Critics (Twin Q-networks)
        self.critic1 = critic_network.to(self.device)
        self.critic2 = copy.deepcopy(critic_network).to(self.device)
        
        # Target critics
        self.critic1_target = copy.deepcopy(self.critic1).to(self.device)
        self.critic2_target = copy.deepcopy(self.critic2).to(self.device)
        
        # Freeze target networks
        for param in self.critic1_target.parameters():
            param.requires_grad = False
        for param in self.critic2_target.parameters():
            param.requires_grad = False
        
        # Entropy temperature
        self.auto_entropy_tuning = auto_entropy_tuning
        if auto_entropy_tuning:
            if target_entropy is None:
                # Heuristic: -action_dim
                target_entropy = -action_dim
            self.target_entropy = target_entropy
            self.log_alpha = torch.tensor([0.0], requires_grad=True, device=self.device)
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=3e-4)
        else:
            self.alpha = alpha
    
    @property
    def alpha(self) -> float:
        """Get current entropy temperature"""
        if self.auto_entropy_tuning:
            return self.log_alpha.exp().item()
        else:
            return self._alpha
    
    @alpha.setter
    def alpha(self, value: float):
        """Set entropy temperature (only when not auto-tuning)"""
        if not self.auto_entropy_tuning:
            self._alpha = value
    
    def select_action(
        self,
        observations: Dict[str, torch.Tensor],
        deterministic: bool = False,
    ) -> torch.Tensor:
        """
        Select action using the diffusion policy.
        
        Args:
            observations: Multi-modal observations
            deterministic: If True, use mean action (no sampling)
            
        Returns:
            action: (B, action_dim) action to execute (first step of predicted horizon)
        """
        with torch.no_grad():
            # Sample action sequence from diffusion policy
            action_sequence = self.actor.sample_actions(
                self.actor.obs_encoder(observations)
            )  # (B, horizon, action_dim)
            
            # Use only the first action
            action = action_sequence[:, 0, :]  # (B, action_dim)
            
            # Clip to valid range
            action = torch.clamp(action, -1.0, 1.0)
        
        return action
    
    def update(
        self,
        observations: Dict[str, torch.Tensor],
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: Dict[str, torch.Tensor],
        dones: torch.Tensor,
        update_actor: bool = True,
    ) -> Dict[str, float]:
        """
        Update SAC agent (critic, actor, alpha).
        
        Args:
            observations: Current observations
            actions: Actions taken (first action from horizon)
            rewards: Rewards received
            next_observations: Next observations
            dones: Episode termination flags
            update_actor: Whether to update actor (for delayed updates)
            
        Returns:
            Dictionary of training metrics
        """
        batch_size = rewards.shape[0]
        
        # ============================================
        # Update Critic
        # ============================================
        
        # Compute target Q value
        with torch.no_grad():
            # Sample next actions from diffusion policy
            next_action_sequence = self.actor.sample_actions(
                self.actor.obs_encoder(next_observations)
            )
            next_actions = next_action_sequence[:, 0, :]  # (B, action_dim)
            
            # Compute target Q values
            next_q1 = self.critic1_target(next_observations, next_actions)
            next_q2 = self.critic2_target(next_observations, next_actions)
            next_q = torch.min(next_q1, next_q2)  # Clipped double Q-learning
            
            # Add entropy bonus (maximum entropy RL)
            # For diffusion policy, we approximate entropy
            action_log_prob = self.actor.compute_log_prob(next_observations, next_action_sequence)
            next_q = next_q - self.alpha * action_log_prob
            
            # Compute target
            target_q = rewards + (1 - dones.float()) * self.gamma * next_q
        
        # Get current Q estimates
        current_q1 = self.critic1(observations, actions)
        current_q2 = self.critic2(observations, actions)
        
        # Compute critic losses
        critic1_loss = F.mse_loss(current_q1, target_q)
        critic2_loss = F.mse_loss(current_q2, target_q)
        
        # Update critics
        self.critic1.optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1.optimizer.step()
        
        self.critic2.optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2.optimizer.step()
        
        # ============================================
        # Update Actor (Diffusion Policy)
        # ============================================
        
        actor_loss = torch.tensor(0.0)
        if update_actor:
            # Sample actions from diffusion policy
            action_sequence = self.actor.sample_actions(
                self.actor.obs_encoder(observations)
            )
            sampled_actions = action_sequence[:, 0, :]  # (B, action_dim)
            
            # Compute Q value for sampled actions
            q1_pi = self.critic1(observations, sampled_actions)
            q2_pi = self.critic2(observations, sampled_actions)
            q_pi = torch.min(q1_pi, q2_pi)
            
            # Compute log probability
            action_log_prob = self.actor.compute_log_prob(observations, action_sequence)
            
            # Actor loss: maximize Q - alpha * log_prob (equivalent to maximizing Q + entropy)
            actor_loss = (self.alpha * action_log_prob - q_pi).mean()
            
            # Update actor
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            self.actor.optimizer.step()
            
            # ============================================
            # Update Alpha (Entropy Temperature)
            # ============================================
            
            if self.auto_entropy_tuning:
                alpha_loss = -(self.log_alpha * (action_log_prob + self.target_entropy).detach()).mean()
                
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()
        
        # ============================================
        # Soft Update Target Networks
        # ============================================
        
        self._soft_update(self.critic1, self.critic1_target)
        self._soft_update(self.critic2, self.critic2_target)
        
        # Return metrics
        metrics = {
            'critic1_loss': critic1_loss.item(),
            'critic2_loss': critic2_loss.item(),
            'actor_loss': actor_loss.item() if update_actor else 0.0,
            'q1_value': current_q1.mean().item(),
            'q2_value': current_q2.mean().item(),
            'alpha': self.alpha,
            'action_log_prob': action_log_prob.mean().item() if update_actor else 0.0,
        }
        
        return metrics
    
    def _soft_update(self, source: nn.Module, target: nn.Module):
        """Soft update target network parameters"""
        for param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )
    
    def save(self, filepath: str):
        """Save agent state"""
        state = {
            'actor': self.actor.state_dict(),
            'critic1': self.critic1.state_dict(),
            'critic2': self.critic2.state_dict(),
            'critic1_target': self.critic1_target.state_dict(),
            'critic2_target': self.critic2_target.state_dict(),
        }
        if self.auto_entropy_tuning:
            state['log_alpha'] = self.log_alpha
        torch.save(state, filepath)
        print(f"[SAC] Saved agent to {filepath}")
    
    def load(self, filepath: str):
        """Load agent state"""
        state = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(state['actor'])
        self.critic1.load_state_dict(state['critic1'])
        self.critic2.load_state_dict(state['critic2'])
        self.critic1_target.load_state_dict(state['critic1_target'])
        self.critic2_target.load_state_dict(state['critic2_target'])
        if self.auto_entropy_tuning and 'log_alpha' in state:
            self.log_alpha = state['log_alpha']
        print(f"[SAC] Loaded agent from {filepath}")


class QNetwork(nn.Module):
    """
    Q-Network for SAC (maps observation and action to Q-value).
    """
    
    def __init__(
        self,
        observation_encoder: nn.Module,
        action_dim: int,
        hidden_dims: Tuple[int, ...] = (512, 512, 256),
    ):
        super().__init__()
        
        self.obs_encoder = observation_encoder
        obs_dim = observation_encoder.output_dim  # Assume encoder has output_dim attribute
        
        # Q-network layers
        layers = []
        input_dim = obs_dim + action_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
            ])
            input_dim = hidden_dim
        
        layers.append(nn.Linear(input_dim, 1))  # Output Q-value
        
        self.q_net = nn.Sequential(*layers)
        
        # Optimizer (will be set externally)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=3e-4)
    
    def forward(
        self,
        observations: Dict[str, torch.Tensor],
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Q(s, a).
        
        Args:
            observations: Multi-modal observations
            actions: (B, action_dim) actions
            
        Returns:
            q_value: (B, 1) Q-values
        """
        # Encode observations
        obs_embedding = self.obs_encoder(observations)  # (B, obs_dim)
        
        # Concatenate with actions
        q_input = torch.cat([obs_embedding, actions], dim=1)  # (B, obs_dim + action_dim)
        
        # Compute Q-value
        q_value = self.q_net(q_input)  # (B, 1)
        
        return q_value.squeeze(-1)  # (B,)
