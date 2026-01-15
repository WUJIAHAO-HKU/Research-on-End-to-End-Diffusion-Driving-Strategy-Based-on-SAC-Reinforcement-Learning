"""
Replay Buffer for Off-Policy RL

Supports:
- Efficient storage with zarr/numpy
- Prioritized experience replay
- N-step returns
- Hindsight experience replay (HER)
"""

import numpy as np
import torch
import zarr
from typing import Dict, Tuple, Optional, List
from pathlib import Path
import pickle


class ReplayBuffer:
    """
    Standard uniform replay buffer.
    
    Stores transitions and samples uniformly.
    """
    
    def __init__(
        self,
        capacity: int = 1_000_000,
        observation_shape: Dict[str, Tuple] = None,
        action_dim: int = 3,
        device: str = "cpu",
        save_dir: Optional[Path] = None,
    ):
        self.capacity = capacity
        self.device = device
        self.save_dir = Path(save_dir) if save_dir else None
        self.ptr = 0
        self.size = 0
        
        # Initialize storage
        if observation_shape is None:
            observation_shape = {
                'vision': (4, 240, 320),
                'lidar': (360, 3),
                'proprio': (8,),
            }
        
        self.observation_shape = observation_shape
        self.action_dim = action_dim
        
        # Use zarr for efficient on-disk storage (optional)
        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            store = zarr.DirectoryStore(str(self.save_dir / 'buffer.zarr'))
            self.root = zarr.group(store=store, overwrite=True)
            
            # Create datasets
            for key, shape in observation_shape.items():
                self.root.create_dataset(
                    f'observations_{key}',
                    shape=(capacity,) + shape,
                    chunks=(1000,) + shape,
                    dtype='float32',
                )
                self.root.create_dataset(
                    f'next_observations_{key}',
                    shape=(capacity,) + shape,
                    chunks=(1000,) + shape,
                    dtype='float32',
                )
            
            self.root.create_dataset('actions', shape=(capacity, action_dim), chunks=(1000, action_dim), dtype='float32')
            self.root.create_dataset('rewards', shape=(capacity,), chunks=(1000,), dtype='float32')
            self.root.create_dataset('dones', shape=(capacity,), chunks=(1000,), dtype='bool')
        else:
            # In-memory storage
            self.observations = {
                key: np.zeros((capacity,) + shape, dtype=np.float32)
                for key, shape in observation_shape.items()
            }
            self.next_observations = {
                key: np.zeros((capacity,) + shape, dtype=np.float32)
                for key, shape in observation_shape.items()
            }
            self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
            self.rewards = np.zeros(capacity, dtype=np.float32)
            self.dones = np.zeros(capacity, dtype=bool)
    
    def add(
        self,
        observation: Dict[str, np.ndarray],
        action: np.ndarray,
        reward: float,
        next_observation: Dict[str, np.ndarray],
        done: bool,
    ):
        """Add a transition to the buffer"""
        if self.save_dir:
            # Save to zarr
            for key in self.observation_shape.keys():
                self.root[f'observations_{key}'][self.ptr] = observation[key]
                self.root[f'next_observations_{key}'][self.ptr] = next_observation[key]
            self.root['actions'][self.ptr] = action
            self.root['rewards'][self.ptr] = reward
            self.root['dones'][self.ptr] = done
        else:
            # Save to memory
            for key in self.observation_shape.keys():
                self.observations[key][self.ptr] = observation[key]
                self.next_observations[key][self.ptr] = next_observation[key]
            self.actions[self.ptr] = action
            self.rewards[self.ptr] = reward
            self.dones[self.ptr] = done
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample a batch of transitions uniformly"""
        indices = np.random.randint(0, self.size, size=batch_size)
        
        if self.save_dir:
            batch = {
                'observations': {
                    key: torch.from_numpy(self.root[f'observations_{key}'][indices]).to(self.device)
                    for key in self.observation_shape.keys()
                },
                'actions': torch.from_numpy(self.root['actions'][indices]).to(self.device),
                'rewards': torch.from_numpy(self.root['rewards'][indices]).to(self.device),
                'next_observations': {
                    key: torch.from_numpy(self.root[f'next_observations_{key}'][indices]).to(self.device)
                    for key in self.observation_shape.keys()
                },
                'dones': torch.from_numpy(self.root['dones'][indices]).to(self.device),
            }
        else:
            batch = {
                'observations': {
                    key: torch.from_numpy(self.observations[key][indices]).to(self.device)
                    for key in self.observation_shape.keys()
                },
                'actions': torch.from_numpy(self.actions[indices]).to(self.device),
                'rewards': torch.from_numpy(self.rewards[indices]).to(self.device),
                'next_observations': {
                    key: torch.from_numpy(self.next_observations[key][indices]).to(self.device)
                    for key in self.observation_shape.keys()
                },
                'dones': torch.from_numpy(self.dones[indices]).to(self.device),
            }
        
        return batch
    
    def __len__(self):
        return self.size
    
    def save(self, path: Path):
        """Save buffer to disk"""
        if not self.save_dir:
            # Save in-memory buffer
            path = Path(path)
            path.mkdir(parents=True, exist_ok=True)
            
            np.save(path / 'observations.npy', {k: v[:self.size] for k, v in self.observations.items()})
            np.save(path / 'next_observations.npy', {k: v[:self.size] for k, v in self.next_observations.items()})
            np.save(path / 'actions.npy', self.actions[:self.size])
            np.save(path / 'rewards.npy', self.rewards[:self.size])
            np.save(path / 'dones.npy', self.dones[:self.size])
            
            with open(path / 'metadata.pkl', 'wb') as f:
                pickle.dump({
                    'capacity': self.capacity,
                    'size': self.size,
                    'ptr': self.ptr,
                    'observation_shape': self.observation_shape,
                    'action_dim': self.action_dim,
                }, f)
    
    def load(self, path: Path):
        """Load buffer from disk"""
        path = Path(path)
        
        with open(path / 'metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)
        
        self.capacity = metadata['capacity']
        self.size = metadata['size']
        self.ptr = metadata['ptr']
        
        observations = np.load(path / 'observations.npy', allow_pickle=True).item()
        next_observations = np.load(path / 'next_observations.npy', allow_pickle=True).item()
        
        for key in self.observation_shape.keys():
            self.observations[key][:self.size] = observations[key]
            self.next_observations[key][:self.size] = next_observations[key]
        
        self.actions[:self.size] = np.load(path / 'actions.npy')
        self.rewards[:self.size] = np.load(path / 'rewards.npy')
        self.dones[:self.size] = np.load(path / 'dones.npy')


class PrioritizedReplayBuffer(ReplayBuffer):
    """
    Prioritized Experience Replay (PER).
    
    Samples transitions proportional to their TD error.
    Reference: Schaul et al., ICLR 2016
    """
    
    def __init__(
        self,
        capacity: int = 1_000_000,
        observation_shape: Dict[str, Tuple] = None,
        action_dim: int = 3,
        device: str = "cpu",
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_annealing_steps: int = 1_000_000,
        epsilon: float = 1e-6,
        **kwargs
    ):
        super().__init__(capacity, observation_shape, action_dim, device, **kwargs)
        
        self.alpha = alpha  # Priority exponent
        self.beta = beta    # Importance sampling exponent
        self.beta_max = 1.0
        self.beta_annealing_steps = beta_annealing_steps
        self.epsilon = epsilon
        self.max_priority = 1.0
        
        # Sum tree for efficient sampling
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.sample_count = 0
    
    def add(
        self,
        observation: Dict[str, np.ndarray],
        action: np.ndarray,
        reward: float,
        next_observation: Dict[str, np.ndarray],
        done: bool,
    ):
        """Add transition with maximum priority"""
        super().add(observation, action, reward, next_observation, done)
        
        # Assign maximum priority to new sample
        self.priorities[self.ptr - 1] = self.max_priority
    
    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample batch according to priorities"""
        # Compute sampling probabilities
        priorities = self.priorities[:self.size]
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        # Sample indices
        indices = np.random.choice(self.size, size=batch_size, p=probs, replace=False)
        
        # Compute importance sampling weights
        self.sample_count += 1
        beta = min(self.beta_max, self.beta + (self.beta_max - self.beta) * 
                   self.sample_count / self.beta_annealing_steps)
        
        weights = (self.size * probs[indices]) ** (-beta)
        weights /= weights.max()
        
        # Get batch
        batch = super().sample.__wrapped__(self, indices)
        batch['weights'] = torch.from_numpy(weights).float().to(self.device)
        batch['indices'] = indices
        
        return batch
    
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """Update priorities after learning"""
        priorities = np.abs(priorities) + self.epsilon
        self.priorities[indices] = priorities
        self.max_priority = max(self.max_priority, priorities.max())


class NStepReplayBuffer(ReplayBuffer):
    """
    N-step replay buffer for multi-step returns.
    
    Stores n-step transitions: (s_t, a_t, R_t^n, s_{t+n})
    where R_t^n = sum_{i=0}^{n-1} gamma^i * r_{t+i}
    """
    
    def __init__(
        self,
        capacity: int = 1_000_000,
        observation_shape: Dict[str, Tuple] = None,
        action_dim: int = 3,
        device: str = "cpu",
        n_step: int = 3,
        gamma: float = 0.99,
        **kwargs
    ):
        super().__init__(capacity, observation_shape, action_dim, device, **kwargs)
        
        self.n_step = n_step
        self.gamma = gamma
        self.n_step_buffer = []
    
    def add(
        self,
        observation: Dict[str, np.ndarray],
        action: np.ndarray,
        reward: float,
        next_observation: Dict[str, np.ndarray],
        done: bool,
    ):
        """Add transition to n-step buffer"""
        self.n_step_buffer.append((observation, action, reward, next_observation, done))
        
        # Compute n-step return when buffer is full
        if len(self.n_step_buffer) == self.n_step:
            # Compute n-step return
            n_step_return = 0
            for i, (_, _, r, _, d) in enumerate(self.n_step_buffer):
                n_step_return += (self.gamma ** i) * r
                if d:
                    break
            
            # Get first observation and last next_observation
            first_obs, first_action = self.n_step_buffer[0][:2]
            last_next_obs, last_done = self.n_step_buffer[-1][3:]
            
            # Add to main buffer
            super().add(first_obs, first_action, n_step_return, last_next_obs, last_done)
            
            # Remove oldest transition
            self.n_step_buffer.pop(0)


class HERReplayBuffer(ReplayBuffer):
    """
    Hindsight Experience Replay (HER).
    
    Re-labels failed experiences with achieved goals.
    Reference: Andrychowicz et al., NeurIPS 2017
    """
    
    def __init__(
        self,
        capacity: int = 1_000_000,
        observation_shape: Dict[str, Tuple] = None,
        action_dim: int = 3,
        device: str = "cpu",
        her_ratio: float = 0.8,
        goal_selection_strategy: str = "future",
        **kwargs
    ):
        super().__init__(capacity, observation_shape, action_dim, device, **kwargs)
        
        self.her_ratio = her_ratio
        self.goal_selection_strategy = goal_selection_strategy
        self.episode_buffer = []
    
    def add_episode(self, episode: List[Tuple]):
        """
        Add full episode and apply HER.
        
        Args:
            episode: List of (obs, action, reward, next_obs, done, achieved_goal, desired_goal)
        """
        # Add original transitions
        for transition in episode:
            obs, action, reward, next_obs, done, achieved_goal, desired_goal = transition
            super().add(obs, action, reward, next_obs, done)
        
        # Apply HER
        if np.random.random() < self.her_ratio:
            # Sample new goals from episode
            for i, transition in enumerate(episode):
                obs, action, _, next_obs, done, achieved_goal, _ = transition
                
                # Select future achieved goal as new desired goal
                if self.goal_selection_strategy == "future":
                    future_idx = np.random.randint(i, len(episode))
                    new_goal = episode[future_idx][5]  # achieved_goal
                elif self.goal_selection_strategy == "final":
                    new_goal = episode[-1][5]
                elif self.goal_selection_strategy == "random":
                    random_idx = np.random.randint(0, len(episode))
                    new_goal = episode[random_idx][5]
                else:
                    raise ValueError(f"Unknown strategy: {self.goal_selection_strategy}")
                
                # Compute new reward (binary sparse reward)
                new_reward = 1.0 if np.allclose(achieved_goal, new_goal, atol=0.1) else 0.0
                
                # Add re-labeled transition
                super().add(obs, action, new_reward, next_obs, done)


def create_replay_buffer(config: dict) -> ReplayBuffer:
    """Factory function to create replay buffer from config"""
    buffer_type = config.get('type', 'uniform')
    
    if buffer_type == 'uniform':
        return ReplayBuffer(
            capacity=config.get('capacity', 1_000_000),
            observation_shape=config.get('observation_shape'),
            action_dim=config.get('action_dim', 3),
            device=config.get('device', 'cpu'),
            save_dir=config.get('save_dir'),
        )
    elif buffer_type == 'prioritized':
        return PrioritizedReplayBuffer(
            capacity=config.get('capacity', 1_000_000),
            observation_shape=config.get('observation_shape'),
            action_dim=config.get('action_dim', 3),
            device=config.get('device', 'cpu'),
            alpha=config.get('alpha', 0.6),
            beta=config.get('beta', 0.4),
        )
    elif buffer_type == 'nstep':
        return NStepReplayBuffer(
            capacity=config.get('capacity', 1_000_000),
            observation_shape=config.get('observation_shape'),
            action_dim=config.get('action_dim', 3),
            device=config.get('device', 'cpu'),
            n_step=config.get('n_step', 3),
            gamma=config.get('gamma', 0.99),
        )
    elif buffer_type == 'her':
        return HERReplayBuffer(
            capacity=config.get('capacity', 1_000_000),
            observation_shape=config.get('observation_shape'),
            action_dim=config.get('action_dim', 3),
            device=config.get('device', 'cpu'),
            her_ratio=config.get('her_ratio', 0.8),
        )
    else:
        raise ValueError(f"Unknown buffer type: {buffer_type}")
