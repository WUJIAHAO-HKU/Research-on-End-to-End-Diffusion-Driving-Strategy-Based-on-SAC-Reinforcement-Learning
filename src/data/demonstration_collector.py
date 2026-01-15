"""
Demonstration Collector

Collects expert demonstrations using:
- MPC controller
- Human teleoperation
- Scripted policies
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Callable
import h5py
import pickle
from tqdm import tqdm
import cv2


class DemonstrationCollector:
    """
    Collects and saves expert demonstrations.
    
    Stores trajectories in HDF5 or pickle format.
    """
    
    def __init__(
        self,
        env,
        save_dir: Path,
        max_episodes: int = 1000,
        max_steps_per_episode: int = 1000,
        save_format: str = 'hdf5',
        save_video: bool = False,
        video_fps: int = 30,
    ):
        self.env = env
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_episodes = max_episodes
        self.max_steps_per_episode = max_steps_per_episode
        self.save_format = save_format
        self.save_video = save_video
        self.video_fps = video_fps
        
        self.trajectories = []
        self.episode_count = 0
    
    def collect_with_policy(
        self,
        policy: Callable,
        num_episodes: int = 100,
        success_only: bool = True,
        min_return: Optional[float] = None,
    ):
        """
        Collect demonstrations using a policy.
        
        Args:
            policy: Policy function (obs -> action)
            num_episodes: Number of episodes to collect
            success_only: Only save successful episodes
            min_return: Minimum episode return to save
        """
        print(f"Collecting {num_episodes} demonstrations...")
        
        success_count = 0
        pbar = tqdm(total=num_episodes)
        
        while success_count < num_episodes and self.episode_count < self.max_episodes:
            trajectory = self._collect_episode(policy)
            
            # Check success criteria
            episode_return = sum(trajectory['rewards'])
            is_success = trajectory.get('success', False)
            
            save_episode = True
            if success_only and not is_success:
                save_episode = False
            if min_return is not None and episode_return < min_return:
                save_episode = False
            
            if save_episode:
                self.trajectories.append(trajectory)
                success_count += 1
                pbar.update(1)
            
            self.episode_count += 1
        
        pbar.close()
        print(f"Collected {success_count} successful demonstrations "
              f"(total attempts: {self.episode_count})")
    
    def _collect_episode(self, policy: Callable) -> Dict:
        """Collect a single episode"""
        observations = {key: [] for key in ['vision', 'lidar', 'proprio']}
        actions = []
        rewards = []
        dones = []
        infos = []
        
        obs = self.env.reset()
        done = False
        step_count = 0
        
        # Video recording
        if self.save_video:
            video_frames = []
        
        while not done and step_count < self.max_steps_per_episode:
            # Get action from policy
            action = policy(obs)
            
            # Step environment
            next_obs, reward, done, info = self.env.step(action)
            
            # Store transition
            for key in observations:
                if key in obs:
                    observations[key].append(obs[key])
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)
            
            # Record video frame
            if self.save_video and 'vision' in obs:
                frame = obs['vision'][:3]  # RGB only
                frame = (frame.transpose(1, 2, 0) * 255).astype(np.uint8)
                video_frames.append(frame)
            
            obs = next_obs
            step_count += 1
        
        # Convert to numpy arrays
        trajectory = {
            'observations': {
                key: np.array(observations[key])
                for key in observations if len(observations[key]) > 0
            },
            'actions': np.array(actions),
            'rewards': np.array(rewards),
            'dones': np.array(dones),
            'infos': infos,
            'length': step_count,
            'return': sum(rewards),
            'success': infos[-1].get('success', False) if infos else False,
        }
        
        # Save video
        if self.save_video and video_frames:
            video_path = self.save_dir / f'episode_{self.episode_count}.mp4'
            self._save_video(video_frames, video_path)
        
        return trajectory
    
    def _save_video(self, frames: List[np.ndarray], path: Path):
        """Save video frames to mp4 file"""
        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(
            str(path), fourcc, self.video_fps, (width, height)
        )
        
        for frame in frames:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            video_writer.write(frame_bgr)
        
        video_writer.release()
    
    def save(self, filename: Optional[str] = None):
        """
        Save collected demonstrations to disk.
        
        Args:
            filename: Optional filename (default: demonstrations.hdf5 or .pkl)
        """
        if filename is None:
            filename = f'demonstrations.{self.save_format}'
        
        save_path = self.save_dir / filename
        
        if self.save_format == 'hdf5':
            self._save_hdf5(save_path)
        elif self.save_format == 'pkl':
            self._save_pickle(save_path)
        else:
            raise ValueError(f"Unknown save format: {self.save_format}")
        
        print(f"Saved {len(self.trajectories)} trajectories to {save_path}")
    
    def _save_hdf5(self, path: Path):
        """Save trajectories to HDF5 file"""
        with h5py.File(path, 'w') as f:
            for traj_idx, traj in enumerate(self.trajectories):
                traj_group = f.create_group(f'traj_{traj_idx}')
                
                # Save observations
                obs_group = traj_group.create_group('observations')
                for key, value in traj['observations'].items():
                    obs_group.create_dataset(key, data=value, compression='gzip')
                
                # Save actions, rewards, dones
                traj_group.create_dataset('actions', data=traj['actions'], compression='gzip')
                traj_group.create_dataset('rewards', data=traj['rewards'], compression='gzip')
                traj_group.create_dataset('dones', data=traj['dones'], compression='gzip')
                
                # Save metadata
                traj_group.attrs['length'] = traj['length']
                traj_group.attrs['return'] = traj['return']
                traj_group.attrs['success'] = traj['success']
    
    def _save_pickle(self, path: Path):
        """Save trajectories to pickle file"""
        with open(path, 'wb') as f:
            pickle.dump(self.trajectories, f)
    
    def load(self, path: Path):
        """Load trajectories from disk"""
        path = Path(path)
        
        if path.suffix == '.hdf5' or path.suffix == '.h5':
            self._load_hdf5(path)
        elif path.suffix == '.pkl':
            self._load_pickle(path)
        else:
            raise ValueError(f"Unknown file format: {path.suffix}")
        
        print(f"Loaded {len(self.trajectories)} trajectories from {path}")
    
    def _load_hdf5(self, path: Path):
        """Load trajectories from HDF5 file"""
        self.trajectories = []
        
        with h5py.File(path, 'r') as f:
            num_trajs = len(f.keys())
            
            for traj_idx in range(num_trajs):
                traj_group = f[f'traj_{traj_idx}']
                
                traj = {
                    'observations': {},
                    'actions': traj_group['actions'][:],
                    'rewards': traj_group['rewards'][:],
                    'dones': traj_group['dones'][:],
                    'length': traj_group.attrs['length'],
                    'return': traj_group.attrs['return'],
                    'success': traj_group.attrs['success'],
                }
                
                # Load observations
                obs_group = traj_group['observations']
                for key in obs_group.keys():
                    traj['observations'][key] = obs_group[key][:]
                
                self.trajectories.append(traj)
    
    def _load_pickle(self, path: Path):
        """Load trajectories from pickle file"""
        with open(path, 'rb') as f:
            self.trajectories = pickle.load(f)
    
    def get_statistics(self) -> Dict:
        """Get statistics about collected demonstrations"""
        if not self.trajectories:
            return {}
        
        lengths = [traj['length'] for traj in self.trajectories]
        returns = [traj['return'] for traj in self.trajectories]
        success_rate = np.mean([traj['success'] for traj in self.trajectories])
        
        return {
            'num_trajectories': len(self.trajectories),
            'avg_length': np.mean(lengths),
            'std_length': np.std(lengths),
            'min_length': np.min(lengths),
            'max_length': np.max(lengths),
            'avg_return': np.mean(returns),
            'std_return': np.std(returns),
            'min_return': np.min(returns),
            'max_return': np.max(returns),
            'success_rate': success_rate,
            'total_steps': sum(lengths),
        }


class MPCDemonstrationCollector(DemonstrationCollector):
    """
    Specialized collector using MPC as expert policy.
    """
    
    def __init__(
        self,
        env,
        save_dir: Path,
        mpc_controller,
        **kwargs
    ):
        super().__init__(env, save_dir, **kwargs)
        self.mpc_controller = mpc_controller
    
    def collect(self, num_episodes: int = 100):
        """Collect demonstrations using MPC"""
        policy = lambda obs: self.mpc_controller.get_action(obs)
        self.collect_with_policy(
            policy=policy,
            num_episodes=num_episodes,
            success_only=True,
        )


class HumanDemonstrationCollector(DemonstrationCollector):
    """
    Collector for human teleoperation demonstrations.
    
    Uses keyboard or joystick input.
    """
    
    def __init__(
        self,
        env,
        save_dir: Path,
        input_type: str = 'keyboard',
        **kwargs
    ):
        super().__init__(env, save_dir, **kwargs)
        self.input_type = input_type
        
        if input_type == 'joystick':
            try:
                import pygame
                pygame.init()
                pygame.joystick.init()
                self.joystick = pygame.joystick.Joystick(0)
                self.joystick.init()
                print(f"Joystick initialized: {self.joystick.get_name()}")
            except:
                print("Joystick not available, falling back to keyboard")
                self.input_type = 'keyboard'
    
    def get_human_action(self) -> np.ndarray:
        """Get action from human input"""
        if self.input_type == 'keyboard':
            return self._get_keyboard_action()
        elif self.input_type == 'joystick':
            return self._get_joystick_action()
    
    def _get_keyboard_action(self) -> np.ndarray:
        """Get action from keyboard (WASD + QE for rotation)"""
        import pygame
        
        linear_x = 0.0
        linear_y = 0.0
        angular_z = 0.0
        
        keys = pygame.key.get_pressed()
        
        # Linear velocity
        if keys[pygame.K_w]:
            linear_x += 1.0
        if keys[pygame.K_s]:
            linear_x -= 1.0
        if keys[pygame.K_a]:
            linear_y += 1.0
        if keys[pygame.K_d]:
            linear_y -= 1.0
        
        # Angular velocity
        if keys[pygame.K_q]:
            angular_z += 1.0
        if keys[pygame.K_e]:
            angular_z -= 1.0
        
        return np.array([linear_x, linear_y, angular_z], dtype=np.float32)
    
    def _get_joystick_action(self) -> np.ndarray:
        """Get action from joystick"""
        import pygame
        pygame.event.pump()
        
        # Left stick for linear velocity
        linear_x = self.joystick.get_axis(1) * -1  # Invert Y axis
        linear_y = self.joystick.get_axis(0) * -1  # Invert X axis
        
        # Right stick X for angular velocity
        angular_z = self.joystick.get_axis(2) * -1
        
        return np.array([linear_x, linear_y, angular_z], dtype=np.float32)
    
    def collect(self, num_episodes: int = 100):
        """Collect demonstrations from human"""
        policy = lambda obs: self.get_human_action()
        
        print("Human demonstration collection started!")
        print("Controls: WASD for movement, QE for rotation, ESC to stop")
        
        self.collect_with_policy(
            policy=policy,
            num_episodes=num_episodes,
            success_only=False,  # Save all human demos
        )
