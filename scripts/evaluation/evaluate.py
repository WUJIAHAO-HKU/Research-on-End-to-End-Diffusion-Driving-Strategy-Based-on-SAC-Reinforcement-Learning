"""
Evaluation Script

Evaluate trained policies in simulation.
"""

import hydra
from omegaconf import DictConfig
import torch
import numpy as np
from pathlib import Path
import sys
from tqdm import tqdm
import pickle

sys.path.append(str(Path(__file__).parent.parent))

from src.envs.isaac_lab.rosorin_car_env import ROSOrinDrivingEnv
from src.models.sac.sac_agent import SACAgent
from src.utils.metrics import EvaluationSuite
from src.utils.visualization import TrajectoryVisualizer, plot_bar_chart
from src.utils.logger import Logger


class Evaluator:
    """Evaluate trained policy"""
    
    def __init__(self, config: DictConfig):
        self.config = config
        self.device = config.device
        
        # Create environment
        self.env = self._create_env()
        
        # Load agent
        self.agent = self._load_agent()
        
        # Evaluation suite
        self.eval_suite = EvaluationSuite()
        
        # Visualizer
        self.vis = TrajectoryVisualizer()
        
        # Logger
        self.logger = Logger(
            log_dir=Path(config.save_dir) / "eval_logs",
            experiment_name=f"eval_{config.checkpoint_name}",
            use_tensorboard=False,
            use_wandb=config.logging.wandb,
        )
    
    def _create_env(self):
        """Create evaluation environment"""
        # Initialize Isaac Lab environment
        env = ROSOrinDrivingEnv(self.config.env)
        return env
    
    def _load_agent(self):
        """Load trained agent"""
        checkpoint_path = Path(self.config.checkpoint_path)
        
        # Create agent
        agent = SACAgent(
            observation_encoders=self.config.model.encoders,
            action_dim=self.config.model.action_dim,
            **self.config.model.sac_agent
        )
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        agent.load_state_dict(checkpoint['model_state_dict'])
        
        self.logger.info(f"Loaded checkpoint from {checkpoint_path}")
        self.logger.info(f"Checkpoint step: {checkpoint.get('step', 'unknown')}")
        
        return agent
    
    def evaluate(self, num_episodes: int = 100):
        """Run evaluation"""
        self.logger.info(f"Evaluating for {num_episodes} episodes...")
        
        trajectories = []
        
        for episode in tqdm(range(num_episodes)):
            trajectory = self._run_episode()
            trajectories.append(trajectory)
            
            # Log episode results
            self.logger.info(
                f"Episode {episode}: "
                f"return={trajectory['return']:.2f}, "
                f"length={trajectory['length']}, "
                f"success={trajectory['success']}"
            )
        
        # Compute metrics
        results = self.eval_suite.evaluate_trajectories(
            trajectories,
            obstacles=self.env.obstacles if hasattr(self.env, 'obstacles') else None,
        )
        
        # Print summary
        self.eval_suite.print_summary(results)
        
        # Save results
        self._save_results(results, trajectories)
        
        # Visualize
        self._visualize_results(trajectories, results)
        
        return results
    
    def _run_episode(self):
        """Run single episode"""
        obs = self.env.reset()
        done = False
        
        trajectory = {
            'observations': [],
            'actions': [],
            'rewards': [],
            'positions': [],
            'orientations': [],
        }
        
        while not done:
            # Select action
            action = self.agent.select_action(obs, eval_mode=True)
            
            # Step environment
            next_obs, reward, done, info = self.env.step(action)
            
            # Store
            trajectory['observations'].append(obs)
            trajectory['actions'].append(action)
            trajectory['rewards'].append(reward)
            
            if 'position' in info:
                trajectory['positions'].append(info['position'])
            if 'orientation' in info:
                trajectory['orientations'].append(info['orientation'])
            
            obs = next_obs
        
        # Convert to arrays
        trajectory['actions'] = np.array(trajectory['actions'])
        trajectory['rewards'] = np.array(trajectory['rewards'])
        trajectory['positions'] = np.array(trajectory['positions'])
        trajectory['orientations'] = np.array(trajectory['orientations'])
        
        trajectory['return'] = trajectory['rewards'].sum()
        trajectory['length'] = len(trajectory['rewards'])
        trajectory['success'] = info.get('success', False)
        trajectory['collision'] = info.get('collision', False)
        
        return trajectory
    
    def _save_results(self, results, trajectories):
        """Save evaluation results"""
        save_dir = Path(self.config.save_dir) / "eval_results"
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metrics
        with open(save_dir / "metrics.pkl", 'wb') as f:
            pickle.dump(results, f)
        
        # Save trajectories
        with open(save_dir / "trajectories.pkl", 'wb') as f:
            pickle.dump(trajectories, f)
        
        self.logger.info(f"Results saved to {save_dir}")
    
    def _visualize_results(self, trajectories, results):
        """Generate visualizations"""
        save_dir = Path(self.config.save_dir) / "visualizations"
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Plot sample trajectories
        for i, traj in enumerate(trajectories[:5]):
            fig = self.vis.plot_trajectory(
                traj,
                save_path=save_dir / f"trajectory_{i}.png"
            )
            plt.close(fig)
        
        # Plot metrics
        metric_data = {
            'Success Rate': results['success_rate'] * 100,
            'Collision Rate': results['collision_rate'] * 100,
            'Path Efficiency': results['path_efficiency']['mean'],
            'Avg Speed': results['average_speed']['mean'],
        }
        
        fig = plot_bar_chart(
            metric_data,
            title="Evaluation Metrics",
            save_path=save_dir / "metrics.png"
        )
        plt.close(fig)


@hydra.main(config_path="../configs", config_name="evaluate", version_base="1.2")
def main(config: DictConfig):
    """Main entry point"""
    evaluator = Evaluator(config)
    evaluator.evaluate(num_episodes=config.num_episodes)


if __name__ == "__main__":
    main()
