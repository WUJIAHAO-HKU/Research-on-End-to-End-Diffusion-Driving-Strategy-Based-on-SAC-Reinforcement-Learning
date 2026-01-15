"""
Deployment Script for Real Robot

Deploy trained policy to ROSOrin robot via ROS2.
"""

import hydra
from omegaconf import DictConfig
import torch
from pathlib import Path
import sys
import rclpy

sys.path.append(str(Path(__file__).parent.parent))

from src.models.sac.sac_agent import SACAgent
from src.sim2real.ros2_interface import PolicyNode, GoalPublisher
from src.sim2real.safety_monitor import SafetyMonitor


class RobotDeployment:
    """
    Manages deployment of trained policy to real robot.
    """
    
    def __init__(self, config: DictConfig):
        self.config = config
        self.device = config.device
        
        # Load agent
        self.agent = self._load_agent()
        
        # Initialize ROS2
        rclpy.init()
        
        # Create nodes
        self.policy_node = PolicyNode(
            agent=self.agent,
            config=config,
            control_frequency=config.control_frequency,
        )
        
        self.goal_publisher = GoalPublisher()
        
        # Safety monitor
        if config.enable_safety_monitor:
            self.safety_monitor = SafetyMonitor(
                max_linear_velocity=config.safety.max_linear_velocity,
                max_angular_velocity=config.safety.max_angular_velocity,
                collision_distance=config.safety.collision_distance,
            )
        else:
            self.safety_monitor = None
        
        print("✅ Robot deployment initialized")
    
    def _load_agent(self):
        """Load trained agent from checkpoint"""
        checkpoint_path = Path(self.config.checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Create agent
        agent = SACAgent(
            observation_encoders=self.config.model.encoders,
            action_dim=self.config.model.action_dim,
            **self.config.model.sac_agent
        )
        
        # Load weights
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        agent.load_state_dict(checkpoint['model_state_dict'])
        agent.eval()
        
        print(f"✅ Loaded checkpoint from {checkpoint_path}")
        print(f"   Checkpoint step: {checkpoint.get('step', 'unknown')}")
        
        return agent
    
    def set_goal(self, x: float, y: float, theta: float = 0.0):
        """Set navigation goal"""
        self.goal_publisher.publish_goal(x, y, theta)
        print(f"🎯 Goal set to: ({x:.2f}, {y:.2f}, {theta:.2f})")
    
    def run(self):
        """Run deployment"""
        print("🚀 Starting deployment...")
        print("Press Ctrl+C to stop")
        
        try:
            # Set initial goal if specified
            if self.config.initial_goal:
                self.set_goal(*self.config.initial_goal)
            
            # Spin ROS2 nodes
            rclpy.spin(self.policy_node)
            
        except KeyboardInterrupt:
            print("\n⏹️  Stopping robot...")
            self.policy_node.stop()
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Cleanup resources"""
        self.policy_node.destroy_node()
        self.goal_publisher.destroy_node()
        rclpy.shutdown()
        print("✅ Deployment stopped")


@hydra.main(config_path="../configs", config_name="deploy", version_base="1.2")
def main(config: DictConfig):
    """Main entry point"""
    print("="*60)
    print("ROSOrin Robot Deployment")
    print("="*60)
    print(f"Checkpoint: {config.checkpoint_path}")
    print(f"Control frequency: {config.control_frequency} Hz")
    print(f"Safety monitor: {config.enable_safety_monitor}")
    print("="*60 + "\n")
    
    # Create deployment
    deployment = RobotDeployment(config)
    
    # Run
    deployment.run()


if __name__ == "__main__":
    main()
