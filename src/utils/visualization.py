"""
Visualization Utilities

Generate plots and videos for analysis and debugging.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, PillowWriter
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import torch
import cv2


# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)


class TrajectoryVisualizer:
    """
    Visualize robot trajectories and navigation results.
    """
    
    def __init__(self, map_size: Tuple[float, float] = (10.0, 10.0)):
        self.map_size = map_size
    
    def plot_trajectory(
        self,
        trajectory: Dict[str, np.ndarray],
        save_path: Optional[Path] = None,
        show_arrows: bool = True,
        show_obstacles: bool = True,
        obstacles: Optional[List[Dict]] = None,
    ) -> plt.Figure:
        """
        Plot 2D trajectory with orientation arrows.
        
        Args:
            trajectory: Dict with 'positions' (N, 2) and 'orientations' (N,)
            save_path: Path to save figure
            show_arrows: Show orientation arrows
            show_obstacles: Show obstacles if provided
            obstacles: List of obstacle dicts with 'position' and 'radius'
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 10))
        
        positions = trajectory['positions']
        orientations = trajectory.get('orientations', None)
        
        # Plot trajectory
        ax.plot(positions[:, 0], positions[:, 1], 'b-', linewidth=2, label='Trajectory')
        
        # Plot start and goal
        ax.plot(positions[0, 0], positions[0, 1], 'go', markersize=15, label='Start')
        ax.plot(positions[-1, 0], positions[-1, 1], 'r*', markersize=20, label='Goal')
        
        # Plot orientation arrows
        if show_arrows and orientations is not None:
            arrow_interval = max(1, len(positions) // 20)
            for i in range(0, len(positions), arrow_interval):
                dx = 0.3 * np.cos(orientations[i])
                dy = 0.3 * np.sin(orientations[i])
                ax.arrow(
                    positions[i, 0], positions[i, 1], dx, dy,
                    head_width=0.2, head_length=0.15, fc='blue', ec='blue', alpha=0.6
                )
        
        # Plot obstacles
        if show_obstacles and obstacles is not None:
            for obs in obstacles:
                circle = plt.Circle(
                    obs['position'], obs['radius'],
                    color='red', alpha=0.3
                )
                ax.add_patch(circle)
        
        # Set limits
        ax.set_xlim(-self.map_size[0]/2, self.map_size[0]/2)
        ax.set_ylim(-self.map_size[1]/2, self.map_size[1]/2)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title('Robot Trajectory')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_trajectory_comparison(
        self,
        trajectories: Dict[str, Dict[str, np.ndarray]],
        save_path: Optional[Path] = None,
    ) -> plt.Figure:
        """
        Compare multiple trajectories.
        
        Args:
            trajectories: Dict of {name: trajectory_dict}
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(12, 10))
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(trajectories)))
        
        for (name, traj), color in zip(trajectories.items(), colors):
            positions = traj['positions']
            ax.plot(positions[:, 0], positions[:, 1], 
                   linewidth=2, label=name, color=color, alpha=0.7)
            
            # Mark start and end
            ax.plot(positions[0, 0], positions[0, 1], 'o', 
                   color=color, markersize=10)
            ax.plot(positions[-1, 0], positions[-1, 1], '*', 
                   color=color, markersize=15)
        
        ax.set_xlim(-self.map_size[0]/2, self.map_size[0]/2)
        ax.set_ylim(-self.map_size[1]/2, self.map_size[1]/2)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title('Trajectory Comparison')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def create_trajectory_animation(
        self,
        trajectory: Dict[str, np.ndarray],
        save_path: Path,
        fps: int = 30,
        obstacles: Optional[List[Dict]] = None,
    ):
        """
        Create animated trajectory video.
        
        Args:
            trajectory: Trajectory dict
            save_path: Path to save video
            fps: Frames per second
            obstacles: Optional obstacles to visualize
        """
        positions = trajectory['positions']
        orientations = trajectory.get('orientations', None)
        
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Plot obstacles
        if obstacles is not None:
            for obs in obstacles:
                circle = plt.Circle(
                    obs['position'], obs['radius'],
                    color='red', alpha=0.3
                )
                ax.add_patch(circle)
        
        # Initialize plot elements
        line, = ax.plot([], [], 'b-', linewidth=2)
        robot = plt.Circle((0, 0), 0.2, color='blue', alpha=0.7)
        ax.add_patch(robot)
        arrow = ax.arrow(0, 0, 0, 0, head_width=0.2, head_length=0.15, fc='blue', ec='blue')
        
        ax.set_xlim(-self.map_size[0]/2, self.map_size[0]/2)
        ax.set_ylim(-self.map_size[1]/2, self.map_size[1]/2)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title('Robot Navigation')
        
        def update(frame):
            # Update trajectory
            line.set_data(positions[:frame, 0], positions[:frame, 1])
            
            # Update robot position
            robot.center = positions[frame]
            
            # Update orientation arrow
            if orientations is not None:
                dx = 0.4 * np.cos(orientations[frame])
                dy = 0.4 * np.sin(orientations[frame])
                arrow.set_data(
                    x=positions[frame, 0],
                    y=positions[frame, 1],
                    dx=dx,
                    dy=dy
                )
            
            return line, robot, arrow
        
        anim = FuncAnimation(
            fig, update, frames=len(positions),
            interval=1000/fps, blit=True
        )
        
        writer = PillowWriter(fps=fps)
        anim.save(save_path, writer=writer)
        plt.close()


class ActionVisualizer:
    """
    Visualize action distributions and sequences.
    """
    
    def plot_action_distribution(
        self,
        actions: np.ndarray,
        action_names: List[str] = None,
        save_path: Optional[Path] = None,
    ) -> plt.Figure:
        """
        Plot action distribution histograms.
        
        Args:
            actions: (N, action_dim) array
            action_names: Names for each action dimension
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        action_dim = actions.shape[1]
        
        if action_names is None:
            action_names = [f'Action {i}' for i in range(action_dim)]
        
        fig, axes = plt.subplots(1, action_dim, figsize=(5*action_dim, 4))
        
        if action_dim == 1:
            axes = [axes]
        
        for i, (ax, name) in enumerate(zip(axes, action_names)):
            ax.hist(actions[:, i], bins=50, alpha=0.7, color='blue', edgecolor='black')
            ax.set_xlabel(name)
            ax.set_ylabel('Frequency')
            ax.set_title(f'{name} Distribution')
            ax.grid(True, alpha=0.3)
            
            # Add statistics
            mean = actions[:, i].mean()
            std = actions[:, i].std()
            ax.axvline(mean, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean:.2f}')
            ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_action_heatmap(
        self,
        actions: np.ndarray,
        save_path: Optional[Path] = None,
    ) -> plt.Figure:
        """
        Plot action sequence as heatmap.
        
        Args:
            actions: (T, action_dim) array
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        im = ax.imshow(actions.T, aspect='auto', cmap='RdBu_r', 
                      interpolation='nearest', vmin=-1, vmax=1)
        
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Action Dimension')
        ax.set_title('Action Sequence Heatmap')
        
        plt.colorbar(im, ax=ax, label='Action Value')
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig


class TrainingVisualizer:
    """
    Visualize training curves and metrics.
    """
    
    def plot_learning_curves(
        self,
        metrics: Dict[str, List[float]],
        save_path: Optional[Path] = None,
        smooth_window: int = 10,
    ) -> plt.Figure:
        """
        Plot training curves with smoothing.
        
        Args:
            metrics: Dict of {metric_name: values_list}
            save_path: Path to save figure
            smooth_window: Window size for smoothing
            
        Returns:
            Matplotlib figure
        """
        num_metrics = len(metrics)
        fig, axes = plt.subplots(num_metrics, 1, figsize=(12, 4*num_metrics))
        
        if num_metrics == 1:
            axes = [axes]
        
        for ax, (name, values) in zip(axes, metrics.items()):
            steps = np.arange(len(values))
            
            # Plot raw values
            ax.plot(steps, values, alpha=0.3, color='blue', label='Raw')
            
            # Plot smoothed values
            if len(values) > smooth_window:
                smoothed = np.convolve(
                    values, 
                    np.ones(smooth_window)/smooth_window, 
                    mode='valid'
                )
                smooth_steps = steps[smooth_window-1:]
                ax.plot(smooth_steps, smoothed, color='red', linewidth=2, label='Smoothed')
            
            ax.set_xlabel('Training Step')
            ax.set_ylabel(name)
            ax.set_title(f'{name} over Training')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_comparison_curves(
        self,
        metrics: Dict[str, Dict[str, List[float]]],
        save_path: Optional[Path] = None,
    ) -> plt.Figure:
        """
        Compare training curves from different runs.
        
        Args:
            metrics: Dict of {metric_name: {run_name: values}}
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        num_metrics = len(metrics)
        fig, axes = plt.subplots(num_metrics, 1, figsize=(12, 4*num_metrics))
        
        if num_metrics == 1:
            axes = [axes]
        
        for ax, (metric_name, runs) in zip(axes, metrics.items()):
            for run_name, values in runs.items():
                steps = np.arange(len(values))
                ax.plot(steps, values, label=run_name, linewidth=2, alpha=0.8)
            
            ax.set_xlabel('Training Step')
            ax.set_ylabel(metric_name)
            ax.set_title(f'{metric_name} Comparison')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig


class ObservationVisualizer:
    """
    Visualize observations (vision, lidar, etc.)
    """
    
    def visualize_rgbd(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        save_path: Optional[Path] = None,
    ) -> plt.Figure:
        """
        Visualize RGB and depth images side by side.
        
        Args:
            rgb: (H, W, 3) RGB image
            depth: (H, W) depth image
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # RGB
        axes[0].imshow(rgb)
        axes[0].set_title('RGB Image')
        axes[0].axis('off')
        
        # Depth
        im = axes[1].imshow(depth, cmap='viridis')
        axes[1].set_title('Depth Image')
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1], label='Depth (m)')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def visualize_lidar(
        self,
        points: np.ndarray,
        save_path: Optional[Path] = None,
    ) -> plt.Figure:
        """
        Visualize LiDAR point cloud (top-down view).
        
        Args:
            points: (N, 3) point cloud
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Plot points (top-down view)
        scatter = ax.scatter(points[:, 0], points[:, 1], 
                           c=points[:, 2], cmap='viridis', 
                           s=5, alpha=0.6)
        
        # Robot position at origin
        ax.plot(0, 0, 'r*', markersize=20, label='Robot')
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title('LiDAR Point Cloud (Top-Down View)')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.colorbar(scatter, ax=ax, label='Height (m)')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def create_observation_video(
        self,
        observations: List[Dict[str, np.ndarray]],
        save_path: Path,
        fps: int = 30,
    ):
        """
        Create video from observation sequence.
        
        Args:
            observations: List of observation dicts with 'vision' key
            save_path: Path to save video
            fps: Frames per second
        """
        if not observations or 'vision' not in observations[0]:
            return
        
        # Get dimensions
        first_obs = observations[0]['vision']
        if first_obs.shape[0] == 4:  # RGBD
            rgb = first_obs[:3].transpose(1, 2, 0)
        else:
            rgb = first_obs.transpose(1, 2, 0)
        
        height, width = rgb.shape[:2]
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(
            str(save_path), fourcc, fps, (width, height)
        )
        
        for obs in observations:
            img = obs['vision'][:3].transpose(1, 2, 0)
            img = (img * 255).astype(np.uint8)
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            video_writer.write(img_bgr)
        
        video_writer.release()


def plot_bar_chart(
    data: Dict[str, float],
    title: str = "Comparison",
    xlabel: str = "Method",
    ylabel: str = "Value",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Create bar chart for comparing methods.
    
    Args:
        data: Dict of {method_name: value}
        title: Chart title
        xlabel: X-axis label
        ylabel: Y-axis label
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    methods = list(data.keys())
    values = list(data.values())
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(methods)))
    bars = ax.bar(methods, values, color=colors, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{value:.2f}',
               ha='center', va='bottom', fontsize=10)
    
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig
