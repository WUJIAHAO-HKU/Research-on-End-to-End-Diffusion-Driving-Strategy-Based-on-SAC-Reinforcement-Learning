"""
Evaluation Metrics for Autonomous Driving

Metrics for:
- Navigation performance
- Safety
- Efficiency
- Comfort
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy.spatial import distance
from scipy.signal import find_peaks


class NavigationMetrics:
    """
    Metrics for evaluating navigation performance.
    """
    
    @staticmethod
    def success_rate(
        trajectories: List[Dict],
        success_threshold: float = 0.5,
    ) -> float:
        """
        Calculate success rate.
        
        Args:
            trajectories: List of trajectory dicts with 'success' key
            success_threshold: Minimum distance threshold to goal (m)
            
        Returns:
            Success rate [0, 1]
        """
        if not trajectories:
            return 0.0
        
        successes = sum(traj.get('success', False) for traj in trajectories)
        return successes / len(trajectories)
    
    @staticmethod
    def goal_distance_error(
        final_positions: np.ndarray,
        goal_positions: np.ndarray,
    ) -> Dict[str, float]:
        """
        Calculate distance to goal at end of episode.
        
        Args:
            final_positions: (N, 2) final robot positions
            goal_positions: (N, 2) goal positions
            
        Returns:
            Dict with mean, std, min, max errors
        """
        errors = np.linalg.norm(final_positions - goal_positions, axis=1)
        
        return {
            'mean': float(errors.mean()),
            'std': float(errors.std()),
            'min': float(errors.min()),
            'max': float(errors.max()),
            'median': float(np.median(errors)),
        }
    
    @staticmethod
    def path_efficiency(
        trajectories: List[Dict],
        optimal_lengths: Optional[List[float]] = None,
    ) -> Dict[str, float]:
        """
        Calculate path efficiency (actual length / optimal length).
        
        Args:
            trajectories: List of trajectory dicts with 'positions'
            optimal_lengths: Optimal path lengths (e.g., from A*)
            
        Returns:
            Dict with efficiency statistics
        """
        efficiencies = []
        
        for i, traj in enumerate(trajectories):
            positions = traj['positions']
            actual_length = np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1))
            
            if optimal_lengths is not None:
                optimal = optimal_lengths[i]
            else:
                # Use straight-line distance as optimal
                optimal = np.linalg.norm(positions[-1] - positions[0])
            
            efficiency = optimal / max(actual_length, 1e-6)
            efficiencies.append(efficiency)
        
        efficiencies = np.array(efficiencies)
        
        return {
            'mean': float(efficiencies.mean()),
            'std': float(efficiencies.std()),
            'min': float(efficiencies.min()),
            'max': float(efficiencies.max()),
        }
    
    @staticmethod
    def path_smoothness(trajectories: List[Dict]) -> Dict[str, float]:
        """
        Calculate path smoothness (based on curvature).
        
        Lower values indicate smoother paths.
        
        Args:
            trajectories: List of trajectory dicts
            
        Returns:
            Dict with smoothness statistics
        """
        smoothness_values = []
        
        for traj in trajectories:
            positions = traj['positions']
            
            if len(positions) < 3:
                continue
            
            # Compute curvature
            dx = np.diff(positions[:, 0])
            dy = np.diff(positions[:, 1])
            ddx = np.diff(dx)
            ddy = np.diff(dy)
            
            curvature = np.abs(dx[:-1] * ddy - dy[:-1] * ddx) / \
                       np.power(dx[:-1]**2 + dy[:-1]**2, 1.5)
            
            # Average curvature as smoothness metric
            smoothness = float(curvature.mean())
            smoothness_values.append(smoothness)
        
        smoothness_values = np.array(smoothness_values)
        
        return {
            'mean': float(smoothness_values.mean()),
            'std': float(smoothness_values.std()),
            'min': float(smoothness_values.min()),
            'max': float(smoothness_values.max()),
        }
    
    @staticmethod
    def completion_time(trajectories: List[Dict]) -> Dict[str, float]:
        """
        Calculate episode completion time.
        
        Args:
            trajectories: List of trajectory dicts with 'length' key
            
        Returns:
            Dict with time statistics (in steps)
        """
        times = np.array([traj['length'] for traj in trajectories])
        
        return {
            'mean': float(times.mean()),
            'std': float(times.std()),
            'min': float(times.min()),
            'max': float(times.max()),
        }


class SafetyMetrics:
    """
    Metrics for evaluating safety.
    """
    
    @staticmethod
    def collision_rate(trajectories: List[Dict]) -> float:
        """
        Calculate collision rate.
        
        Args:
            trajectories: List of trajectory dicts with 'collision' key
            
        Returns:
            Collision rate [0, 1]
        """
        if not trajectories:
            return 0.0
        
        collisions = sum(traj.get('collision', False) for traj in trajectories)
        return collisions / len(trajectories)
    
    @staticmethod
    def minimum_clearance(
        trajectories: List[Dict],
        obstacles: List[Dict],
    ) -> Dict[str, float]:
        """
        Calculate minimum clearance to obstacles.
        
        Args:
            trajectories: List of trajectory dicts
            obstacles: List of obstacle dicts with 'position' and 'radius'
            
        Returns:
            Dict with clearance statistics
        """
        min_clearances = []
        
        for traj in trajectories:
            positions = traj['positions']
            
            # Compute distance to each obstacle
            clearances = []
            for obs in obstacles:
                obs_pos = obs['position']
                obs_radius = obs['radius']
                
                distances = np.linalg.norm(positions - obs_pos, axis=1)
                clearance = distances.min() - obs_radius
                clearances.append(clearance)
            
            if clearances:
                min_clearances.append(min(clearances))
        
        min_clearances = np.array(min_clearances)
        
        return {
            'mean': float(min_clearances.mean()),
            'std': float(min_clearances.std()),
            'min': float(min_clearances.min()),
            'max': float(min_clearances.max()),
        }
    
    @staticmethod
    def time_to_collision(
        positions: np.ndarray,
        velocities: np.ndarray,
        obstacles: List[Dict],
    ) -> np.ndarray:
        """
        Calculate time to collision (TTC) for each timestep.
        
        Args:
            positions: (T, 2) robot positions
            velocities: (T, 2) robot velocities
            obstacles: List of obstacle dicts
            
        Returns:
            (T,) array of TTC values (inf if no collision)
        """
        T = len(positions)
        ttc = np.full(T, np.inf)
        
        for i in range(T):
            pos = positions[i]
            vel = velocities[i]
            
            if np.linalg.norm(vel) < 1e-6:
                continue
            
            for obs in obstacles:
                obs_pos = obs['position']
                obs_radius = obs['radius']
                
                # Vector from robot to obstacle
                to_obs = obs_pos - pos
                
                # Project velocity onto direction to obstacle
                proj = np.dot(to_obs, vel) / (np.linalg.norm(vel) + 1e-6)
                
                if proj > 0:  # Moving towards obstacle
                    dist = np.linalg.norm(to_obs)
                    collision_dist = obs_radius + 0.2  # Robot radius
                    
                    if dist < collision_dist:
                        ttc[i] = 0
                    else:
                        time = (dist - collision_dist) / np.linalg.norm(vel)
                        ttc[i] = min(ttc[i], time)
        
        return ttc


class ComfortMetrics:
    """
    Metrics for evaluating ride comfort.
    """
    
    @staticmethod
    def jerk(trajectories: List[Dict], dt: float = 0.1) -> Dict[str, float]:
        """
        Calculate jerk (rate of change of acceleration).
        
        Lower jerk indicates smoother motion.
        
        Args:
            trajectories: List of trajectory dicts with 'velocities'
            dt: Time step (seconds)
            
        Returns:
            Dict with jerk statistics
        """
        jerk_values = []
        
        for traj in trajectories:
            velocities = traj.get('velocities', None)
            
            if velocities is None or len(velocities) < 3:
                continue
            
            # Compute acceleration
            acc = np.diff(velocities, axis=0) / dt
            
            # Compute jerk
            jerk = np.diff(acc, axis=0) / dt
            jerk_magnitude = np.linalg.norm(jerk, axis=1)
            
            jerk_values.append(jerk_magnitude.mean())
        
        jerk_values = np.array(jerk_values)
        
        return {
            'mean': float(jerk_values.mean()),
            'std': float(jerk_values.std()),
            'min': float(jerk_values.min()),
            'max': float(jerk_values.max()),
        }
    
    @staticmethod
    def lateral_acceleration(
        trajectories: List[Dict],
        dt: float = 0.1,
    ) -> Dict[str, float]:
        """
        Calculate lateral acceleration.
        
        Args:
            trajectories: List of trajectory dicts
            dt: Time step (seconds)
            
        Returns:
            Dict with lateral acceleration statistics
        """
        lat_acc_values = []
        
        for traj in trajectories:
            positions = traj['positions']
            orientations = traj.get('orientations', None)
            
            if len(positions) < 3 or orientations is None:
                continue
            
            # Compute velocity
            velocities = np.diff(positions, axis=0) / dt
            
            # Compute heading direction
            heading = np.stack([
                np.cos(orientations[:-1]),
                np.sin(orientations[:-1])
            ], axis=1)
            
            # Lateral direction (perpendicular to heading)
            lateral = np.stack([
                -np.sin(orientations[:-1]),
                np.cos(orientations[:-1])
            ], axis=1)
            
            # Project velocity onto lateral direction
            lateral_velocity = np.sum(velocities * lateral, axis=1)
            
            # Compute lateral acceleration
            lateral_acc = np.diff(lateral_velocity) / dt
            
            lat_acc_values.append(np.abs(lateral_acc).mean())
        
        lat_acc_values = np.array(lat_acc_values)
        
        return {
            'mean': float(lat_acc_values.mean()),
            'std': float(lat_acc_values.std()),
            'min': float(lat_acc_values.min()),
            'max': float(lat_acc_values.max()),
        }
    
    @staticmethod
    def control_smoothness(actions: np.ndarray) -> Dict[str, float]:
        """
        Calculate control smoothness (action variations).
        
        Args:
            actions: (T, action_dim) action sequence
            
        Returns:
            Dict with smoothness statistics
        """
        # Compute action differences
        action_diff = np.diff(actions, axis=0)
        action_var = np.linalg.norm(action_diff, axis=1)
        
        return {
            'mean': float(action_var.mean()),
            'std': float(action_var.std()),
            'max': float(action_var.max()),
        }


class EfficiencyMetrics:
    """
    Metrics for evaluating efficiency.
    """
    
    @staticmethod
    def energy_consumption(
        actions: np.ndarray,
        velocities: np.ndarray,
        dt: float = 0.1,
    ) -> float:
        """
        Estimate energy consumption.
        
        Simplified model: E = sum(|v| * |a| * dt)
        
        Args:
            actions: (T, action_dim) control actions
            velocities: (T, 2) velocities
            dt: Time step (seconds)
            
        Returns:
            Total energy consumption
        """
        action_magnitude = np.linalg.norm(actions, axis=1)
        velocity_magnitude = np.linalg.norm(velocities, axis=1)
        
        energy = np.sum(action_magnitude * velocity_magnitude * dt)
        
        return float(energy)
    
    @staticmethod
    def average_speed(trajectories: List[Dict]) -> Dict[str, float]:
        """
        Calculate average speed.
        
        Args:
            trajectories: List of trajectory dicts
            
        Returns:
            Dict with speed statistics
        """
        speeds = []
        
        for traj in trajectories:
            positions = traj['positions']
            
            if len(positions) < 2:
                continue
            
            # Compute speed
            distances = np.linalg.norm(np.diff(positions, axis=0), axis=1)
            speed = distances.mean()
            speeds.append(speed)
        
        speeds = np.array(speeds)
        
        return {
            'mean': float(speeds.mean()),
            'std': float(speeds.std()),
            'min': float(speeds.min()),
            'max': float(speeds.max()),
        }


class EvaluationSuite:
    """
    Comprehensive evaluation suite for autonomous driving.
    """
    
    def __init__(self):
        self.nav_metrics = NavigationMetrics()
        self.safety_metrics = SafetyMetrics()
        self.comfort_metrics = ComfortMetrics()
        self.efficiency_metrics = EfficiencyMetrics()
    
    def evaluate_trajectories(
        self,
        trajectories: List[Dict],
        obstacles: Optional[List[Dict]] = None,
        dt: float = 0.1,
    ) -> Dict[str, Any]:
        """
        Comprehensive evaluation of trajectories.
        
        Args:
            trajectories: List of trajectory dicts
            obstacles: Optional obstacles for safety metrics
            dt: Time step (seconds)
            
        Returns:
            Dict with all metrics
        """
        results = {}
        
        # Navigation metrics
        results['success_rate'] = self.nav_metrics.success_rate(trajectories)
        results['path_efficiency'] = self.nav_metrics.path_efficiency(trajectories)
        results['path_smoothness'] = self.nav_metrics.path_smoothness(trajectories)
        results['completion_time'] = self.nav_metrics.completion_time(trajectories)
        
        # Safety metrics
        results['collision_rate'] = self.safety_metrics.collision_rate(trajectories)
        if obstacles is not None:
            results['minimum_clearance'] = self.safety_metrics.minimum_clearance(
                trajectories, obstacles
            )
        
        # Comfort metrics
        results['jerk'] = self.comfort_metrics.jerk(trajectories, dt)
        results['lateral_acceleration'] = self.comfort_metrics.lateral_acceleration(
            trajectories, dt
        )
        
        # Efficiency metrics
        results['average_speed'] = self.efficiency_metrics.average_speed(trajectories)
        
        return results
    
    def print_summary(self, results: Dict[str, Any]):
        """Print formatted summary of results"""
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        
        print("\n📍 Navigation Performance:")
        print(f"  Success Rate: {results['success_rate']*100:.1f}%")
        print(f"  Path Efficiency: {results['path_efficiency']['mean']:.3f} ± {results['path_efficiency']['std']:.3f}")
        print(f"  Completion Time: {results['completion_time']['mean']:.1f} ± {results['completion_time']['std']:.1f} steps")
        
        print("\n🛡️ Safety Metrics:")
        print(f"  Collision Rate: {results['collision_rate']*100:.1f}%")
        if 'minimum_clearance' in results:
            print(f"  Min Clearance: {results['minimum_clearance']['mean']:.3f} ± {results['minimum_clearance']['std']:.3f} m")
        
        print("\n🎯 Comfort Metrics:")
        print(f"  Average Jerk: {results['jerk']['mean']:.3f} m/s³")
        print(f"  Lateral Acc: {results['lateral_acceleration']['mean']:.3f} m/s²")
        
        print("\n⚡ Efficiency:")
        print(f"  Average Speed: {results['average_speed']['mean']:.3f} ± {results['average_speed']['std']:.3f} m/s")
        
        print("\n" + "="*60 + "\n")
