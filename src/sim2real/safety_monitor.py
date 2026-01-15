"""
Safety Monitor for Real Robot Deployment

Monitors robot state and enforces safety constraints.
"""

import numpy as np
from typing import Dict, Optional, Tuple
import logging


class SafetyMonitor:
    """
    Safety monitor that enforces constraints and emergency stops.
    
    Features:
    - Velocity limiting
    - Collision avoidance
    - Emergency stop
    - Tilt detection
    """
    
    def __init__(
        self,
        max_linear_velocity: float = 1.0,
        max_angular_velocity: float = 2.0,
        collision_distance: float = 0.3,
        max_tilt_angle: float = 30.0,  # degrees
        enable_collision_avoidance: bool = True,
    ):
        self.max_linear_velocity = max_linear_velocity
        self.max_angular_velocity = max_angular_velocity
        self.collision_distance = collision_distance
        self.max_tilt_angle = np.deg2rad(max_tilt_angle)
        self.enable_collision_avoidance = enable_collision_avoidance
        
        # Emergency stop flag
        self.emergency_stop = False
        
        # Logging
        self.logger = logging.getLogger('SafetyMonitor')
        self.logger.setLevel(logging.INFO)
    
    def check_action(
        self,
        action: np.ndarray,
        observation: Dict[str, np.ndarray],
    ) -> Tuple[np.ndarray, bool]:
        """
        Check and modify action for safety.
        
        Args:
            action: Commanded action [vx, vy, omega]
            observation: Current observation with sensor data
            
        Returns:
            safe_action: Modified safe action
            is_safe: Whether action is safe
        """
        if self.emergency_stop:
            return np.zeros_like(action), False
        
        safe_action = action.copy()
        is_safe = True
        
        # Velocity limiting
        linear_velocity = np.linalg.norm(action[:2])
        if linear_velocity > self.max_linear_velocity:
            scale = self.max_linear_velocity / linear_velocity
            safe_action[:2] *= scale
            is_safe = False
            self.logger.warning(f"Linear velocity limited: {linear_velocity:.2f} -> {self.max_linear_velocity:.2f}")
        
        angular_velocity = abs(action[2])
        if angular_velocity > self.max_angular_velocity:
            safe_action[2] = np.sign(action[2]) * self.max_angular_velocity
            is_safe = False
            self.logger.warning(f"Angular velocity limited: {angular_velocity:.2f} -> {self.max_angular_velocity:.2f}")
        
        # Collision avoidance
        if self.enable_collision_avoidance and 'lidar' in observation:
            collision_risk = self._check_collision_risk(
                observation['lidar'],
                safe_action
            )
            
            if collision_risk:
                safe_action = self._apply_collision_avoidance(
                    safe_action,
                    observation['lidar']
                )
                is_safe = False
                self.logger.warning("Collision risk detected, applying avoidance")
        
        # Tilt detection
        if 'imu' in observation:
            tilt_angle = self._get_tilt_angle(observation['imu'])
            if tilt_angle > self.max_tilt_angle:
                self.emergency_stop = True
                self.logger.error(f"Excessive tilt detected: {np.rad2deg(tilt_angle):.1f}°")
                return np.zeros_like(action), False
        
        return safe_action, is_safe
    
    def _check_collision_risk(
        self,
        lidar_points: np.ndarray,
        action: np.ndarray,
    ) -> bool:
        """
        Check if action will lead to collision.
        
        Args:
            lidar_points: (N, 3) LiDAR points
            action: Commanded action
            
        Returns:
            collision_risk: True if collision risk detected
        """
        if len(lidar_points) == 0:
            return False
        
        # Get points in front of robot (in direction of motion)
        velocity_direction = action[:2]
        if np.linalg.norm(velocity_direction) < 0.1:
            return False
        
        velocity_direction = velocity_direction / np.linalg.norm(velocity_direction)
        
        # Filter points in direction of motion
        points_xy = lidar_points[:, :2]
        dot_products = points_xy @ velocity_direction
        
        # Points in front (dot product > 0)
        front_points = points_xy[dot_products > 0]
        
        if len(front_points) == 0:
            return False
        
        # Check minimum distance
        distances = np.linalg.norm(front_points, axis=1)
        min_distance = distances.min()
        
        return min_distance < self.collision_distance
    
    def _apply_collision_avoidance(
        self,
        action: np.ndarray,
        lidar_points: np.ndarray,
    ) -> np.ndarray:
        """
        Modify action to avoid collision.
        
        Simple strategy: reduce forward velocity or stop.
        """
        # Reduce velocity significantly
        safe_action = action.copy()
        safe_action[:2] *= 0.2  # Reduce to 20%
        
        # Or stop completely if very close
        if len(lidar_points) > 0:
            min_distance = np.linalg.norm(lidar_points[:, :2], axis=1).min()
            if min_distance < self.collision_distance * 0.5:
                safe_action[:2] = 0.0
        
        return safe_action
    
    def _get_tilt_angle(self, imu_data: Dict) -> float:
        """
        Compute tilt angle from IMU data.
        
        Args:
            imu_data: IMU dictionary with 'orientation' quaternion
            
        Returns:
            tilt_angle: Tilt angle in radians
        """
        # Extract roll and pitch from quaternion
        quat = imu_data.get('orientation', [1, 0, 0, 0])  # w, x, y, z
        
        # Compute roll and pitch
        roll = np.arctan2(
            2.0 * (quat[0] * quat[1] + quat[2] * quat[3]),
            1.0 - 2.0 * (quat[1]**2 + quat[2]**2)
        )
        
        pitch = np.arcsin(2.0 * (quat[0] * quat[2] - quat[3] * quat[1]))
        
        # Total tilt
        tilt = np.sqrt(roll**2 + pitch**2)
        
        return tilt
    
    def trigger_emergency_stop(self, reason: str = "Manual"):
        """Trigger emergency stop"""
        self.emergency_stop = True
        self.logger.error(f"EMERGENCY STOP: {reason}")
    
    def reset_emergency_stop(self):
        """Reset emergency stop"""
        self.emergency_stop = False
        self.logger.info("Emergency stop reset")
    
    def get_status(self) -> Dict:
        """Get safety monitor status"""
        return {
            'emergency_stop': self.emergency_stop,
            'max_linear_velocity': self.max_linear_velocity,
            'max_angular_velocity': self.max_angular_velocity,
            'collision_distance': self.collision_distance,
            'max_tilt_angle': np.rad2deg(self.max_tilt_angle),
        }
