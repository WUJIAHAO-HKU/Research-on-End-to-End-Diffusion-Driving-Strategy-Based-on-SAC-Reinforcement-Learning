"""
Isaac Lab Environment for ROSOrin Mecanum Wheel Robot

This module implements the driving environment in NVIDIA Isaac Lab,
supporting parallel simulation and domain randomization.
"""

import torch
import numpy as np
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
import os

# Isaac Lab imports
try:
    from isaaclab.app import AppLauncher
    from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
    from isaaclab.sensors import Camera, CameraCfg, RayCaster, RayCasterCfg
    from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
    from isaaclab.sim import SimulationCfg
    from isaaclab.utils import configclass
    import isaaclab.utils.math as math_utils
except ImportError:
    print("[WARNING] Isaac Lab not found. Using placeholder imports.")
    DirectRLEnv = object
    DirectRLEnvCfg = object


@dataclass
class ObservationSpace:
    """Define observation space structure"""
    rgb_image: torch.Tensor  # (B, 3, H, W)
    depth_image: torch.Tensor  # (B, 1, H, W)
    lidar_scan: torch.Tensor  # (B, num_rays, 3)
    robot_state: torch.Tensor  # (B, state_dim)
    task_info: torch.Tensor  # (B, task_dim)


@dataclass
class ActionSpace:
    """Define action space structure"""
    linear_x: float  # Forward velocity
    linear_y: float  # Lateral velocity (mecanum)
    angular_z: float  # Angular velocity


class ROSOrinDrivingEnv:
    """
    Isaac Lab environment for end-to-end driving with ROSOrin robot.
    
    Features:
    - Multi-modal observations (vision, LiDAR, proprioception)
    - Mecanum wheel dynamics
    - Configurable reward functions
    - Domain randomization for Sim2Real
    - Parallel environment support (GPU-accelerated)
    """
    
    def __init__(
        self,
        cfg: Dict,
        num_envs: int = 64,
        device: str = "cuda:0",
        render: bool = False,
    ):
        """
        Initialize the driving environment.
        
        Args:
            cfg: Configuration dictionary
            num_envs: Number of parallel environments
            device: Device for computation
            render: Whether to enable rendering
        """
        self.cfg = cfg
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.render_mode = render
        
        # Episode tracking
        self.episode_length_buf = torch.zeros(num_envs, dtype=torch.int, device=self.device)
        self.reset_buf = torch.ones(num_envs, dtype=torch.bool, device=self.device)
        self.max_episode_length = int(cfg['env']['episode_length_s'] / cfg['env']['sim']['dt'])
        
        # Observation and action dimensions
        self._setup_spaces()
        
        # Initialize Isaac Lab scene
        self._create_scene()
        
        # Setup sensors
        self._setup_sensors()
        
        # Path and goal generation
        self._generate_paths()
        
        # Reward tracking
        self.episode_rewards = torch.zeros(num_envs, device=self.device)
        self.episode_successes = torch.zeros(num_envs, dtype=torch.int, device=self.device)
        
        print(f"[ROSOrinDrivingEnv] Initialized {num_envs} parallel environments on {device}")
    
    def _setup_spaces(self):
        """Define observation and action space dimensions"""
        cfg = self.cfg
        
        # Observation dimensions
        self.obs_dims = {
            'rgb': (3, cfg['env']['observation']['rgb_camera']['height'],
                    cfg['env']['observation']['rgb_camera']['width']),
            'depth': (1, cfg['env']['observation']['depth_camera']['height'],
                      cfg['env']['observation']['depth_camera']['width']),
            'lidar': (cfg['env']['observation']['lidar']['num_rays'], 3),
            'robot_state': 12,  # vel(3) + angular_vel(3) + quat(4) + wheel_vel(4) - 2 = 12
            'task_info': 3,  # distance_to_goal + goal_position(2)
        }
        
        # Action dimensions
        self.action_dim = 3  # [linear_x, linear_y, angular_z]
        self.action_scale = torch.tensor(
            [1.0, 0.5, 2.0],  # Max speeds
            device=self.device
        )
    
    def _create_scene(self):
        """
        Create the Isaac Lab scene with robot, ground, and obstacles.
        
        This is a placeholder for Isaac Lab integration. When ready:
        1. Create InteractiveScene with scene config
        2. Add ground plane with friction properties
        3. Spawn robot from USD file (rosorin.usd)
        4. Add obstacles from config
        5. Setup lighting (directional, ambient)
        6. Position camera for visualization
        """
        if self.verbose:
            print("[Scene] Creating Isaac Lab scene (placeholder)...")
        
        # TODO: When Isaac Lab is integrated, uncomment:
        # from omni.isaac.lab.scene import InteractiveScene
        # from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR
        
        # scene_cfg = {
        #     "ground": {
        #         "usd_path": f"{ISAAC_NUCLEUS_DIR}/Environments/Grid/default_environment.usd",
        #         "spawn": {"pos": (0.0, 0.0, 0.0)}
        #     },
        #     "robot": {
        #         "usd_path": "data/assets/rosorin/rosorin.usd",
        #         "spawn": {"pos": (0.0, 0.0, 0.1), "quat": (0, 0, 0, 1)},
        #         "articulation": {
        #             "num_dofs": 4,  # 4 wheel joints
        #             "joint_names": ["front_left_wheel_joint", "front_right_wheel_joint",
        #                           "rear_left_wheel_joint", "rear_right_wheel_joint"]
        #         }
        #     }
        # }
        # 
        # self.scene = InteractiveScene(scene_cfg)
        # self.robot = self.scene.articulations["robot"]
        
        pass
    
    def _setup_sensors(self):
        """
        Setup sensors: cameras, LiDAR, IMU.
        
        This is a placeholder for Isaac Lab sensor integration. When ready:
        1. Create Camera sensor with RGB/Depth output
        2. Create RayCaster for LiDAR simulation
        3. Link IMU to robot articulation state
        """
        cfg_obs = self.cfg['env']['observation']
        
        if self.verbose:
            print("[Sensors] Setting up sensors (placeholder)...")
        
        # RGB-D Camera
        if cfg_obs['rgb_camera']['enabled']:
            # TODO: When Isaac Lab is integrated:
            # from omni.isaac.lab.sensors import Camera, CameraCfg
            # 
            # camera_cfg = CameraCfg(
            #     prim_path="/World/Camera",
            #     offset=CameraCfg.OffsetCfg(pos=(0.15, 0.0, 0.08), rot=(0, 0, 0, 1)),
            #     data_types=["rgb", "distance_to_camera"],
            #     spawn=None,  # Camera already exists in USD
            #     width=cfg_obs['rgb_camera']['width'],
            #     height=cfg_obs['rgb_camera']['height']
            # )
            # self.rgb_camera = Camera(camera_cfg)
            pass
        
        # LiDAR (RayCaster)
        if cfg_obs['lidar']['enabled']:
            # TODO: When Isaac Lab is integrated:
            # from omni.isaac.lab.sensors import RayCaster, RayCasterCfg, patterns
            # 
            # lidar_cfg = RayCasterCfg(
            #     prim_path="/World/Robot/lidar",
            #     offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.1)),
            #     attach_yaw_only=True,
            #     pattern_cfg=patterns.LidarPatternCfg(
            #         channels=cfg_obs['lidar']['num_beams'],
            #         vertical_fov_range=(-15.0, 15.0),
            #         horizontal_fov_range=(-180.0, 180.0),
            #     ),
            #     max_distance=cfg_obs['lidar']['max_range'],
            #     drift_range=(0.0, 0.0),
            # )
            # self.lidar = RayCaster(lidar_cfg)
            pass
        
        if self.verbose:
            print("[Sensors] Multi-modal sensor setup complete (placeholder)")
    
    def _generate_paths(self):
        """
        Generate driving paths for each environment.
        
        Supports:
        - Straight paths
        - Curved paths (Bezier curves)
        - Lane following tasks
        """
        # Generate different paths for each environment
        self.paths = []
        for i in range(self.num_envs):
            # TODO: Implement path generation (Bezier, spline, etc.)
            path = self._generate_bezier_path()
            self.paths.append(path)
        
        # Current waypoint indices
        self.current_waypoint_idx = torch.zeros(self.num_envs, dtype=torch.int, device=self.device)
    
    def _generate_bezier_path(self) -> torch.Tensor:
        """Generate a random Bezier curve path"""
        # Control points for cubic Bezier curve
        control_points = torch.rand(4, 2) * 10.0 - 5.0  # Random path from -5 to 5
        
        # Ensure reasonable path (start at origin)
        control_points[0] = torch.tensor([0.0, 0.0])
        control_points[3] = control_points[0] + torch.rand(2) * 8.0 + 2.0  # End 2-10m away
        
        # Sample points along curve
        num_samples = 100
        t = torch.linspace(0, 1, num_samples)
        
        # Cubic Bezier curve: B(t) = (1-t)^3*P0 + 3*(1-t)^2*t*P1 + 3*(1-t)*t^2*P2 + t^3*P3
        t = t.unsqueeze(1)  # (num_samples, 1)
        one_minus_t = 1 - t
        
        path_points = (
            one_minus_t**3 * control_points[0] +
            3 * one_minus_t**2 * t * control_points[1] +
            3 * one_minus_t * t**2 * control_points[2] +
            t**3 * control_points[3]
        )
        
        return path_points  # (num_samples, 2)
    
    def reset(self, env_ids: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Reset specified environments (or all if env_ids is None).
        
        Args:
            env_ids: Indices of environments to reset
            
        Returns:
            observations: Dictionary of observation tensors
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        
        # Reset robot poses
        self._reset_robot_state(env_ids)
        
        # Reset episode counters
        self.episode_length_buf[env_ids] = 0
        self.episode_rewards[env_ids] = 0
        self.reset_buf[env_ids] = False
        
        # Regenerate paths
        # self._generate_paths()  # Could randomize paths per episode
        
        # Domain randomization (if enabled)
        if self.cfg['env']['domain_randomization']['enabled']:
            self._apply_domain_randomization(env_ids)
        
        # Get initial observations
        obs = self._get_observations()
        
        return obs
    
    def _reset_robot_state(self, env_ids: torch.Tensor):
        """Reset robot to initial pose"""
        num_resets = len(env_ids)
        
        # Set robot position to start of path with small random offset
        start_positions = torch.zeros((num_resets, 3), device=self.device)
        start_positions[:, :2] = torch.rand((num_resets, 2), device=self.device) * 0.2 - 0.1  # ±10cm
        start_positions[:, 2] = 0.1  # 10cm above ground
        
        # Random initial orientation (yaw only)
        yaw_angles = (torch.rand(num_resets, device=self.device) - 0.5) * 0.5  # ±15 degrees
        start_orientations = torch.zeros((num_resets, 4), device=self.device)
        start_orientations[:, 3] = torch.cos(yaw_angles / 2)  # w
        start_orientations[:, 2] = torch.sin(yaw_angles / 2)  # z
        
        # Store reset states (will be applied when Isaac Lab is integrated)
        if not hasattr(self, 'robot_positions'):
            self.robot_positions = torch.zeros((self.num_envs, 3), device=self.device)
            self.robot_orientations = torch.zeros((self.num_envs, 4), device=self.device)
            self.robot_orientations[:, 3] = 1.0  # Default quaternion
            self.robot_velocities = torch.zeros((self.num_envs, 6), device=self.device)
            self.wheel_velocities = torch.zeros((self.num_envs, 4), device=self.device)
        
        self.robot_positions[env_ids] = start_positions
        self.robot_orientations[env_ids] = start_orientations
        self.robot_velocities[env_ids] = 0.0
        self.wheel_velocities[env_ids] = 0.0
        
        # Reset path tracking
        self.current_waypoint_idx[env_ids] = 0
    
    def _apply_domain_randomization(self, env_ids: torch.Tensor):
        """
        Apply domain randomization for specified environments.
        
        Randomizes:
        - Dynamics (mass, friction, motor strength)
        - Sensor noise
        - Visual appearance
        - Latency
        """
        cfg_dr = self.cfg['env']['domain_randomization']
        
        if not cfg_dr['enabled']:
            return
        
        num_randomize = len(env_ids)
        
        # 1. Randomize mass (±20%)
        mass_scale = torch.rand(num_randomize, device=self.device) * \
                     (cfg_dr['mass_scale'][1] - cfg_dr['mass_scale'][0]) + \
                     cfg_dr['mass_scale'][0]
        # TODO: When Isaac Lab is integrated, apply:
        # self.robot.set_mass_scale(mass_scale, env_ids)
        
        # 2. Randomize friction coefficients
        friction_scale = torch.rand(num_randomize, device=self.device) * \
                        (cfg_dr['friction_scale'][1] - cfg_dr['friction_scale'][0]) + \
                        cfg_dr['friction_scale'][0]
        # TODO: self.robot.set_friction_coefficients(friction_scale, env_ids)
        
        # 3. Randomize motor strength (already done in _apply_actions, stored for reference)
        if not hasattr(self, 'motor_strength_scale'):
            self.motor_strength_scale = torch.ones(self.num_envs, device=self.device)
        
        motor_strength = torch.rand(num_randomize, device=self.device) * \
                        (cfg_dr['motor_strength_scale'][1] - cfg_dr['motor_strength_scale'][0]) + \
                        cfg_dr['motor_strength_scale'][0]
        self.motor_strength_scale[env_ids] = motor_strength
        
        # 4. Sensor noise is applied in _get_observations()
        # 5. Visual randomization (lighting, textures) would be applied to scene
        # TODO: Randomize lighting intensity, color temperature
        # TODO: Randomize ground texture
        
        if self.verbose:
            print(f"[Domain Rand] Applied to {num_randomize} environments: "
                  f"mass={mass_scale.mean():.2f}, friction={friction_scale.mean():.2f}, "
                  f"motor={motor_strength.mean():.2f}")
    
    def step(self, actions: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict]:
        """
        Step the environment with given actions.
        
        Args:
            actions: Action tensor (num_envs, action_dim)
            
        Returns:
            observations: Next observations
            rewards: Reward for each environment
            dones: Done flags
            infos: Additional information
        """
        # Clip and scale actions
        actions = torch.clamp(actions, -1.0, 1.0)
        scaled_actions = actions * self.action_scale
        
        # Apply actions to robot
        self._apply_actions(scaled_actions)
        
        # Step simulation
        self._step_simulation()
        
        # Get observations
        obs = self._get_observations()
        
        # Compute rewards
        rewards = self._compute_rewards(obs, actions)
        
        # Check termination conditions
        dones = self._check_termination(obs)
        
        # Update episode statistics
        self.episode_length_buf += 1
        self.episode_rewards += rewards
        
        # Handle resets
        reset_env_ids = dones.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset(reset_env_ids)
        
        # Prepare info dict
        infos = self._get_infos()
        
        return obs, rewards, dones, infos
    
    def _apply_actions(self, actions: torch.Tensor):
        """
        Apply velocity commands to the mecanum wheel robot.
        
        Args:
            actions: (num_envs, 3) [linear_x, linear_y, angular_z]
        """
        # Convert body velocities to wheel velocities (mecanum kinematics)
        wheel_velocities = self._mecanum_inverse_kinematics(actions)
        
        # Apply motor strength randomization
        if hasattr(self, 'motor_strength_scales'):
            wheel_velocities = wheel_velocities * self.motor_strength_scales.unsqueeze(1)
        
        # Store for simulation (will be applied when Isaac Lab is integrated)
        self.wheel_velocities = wheel_velocities
        
        # TODO: When Isaac Lab is integrated, use:
        # self.robot.set_joint_velocity_targets(wheel_velocities)
    
    def _mecanum_inverse_kinematics(self, body_vel: torch.Tensor) -> torch.Tensor:
        """
        Convert body frame velocities to wheel velocities.
        
        Mecanum wheel kinematics:
        v_fl = vx - vy - wz * (l + w)
        v_fr = vx + vy + wz * (l + w)
        v_rl = vx + vy - wz * (l + w)
        v_rr = vx - vy + wz * (l + w)
        
        Args:
            body_vel: (N, 3) [vx, vy, wz]
            
        Returns:
            wheel_vel: (N, 4) [fl, fr, rl, rr]
        """
        vx, vy, wz = body_vel[:, 0], body_vel[:, 1], body_vel[:, 2]
        
        # Robot dimensions from config
        wheelbase = self.cfg['robot']['dimensions']['wheelbase']
        track_width = self.cfg['robot']['dimensions']['track_width']
        wheel_radius = self.cfg['robot']['dimensions']['wheel_radius']
        
        lw = (wheelbase + track_width) / 2.0
        
        # Wheel velocities (linear velocity)
        v_fl = vx - vy - wz * lw
        v_fr = vx + vy + wz * lw
        v_rl = vx + vy - wz * lw
        v_rr = vx - vy + wz * lw
        
        # Convert to angular velocity (rad/s)
        wheel_vel = torch.stack([v_fl, v_fr, v_rl, v_rr], dim=1) / wheel_radius
        
        return wheel_vel
    
    def _step_simulation(self):
        """Step the physics simulation"""
        # Simple kinematic simulation for now (will be replaced by Isaac Lab physics)
        dt = self.cfg['env']['sim']['dt']
        
        # Get current velocities from wheel commands
        # Forward kinematics: convert wheel velocities to body velocities
        body_velocities = self._mecanum_forward_kinematics(self.wheel_velocities)
        
        # Update robot velocities
        self.robot_velocities[:, :3] = body_velocities
        
        # Update positions (simple integration)
        # Convert body frame velocity to world frame
        yaw = self._quat_to_yaw(self.robot_orientations)
        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)
        
        vx_world = body_velocities[:, 0] * cos_yaw - body_velocities[:, 1] * sin_yaw
        vy_world = body_velocities[:, 0] * sin_yaw + body_velocities[:, 1] * cos_yaw
        
        self.robot_positions[:, 0] += vx_world * dt
        self.robot_positions[:, 1] += vy_world * dt
        
        # Update orientation
        yaw += body_velocities[:, 2] * dt
        self.robot_orientations[:, 2] = torch.sin(yaw / 2)  # z
        self.robot_orientations[:, 3] = torch.cos(yaw / 2)  # w
        
        # TODO: When Isaac Lab is integrated, use:
        # self.scene.step(render=self.render_mode)
    
    def _mecanum_forward_kinematics(self, wheel_vel: torch.Tensor) -> torch.Tensor:
        """
        Convert wheel velocities to body frame velocities.
        
        Args:
            wheel_vel: (N, 4) [fl, fr, rl, rr] in rad/s
            
        Returns:
            body_vel: (N, 3) [vx, vy, wz]
        """
        # Robot dimensions
        wheelbase = self.cfg['robot']['dimensions']['wheelbase']
        track_width = self.cfg['robot']['dimensions']['track_width']
        wheel_radius = self.cfg['robot']['dimensions']['wheel_radius']
        
        lw = (wheelbase + track_width) / 2.0
        
        # Convert angular to linear velocity
        v_fl, v_fr, v_rl, v_rr = wheel_vel[:, 0], wheel_vel[:, 1], wheel_vel[:, 2], wheel_vel[:, 3]
        v_fl_lin = v_fl * wheel_radius
        v_fr_lin = v_fr * wheel_radius
        v_rl_lin = v_rl * wheel_radius
        v_rr_lin = v_rr * wheel_radius
        
        # Forward kinematics (inverse of the inverse kinematics)
        vx = (v_fl_lin + v_fr_lin + v_rl_lin + v_rr_lin) / 4.0
        vy = (-v_fl_lin + v_fr_lin + v_rl_lin - v_rr_lin) / 4.0
        wz = (-v_fl_lin + v_fr_lin - v_rl_lin + v_rr_lin) / (4.0 * lw)
        
        body_vel = torch.stack([vx, vy, wz], dim=1)
        return body_vel
    
    def _quat_to_yaw(self, quat: torch.Tensor) -> torch.Tensor:
        """Extract yaw angle from quaternion [x, y, z, w]"""
        # yaw = atan2(2*(w*z + x*y), 1 - 2*(y^2 + z^2))
        return torch.atan2(
            2.0 * (quat[:, 3] * quat[:, 2] + quat[:, 0] * quat[:, 1]),
            1.0 - 2.0 * (quat[:, 1]**2 + quat[:, 2]**2)
        )
    
    def _get_observations(self) -> Dict[str, torch.Tensor]:
        """
        Collect multi-modal observations from sensors.
        
        Returns:
            Dictionary with observation tensors
        """
        obs = {}
        
        # RGB image
        if self.cfg['env']['observation']['rgb_camera']['enabled']:
            # obs['rgb'] = self.rgb_camera.get_image()
            obs['rgb'] = torch.zeros(
                (self.num_envs, *self.obs_dims['rgb']),
                device=self.device
            )  # Placeholder
        
        # Depth image
        if self.cfg['env']['observation']['depth_camera']['enabled']:
            obs['depth'] = torch.zeros(
                (self.num_envs, *self.obs_dims['depth']),
                device=self.device
            )  # Placeholder
        
        # LiDAR scan
        if self.cfg['env']['observation']['lidar']['enabled']:
            # obs['lidar'] = self.lidar.get_scan()
            obs['lidar'] = torch.zeros(
                (self.num_envs, *self.obs_dims['lidar']),
                device=self.device
            )  # Placeholder
        
        # Robot proprioceptive state
        obs['robot_state'] = self._get_robot_state()
        
        # Task information
        obs['task_info'] = self._get_task_info()
        
        return obs
    
    def _get_robot_state(self) -> torch.Tensor:
        """Get robot proprioceptive state"""
        # Get from Isaac Lab articulation
        # state = self.robot.get_state()
        
        # Placeholder: [vel_x, vel_y, vel_z, ang_vel_x, ang_vel_y, ang_vel_z, 
        #                quat_x, quat_y, quat_z, quat_w, wheel_vel_1, wheel_vel_2]
        state = torch.zeros((self.num_envs, self.obs_dims['robot_state']), device=self.device)
        return state
    
    def _get_task_info(self) -> torch.Tensor:
        """Get task-related information (goal distance, direction, etc.)"""
        # Compute distance to next waypoint
        # Compute relative goal position in robot frame
        
        # Placeholder: [distance_to_goal, goal_x, goal_y]
        task_info = torch.zeros((self.num_envs, self.obs_dims['task_info']), device=self.device)
        return task_info
    
    def _compute_rewards(self, obs: Dict[str, torch.Tensor], actions: torch.Tensor) -> torch.Tensor:
        """
        Compute reward for each environment.
        
        Reward components:
        1. Progress reward (moving towards goal)
        2. Path tracking reward (staying on path)
        3. Speed tracking reward
        4. Collision penalty
        5. Off-road penalty
        6. Action smoothness reward
        
        Args:
            obs: Current observations
            actions: Applied actions
            
        Returns:
            rewards: (num_envs,) reward tensor
        """
        cfg_reward = self.cfg['env']['reward']
        
        rewards = torch.zeros(self.num_envs, device=self.device)
        
        # 1. Progress reward
        progress = self._compute_progress()
        rewards += progress * cfg_reward['progress_weight']
        
        # 2. Path tracking reward
        path_error = self._compute_path_tracking_error()
        rewards -= path_error * cfg_reward['path_tracking_weight']
        
        # 3. Speed tracking reward
        speed_error = self._compute_speed_error(obs['robot_state'])
        rewards -= speed_error * cfg_reward['speed_tracking_weight']
        
        # 4. Collision penalty
        collisions = self._check_collisions()
        rewards += collisions.float() * cfg_reward['collision_penalty']
        
        # 5. Off-road penalty
        off_road = self._check_off_road()
        rewards += off_road.float() * cfg_reward['off_road_penalty']
        
        # 6. Action smoothness (penalize large changes)
        if hasattr(self, 'prev_actions'):
            action_diff = torch.norm(actions - self.prev_actions, dim=1)
            rewards -= action_diff * cfg_reward['action_smoothness_weight']
        self.prev_actions = actions.clone()
        
        # 7. Goal reached bonus
        goal_reached = self._check_goal_reached()
        rewards += goal_reached.float() * cfg_reward['goal_reached_bonus']
        
        return rewards
    
    def _compute_progress(self) -> torch.Tensor:
        """
        Compute progress along reference path.
        
        Returns:
            progress: (N,) progress metric [0, 1], 1 = reached end
        """
        # Find closest point on path to robot position
        robot_xy = self.robot_positions[:, :2]  # (N, 2)
        path_xy = self.reference_path[:, :, :2]  # (N, num_points, 2)
        
        # Compute distances to all path points
        # (N, num_points)
        dists = torch.norm(path_xy - robot_xy.unsqueeze(1), dim=2)
        
        # Find closest point index
        closest_idx = torch.argmin(dists, dim=1)  # (N,)
        
        # Progress is ratio of closest point to path length
        num_points = path_xy.shape[1]
        progress = closest_idx.float() / float(num_points - 1)
        
        return progress
    
    def _compute_path_tracking_error(self) -> torch.Tensor:
        """
        Compute lateral error from desired path.
        
        Returns:
            error: (N,) distance to nearest path point (meters)
        """
        # Find minimum distance to path
        robot_xy = self.robot_positions[:, :2]  # (N, 2)
        path_xy = self.reference_path[:, :, :2]  # (N, num_points, 2)
        
        # Compute distances to all path points
        dists = torch.norm(path_xy - robot_xy.unsqueeze(1), dim=2)  # (N, num_points)
        
        # Return minimum distance
        min_dist, _ = torch.min(dists, dim=1)
        
        return min_dist
    
    def _compute_speed_error(self, robot_state: torch.Tensor) -> torch.Tensor:
        """Compute error from target speed"""
        current_speed = torch.norm(robot_state[:, :2], dim=1)  # Linear speed
        target_speed = self.cfg['env']['reward']['target_speed']
        return (current_speed - target_speed) ** 2
    
    def _check_collisions(self) -> torch.Tensor:
        """
        Check if robot collided with obstacles.
        
        Returns:
            collisions: (N,) boolean tensor, True if collision detected
        """
        # TODO: When Isaac Lab is integrated, use contact sensors:
        # contacts = self.contact_sensor.get_data()
        # return contacts.net_forces_w.norm(dim=-1) > collision_threshold
        
        # For now, assume no collisions (environment is empty)
        return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
    
    def _check_off_road(self) -> torch.Tensor:
        """
        Check if robot is off the designated path.
        
        Returns:
            off_road: (N,) boolean tensor, True if too far from path
        """
        # Define maximum allowed distance from path
        max_distance = self.cfg['env']['termination'].get('off_road_distance', 0.5)
        
        # Compute path tracking error
        path_error = self._compute_path_tracking_error()
        
        # Check if exceeds threshold
        return path_error > max_distance
    
    def _check_goal_reached(self) -> torch.Tensor:
        """
        Check if robot reached the goal.
        
        Returns:
            goal_reached: (N,) boolean tensor, True if goal is reached
        """
        # Goal is the last point on the reference path
        goal_pos = self.reference_path[:, -1, :2]  # (N, 2)
        robot_pos = self.robot_positions[:, :2]  # (N, 2)
        
        # Compute distance to goal
        dist_to_goal = torch.norm(goal_pos - robot_pos, dim=1)
        
        # Check if within threshold
        goal_threshold = self.cfg['env']['termination'].get('goal_reached_distance', 0.2)
        return dist_to_goal < goal_threshold
    
    def _check_termination(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Check termination conditions for each environment.
        
        Termination reasons:
        1. Collision
        2. Off-road (too far from path)
        3. Goal reached
        4. Max episode length
        
        Returns:
            dones: (num_envs,) boolean tensor
        """
        dones = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        
        # Collision
        if self.cfg['env']['termination']['collision']:
            dones |= self._check_collisions()
        
        # Off-road
        off_road_dist = self.cfg['env']['termination'].get('off_road_distance', 0.5)
        path_error = self._compute_path_tracking_error()
        dones |= (path_error > off_road_dist)
        
        # Goal reached
        goal_dist = self.cfg['env']['termination'].get('goal_reached_distance', 0.2)
        to_goal = torch.norm(obs['task_info'][:, 1:], dim=1)
        dones |= (to_goal < goal_dist)
        
        # Max episode length
        if self.cfg['env']['termination']['max_episode_length']:
            dones |= (self.episode_length_buf >= self.max_episode_length)
        
        return dones
    
    def _get_infos(self) -> Dict:
        """Get additional information for logging"""
        infos = {
            'episode_length': self.episode_length_buf.float().mean().item(),
            'episode_reward': self.episode_rewards.mean().item(),
        }
        return infos
    
    def close(self):
        """Clean up resources"""
        # TODO: Close Isaac Lab simulation
        print("[ROSOrinDrivingEnv] Closing environment...")
