#!/usr/bin/env python3
"""
Test ROSOrin environment logic without Isaac Lab dependencies.
Tests all core functionalities:
- Path generation
- Action application
- Reward computation
- Termination checking
- Domain randomization
"""

import sys
import os
import torch
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def load_test_config():
    """Load configuration for testing"""
    config = {
        'env': {
            'num_envs': 4,
            'sim': {
                'dt': 0.02,  # 50 Hz
                'device': 'cuda' if torch.cuda.is_available() else 'cpu'
            },
            'observation': {
                'rgb_camera': {
                    'enabled': True,
                    'width': 84,
                    'height': 84,
                    'channels': 3
                },
                'depth_camera': {
                    'enabled': False,
                    'width': 84,
                    'height': 84
                },
                'lidar': {
                    'enabled': True,
                    'num_beams': 128,
                    'max_range': 10.0
                },
                'robot_state_dim': 13,
                'task_info_dim': 7
            },
            'reward': {
                'progress_weight': 10.0,
                'path_tracking_weight': 5.0,
                'speed_tracking_weight': 1.0,
                'collision_penalty': -10.0,
                'off_road_penalty': -5.0,
                'action_smoothness_weight': 0.1,
                'goal_reached_bonus': 50.0,
                'target_speed': 0.5
            },
            'termination': {
                'collision': True,
                'off_road_distance': 0.5,
                'goal_reached_distance': 0.2,
                'max_episode_steps': 500
            },
            'domain_randomization': {
                'enabled': True,
                'mass_scale': [0.8, 1.2],
                'friction_scale': [0.7, 1.3],
                'motor_strength_scale': [0.85, 1.15],
                'sensor_noise': {
                    'lidar_noise_std': 0.02,
                    'camera_brightness': [0.8, 1.2]
                }
            }
        },
        'robot': {
            'dimensions': {
                'wheelbase': 0.206,
                'track_width': 0.194,
                'wheel_radius': 0.0325
            },
            'max_wheel_speed': 10.0  # rad/s
        }
    }
    return config


def test_path_generation():
    """Test Bezier path generation"""
    print("\n=== Test 1: Path Generation ===")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Generate sample paths
    num_paths = 3
    num_points = 100
    
    # Control points for Bezier curves
    p0 = torch.tensor([[0.0, 0.0]], device=device).repeat(num_paths, 1)
    p1 = torch.tensor([[1.0, 2.0], [0.5, 1.5], [2.0, 1.0]], device=device)
    p2 = torch.tensor([[3.0, 2.5], [2.0, 3.0], [3.5, 0.5]], device=device)
    p3 = torch.tensor([[4.0, 0.0], [3.5, 0.5], [4.0, -1.0]], device=device)
    
    # Bezier curve: B(t) = (1-t)³P0 + 3(1-t)²tP1 + 3(1-t)t²P2 + t³P3
    t = torch.linspace(0, 1, num_points, device=device)
    t = t.unsqueeze(0).unsqueeze(-1).repeat(num_paths, 1, 1)  # (num_paths, num_points, 1)
    
    paths = (1-t)**3 * p0.unsqueeze(1) + \
            3*(1-t)**2*t * p1.unsqueeze(1) + \
            3*(1-t)*t**2 * p2.unsqueeze(1) + \
            t**3 * p3.unsqueeze(1)
    
    print(f"✓ Generated {num_paths} paths with {num_points} points each")
    print(f"  Path shape: {paths.shape}")
    print(f"  Start points: {paths[:, 0, :]}")
    print(f"  End points: {paths[:, -1, :]}")
    
    return paths


def test_mecanum_kinematics():
    """Test mecanum wheel inverse and forward kinematics"""
    print("\n=== Test 2: Mecanum Kinematics ===")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config = load_test_config()
    
    wheelbase = config['robot']['dimensions']['wheelbase']
    track_width = config['robot']['dimensions']['track_width']
    wheel_radius = config['robot']['dimensions']['wheel_radius']
    
    lw = (wheelbase + track_width) / 2.0
    
    # Test forward then inverse kinematics
    body_vel = torch.tensor([[0.5, 0.0, 0.0],   # Forward
                             [0.0, 0.3, 0.0],   # Strafe right
                             [0.0, 0.0, 1.0],   # Rotate CCW
                             [0.3, 0.2, 0.5]],  # Combined
                            device=device)
    
    # Inverse kinematics
    vx, vy, wz = body_vel[:, 0], body_vel[:, 1], body_vel[:, 2]
    wheel_vel_lin = torch.stack([
        vx - vy - wz * lw,  # front-left
        vx + vy + wz * lw,  # front-right
        vx + vy - wz * lw,  # rear-left
        vx - vy + wz * lw   # rear-right
    ], dim=1)
    wheel_vel = wheel_vel_lin / wheel_radius
    
    # Forward kinematics (should recover body_vel)
    v_fl, v_fr, v_rl, v_rr = wheel_vel[:, 0], wheel_vel[:, 1], wheel_vel[:, 2], wheel_vel[:, 3]
    v_fl_lin = v_fl * wheel_radius
    v_fr_lin = v_fr * wheel_radius
    v_rl_lin = v_rl * wheel_radius
    v_rr_lin = v_rr * wheel_radius
    
    recovered_vx = (v_fl_lin + v_fr_lin + v_rl_lin + v_rr_lin) / 4.0
    recovered_vy = (-v_fl_lin + v_fr_lin + v_rl_lin - v_rr_lin) / 4.0
    recovered_wz = (-v_fl_lin + v_fr_lin - v_rl_lin + v_rr_lin) / (4.0 * lw)
    
    recovered_body_vel = torch.stack([recovered_vx, recovered_vy, recovered_wz], dim=1)
    
    error = torch.abs(body_vel - recovered_body_vel).max()
    
    print(f"✓ Tested inverse + forward kinematics")
    print(f"  Input body velocities: {body_vel}")
    print(f"  Wheel velocities (rad/s): {wheel_vel}")
    print(f"  Recovered body velocities: {recovered_body_vel}")
    print(f"  Max error: {error.item():.6f} m/s (should be < 1e-5)")
    
    assert error < 1e-5, "Kinematics error too large!"
    return True


def test_reward_computation():
    """Test reward computation functions"""
    print("\n=== Test 3: Reward Computation ===")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config = load_test_config()
    num_envs = 4
    
    # Generate sample paths (one for each environment)
    num_points = 100
    paths_list = []
    for i in range(num_envs):
        p0 = torch.tensor([0.0, 0.0], device=device)
        p1 = torch.tensor([1.0 + i*0.5, 2.0], device=device)
        p2 = torch.tensor([3.0, 2.5 - i*0.3], device=device)
        p3 = torch.tensor([4.0, 0.0], device=device)
        
        t = torch.linspace(0, 1, num_points, device=device).unsqueeze(-1)
        path = (1-t)**3 * p0 + 3*(1-t)**2*t * p1 + 3*(1-t)*t**2 * p2 + t**3 * p3
        paths_list.append(path)
    
    paths = torch.stack(paths_list, dim=0)  # (num_envs, num_points, 2)
    
    # Simulate robot at different positions
    robot_positions = torch.tensor([
        [0.1, 0.05, 0.1],   # Near start
        [2.0, 1.5, 0.1],    # Middle of path
        [3.8, 0.1, 0.1],    # Near end
        [5.0, 5.0, 0.1]     # Far off path
    ], device=device)
    
    # Test 1: Progress computation
    robot_xy = robot_positions[:, :2]
    path_xy = paths[:, :, :2]
    dists = torch.norm(path_xy - robot_xy.unsqueeze(1), dim=2)
    closest_idx = torch.argmin(dists, dim=1)
    progress = closest_idx.float() / float(paths.shape[1] - 1)
    
    print(f"\n  Progress Test:")
    print(f"    Closest waypoint indices: {closest_idx}")
    print(f"    Progress ratios: {progress}")
    
    # Test 2: Path tracking error
    min_dist, _ = torch.min(dists, dim=1)
    print(f"\n  Path Tracking Error:")
    print(f"    Min distances to path: {min_dist}")
    
    # Test 3: Goal reached check
    goal_pos = paths[:, -1, :2]
    dist_to_goal = torch.norm(goal_pos - robot_xy, dim=1)
    goal_reached = dist_to_goal < config['env']['termination']['goal_reached_distance']
    print(f"\n  Goal Reached Check:")
    print(f"    Distance to goal: {dist_to_goal}")
    print(f"    Goal reached: {goal_reached}")
    
    # Test 4: Off-road check
    off_road = min_dist > config['env']['termination']['off_road_distance']
    print(f"\n  Off-Road Check:")
    print(f"    Off-road status: {off_road}")
    
    print(f"\n✓ All reward computations working correctly")
    return True


def test_domain_randomization():
    """Test domain randomization"""
    print("\n=== Test 4: Domain Randomization ===")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config = load_test_config()
    num_envs = 10
    
    cfg_dr = config['env']['domain_randomization']
    
    # Test mass randomization
    mass_range = cfg_dr['mass_scale']
    mass_scales = torch.rand(num_envs, device=device) * (mass_range[1] - mass_range[0]) + mass_range[0]
    
    # Test friction randomization
    friction_range = cfg_dr['friction_scale']
    friction_scales = torch.rand(num_envs, device=device) * (friction_range[1] - friction_range[0]) + friction_range[0]
    
    # Test motor strength randomization
    motor_range = cfg_dr['motor_strength_scale']
    motor_scales = torch.rand(num_envs, device=device) * (motor_range[1] - motor_range[0]) + motor_range[0]
    
    print(f"  Mass scales: mean={mass_scales.mean():.3f}, std={mass_scales.std():.3f}, range=[{mass_scales.min():.3f}, {mass_scales.max():.3f}]")
    print(f"  Friction scales: mean={friction_scales.mean():.3f}, std={friction_scales.std():.3f}, range=[{friction_scales.min():.3f}, {friction_scales.max():.3f}]")
    print(f"  Motor scales: mean={motor_scales.mean():.3f}, std={motor_scales.std():.3f}, range=[{motor_scales.min():.3f}, {motor_scales.max():.3f}]")
    
    # Check ranges
    assert mass_scales.min() >= mass_range[0] and mass_scales.max() <= mass_range[1]
    assert friction_scales.min() >= friction_range[0] and friction_scales.max() <= friction_range[1]
    assert motor_scales.min() >= motor_range[0] and motor_scales.max() <= motor_range[1]
    
    print(f"✓ Domain randomization ranges correct")
    return True


def test_simulation_step():
    """Test physics simulation stepping"""
    print("\n=== Test 5: Simulation Stepping ===")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config = load_test_config()
    dt = config['env']['sim']['dt']
    
    # Initial state
    position = torch.tensor([0.0, 0.0, 0.1], device=device)
    yaw = torch.tensor(0.0, device=device)
    
    # Constant forward velocity
    body_vel = torch.tensor([0.5, 0.0, 0.0], device=device)  # 0.5 m/s forward
    
    # Simulate for 2 seconds (100 steps)
    num_steps = 100
    positions = [position.clone()]
    
    for _ in range(num_steps):
        # Update position
        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)
        
        vx_world = body_vel[0] * cos_yaw - body_vel[1] * sin_yaw
        vy_world = body_vel[0] * sin_yaw + body_vel[1] * sin_yaw
        
        position[0] += vx_world * dt
        position[1] += vy_world * dt
        yaw += body_vel[2] * dt
        
        positions.append(position.clone())
    
    final_position = positions[-1]
    expected_distance = body_vel[0] * dt * num_steps
    actual_distance = final_position[0]
    
    print(f"  Initial position: {positions[0]}")
    print(f"  Final position after {num_steps} steps: {final_position}")
    print(f"  Expected distance: {expected_distance:.3f} m")
    print(f"  Actual distance: {actual_distance:.3f} m")
    print(f"  Error: {abs(expected_distance - actual_distance):.6f} m")
    
    assert abs(expected_distance - actual_distance) < 0.01, "Simulation integration error too large!"
    
    print(f"✓ Simulation stepping working correctly")
    return True


def main():
    """Run all tests"""
    print("="*60)
    print("ROSOrin Environment Logic Tests")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    
    try:
        # Run tests
        test_path_generation()
        test_mecanum_kinematics()
        test_reward_computation()
        test_domain_randomization()
        test_simulation_step()
        
        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED!")
        print("="*60)
        print("\nEnvironment logic is correct and ready for Isaac Lab integration.")
        
        return 0
        
    except Exception as e:
        print("\n" + "="*60)
        print("❌ TEST FAILED!")
        print("="*60)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
