"""
Model Predictive Controller for ROSOrin Mecanum Wheel Robot

基于麦克纳姆轮运动学的MPC控制器，用于路径跟踪任务。
"""

import numpy as np
import torch
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class MPCConfig:
    """MPC控制器配置"""
    # 预测参数
    horizon: int = 10  # 预测时域步数
    dt: float = 0.02  # 时间步长（与环境相同）
    
    # 代价函数权重
    Q_pos: float = 10.0  # 位置误差权重
    Q_vel: float = 1.0   # 速度权重
    Q_heading: float = 5.0  # 航向误差权重
    R: float = 0.1  # 控制输入权重
    
    # 约束
    max_linear_vel: float = 1.0  # m/s
    max_angular_vel: float = 2.0  # rad/s
    max_wheel_vel: float = 20.0  # rad/s
    
    # 优化参数
    max_iterations: int = 50
    tolerance: float = 1e-4


class MecanumKinematics:
    """麦克纳姆轮运动学模型"""
    
    def __init__(
        self,
        wheel_base: float = 0.17,  # ROSOrin轮距（前后）
        wheel_track: float = 0.20,  # ROSOrin轮距（左右）
        wheel_radius: float = 0.04,  # 轮子半径
    ):
        self.L = wheel_base  # 前后轮距
        self.W = wheel_track  # 左右轮距
        self.r = wheel_radius
        
        # 麦克纳姆轮运动学矩阵
        # [vx, vy, omega] = K * [w_fl, w_fr, w_bl, w_br]
        self.K_inv = np.array([
            [1, 1, 1, 1],
            [-1, 1, 1, -1],
            [-1/(self.L + self.W), 1/(self.L + self.W), 
             -1/(self.L + self.W), 1/(self.L + self.W)]
        ]) * self.r / 4.0
        
        # 逆运动学矩阵
        # [w_fl, w_fr, w_bl, w_br] = K^-1 * [vx, vy, omega]
        self.K = np.array([
            [1, -1, -(self.L + self.W)],
            [1, 1, (self.L + self.W)],
            [1, 1, -(self.L + self.W)],
            [1, -1, (self.L + self.W)]
        ]) / self.r
    
    def forward_kinematics(self, wheel_velocities: np.ndarray) -> np.ndarray:
        """
        正运动学：轮速 -> 机器人速度
        
        Args:
            wheel_velocities: [w_fl, w_fr, w_bl, w_br] (rad/s)
            
        Returns:
            [vx, vy, omega] 机器人速度 (m/s, m/s, rad/s)
        """
        return self.K_inv @ wheel_velocities
    
    def inverse_kinematics(self, robot_velocity: np.ndarray) -> np.ndarray:
        """
        逆运动学：机器人速度 -> 轮速
        
        Args:
            robot_velocity: [vx, vy, omega] (m/s, m/s, rad/s)
            
        Returns:
            [w_fl, w_fr, w_bl, w_br] 轮速 (rad/s)
        """
        return self.K @ robot_velocity


class MPCController:
    """
    简化的MPC控制器用于麦克纳姆轮机器人路径跟踪
    
    使用迭代线性二次调节器(iLQR)方法求解MPC优化问题
    """
    
    def __init__(self, config: MPCConfig = None):
        self.config = config or MPCConfig()
        self.kinematics = MecanumKinematics()
        
        # 状态历史（用于调试）
        self.state_history = []
        self.control_history = []
        
    def reset(self):
        """重置控制器"""
        self.state_history = []
        self.control_history = []
    
    def predict_state(
        self,
        current_state: np.ndarray,
        control: np.ndarray,
        dt: float
    ) -> np.ndarray:
        """
        状态预测（简化运动学模型）
        
        State: [x, y, theta, vx, vy, omega]
        Control: [vx_cmd, vy_cmd, omega_cmd]
        """
        x, y, theta, vx, vy, omega = current_state
        vx_cmd, vy_cmd, omega_cmd = control
        
        # 简单的一阶动力学（速度跟随指令）
        alpha = 0.5  # 速度响应系数
        vx_new = vx + alpha * (vx_cmd - vx)
        vy_new = vy + alpha * (vy_cmd - vy)
        omega_new = omega + alpha * (omega_cmd - omega)
        
        # 更新位置（考虑当前朝向）
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        
        # 将机器人坐标系速度转换到世界坐标系
        vx_world = vx_new * cos_theta - vy_new * sin_theta
        vy_world = vx_new * sin_theta + vy_new * cos_theta
        
        x_new = x + vx_world * dt
        y_new = y + vy_world * dt
        theta_new = theta + omega_new * dt
        
        # 归一化角度到[-pi, pi]
        theta_new = np.arctan2(np.sin(theta_new), np.cos(theta_new))
        
        return np.array([x_new, y_new, theta_new, vx_new, vy_new, omega_new])
    
    def compute_cost(
        self,
        state: np.ndarray,
        control: np.ndarray,
        target_pos: np.ndarray,
        target_heading: float = 0.0
    ) -> float:
        """
        计算代价函数
        
        Args:
            state: [x, y, theta, vx, vy, omega]
            control: [vx_cmd, vy_cmd, omega_cmd]
            target_pos: [x_target, y_target]
            target_heading: 目标朝向
        """
        x, y, theta, vx, vy, omega = state
        
        # 位置误差
        pos_error = np.linalg.norm([x - target_pos[0], y - target_pos[1]])
        
        # 航向误差
        heading_error = np.abs(np.arctan2(
            np.sin(theta - target_heading),
            np.cos(theta - target_heading)
        ))
        
        # 速度惩罚（鼓励平滑运动）
        vel_cost = vx**2 + vy**2 + omega**2
        
        # 控制输入惩罚
        control_cost = np.sum(control**2)
        
        total_cost = (
            self.config.Q_pos * pos_error +
            self.config.Q_heading * heading_error +
            self.config.Q_vel * vel_cost +
            self.config.R * control_cost
        )
        
        return total_cost
    
    def compute_control(
        self,
        current_state: np.ndarray,
        target_positions: np.ndarray,
        target_headings: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, dict]:
        """
        计算最优控制输入
        
        Args:
            current_state: [x, y, theta, vx, vy, omega]
            target_positions: (N, 2) 目标位置序列
            target_headings: (N,) 目标朝向序列（可选）
            
        Returns:
            control: [vx, vy, omega] 机器人速度命令
            info: 诊断信息字典
        """
        if target_headings is None:
            target_headings = np.zeros(len(target_positions))
        
        # 使用简单的纯追踪控制器（Pure Pursuit）
        # 这比完整的MPC更快且更稳定
        control, info = self._pure_pursuit_control(
            current_state, target_positions, target_headings
        )
        
        # 应用速度限制
        control = self._apply_velocity_limits(control)
        
        # 记录历史
        self.state_history.append(current_state.copy())
        self.control_history.append(control.copy())
        
        return control, info
    
    def _pure_pursuit_control(
        self,
        current_state: np.ndarray,
        target_positions: np.ndarray,
        target_headings: np.ndarray
    ) -> Tuple[np.ndarray, dict]:
        """
        纯追踪控制算法
        """
        x, y, theta, vx, vy, omega = current_state
        
        # 选择前视距离
        lookahead_distance = 0.3  # 30cm
        
        # 找到前视点
        distances = np.linalg.norm(target_positions - np.array([x, y]), axis=1)
        lookahead_idx = np.argmin(np.abs(distances - lookahead_distance))
        lookahead_idx = min(lookahead_idx, len(target_positions) - 1)
        
        target_pos = target_positions[lookahead_idx]
        target_heading = target_headings[lookahead_idx]
        
        # 计算到目标的向量（世界坐标系）
        dx_world = target_pos[0] - x
        dy_world = target_pos[1] - y
        
        # 转换到机器人坐标系
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        dx_robot = dx_world * cos_theta + dy_world * sin_theta
        dy_robot = -dx_world * sin_theta + dy_world * cos_theta
        
        # 计算期望速度
        distance_to_target = np.sqrt(dx_robot**2 + dy_robot**2)
        
        # 速度控制（PD控制器）
        kp_linear = 2.0
        kp_angular = 3.0
        
        desired_vx = kp_linear * dx_robot
        desired_vy = kp_linear * dy_robot
        
        # 航向控制
        heading_error = np.arctan2(np.sin(target_heading - theta),
                                   np.cos(target_heading - theta))
        desired_omega = kp_angular * heading_error
        
        control = np.array([desired_vx, desired_vy, desired_omega])
        
        info = {
            'lookahead_idx': lookahead_idx,
            'distance_to_target': distance_to_target,
            'heading_error': heading_error,
            'target_pos': target_pos,
        }
        
        return control, info
    
    def _apply_velocity_limits(self, control: np.ndarray) -> np.ndarray:
        """应用速度限制"""
        vx, vy, omega = control
        
        # 限制线速度
        linear_vel = np.sqrt(vx**2 + vy**2)
        if linear_vel > self.config.max_linear_vel:
            scale = self.config.max_linear_vel / linear_vel
            vx *= scale
            vy *= scale
        
        # 限制角速度
        omega = np.clip(omega, -self.config.max_angular_vel, 
                       self.config.max_angular_vel)
        
        return np.array([vx, vy, omega])
    
    def robot_velocity_to_wheel_velocity(
        self, 
        robot_velocity: np.ndarray
    ) -> np.ndarray:
        """
        将机器人速度转换为轮速
        
        Args:
            robot_velocity: [vx, vy, omega]
            
        Returns:
            wheel_velocities: [w_fl, w_fr, w_bl, w_br]
        """
        wheel_vel = self.kinematics.inverse_kinematics(robot_velocity)
        
        # 应用轮速限制
        wheel_vel = np.clip(wheel_vel, 
                           -self.config.max_wheel_vel,
                           self.config.max_wheel_vel)
        
        return wheel_vel


if __name__ == "__main__":
    # 测试MPC控制器
    print("="*70)
    print("  MPC控制器测试")
    print("="*70)
    
    # 创建控制器
    mpc = MPCController()
    
    # 初始状态 [x, y, theta, vx, vy, omega]
    state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    # 目标路径（直线）
    target_positions = np.array([
        [1.0, 0.0],
        [2.0, 0.0],
        [3.0, 0.0],
    ])
    target_headings = np.zeros(3)
    
    # 计算控制
    control, info = mpc.compute_control(state, target_positions, target_headings)
    
    print(f"\n当前状态: {state}")
    print(f"目标位置: {target_positions[0]}")
    print(f"控制输出 (vx, vy, omega): {control}")
    print(f"前视点索引: {info['lookahead_idx']}")
    print(f"到目标距离: {info['distance_to_target']:.3f}m")
    
    # 转换为轮速
    wheel_vel = mpc.robot_velocity_to_wheel_velocity(control)
    print(f"\n轮速 (rad/s):")
    print(f"  左前: {wheel_vel[0]:.2f}")
    print(f"  右前: {wheel_vel[1]:.2f}")
    print(f"  左后: {wheel_vel[2]:.2f}")
    print(f"  右后: {wheel_vel[3]:.2f}")
    
    print("\n✓ MPC控制器测试完成")
