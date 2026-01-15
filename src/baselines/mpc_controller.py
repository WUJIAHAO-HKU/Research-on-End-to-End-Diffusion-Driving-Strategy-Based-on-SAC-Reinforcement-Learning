"""
Model Predictive Control (MPC) Controller

Uses optimization to find optimal control sequence.
Serves as expert policy for demonstration collection.
"""

import numpy as np
from typing import Dict, Optional, Callable, Tuple
import cvxpy as cp
from scipy.optimize import minimize


class MPCController:
    """
    Model Predictive Control for mecanum wheel robot.
    
    Solves optimization problem:
        min sum_{t=0}^{H-1} [cost(x_t, u_t) + terminal_cost(x_H)]
        s.t. x_{t+1} = f(x_t, u_t)
             u_min <= u_t <= u_max
             constraints(x_t, u_t)
    """
    
    def __init__(
        self,
        horizon: int = 20,
        dt: float = 0.1,
        action_bounds: Tuple[np.ndarray, np.ndarray] = None,
        wheelbase: float = 0.206,
        track: float = 0.194,
        wheel_radius: float = 0.0325,
        obstacle_avoidance: bool = True,
        safety_margin: float = 0.3,
    ):
        """
        Args:
            horizon: Prediction horizon (steps)
            dt: Time step (seconds)
            action_bounds: (lower, upper) bounds for actions
            wheelbase: Distance between front and rear wheels (m)
            track: Distance between left and right wheels (m)
            wheel_radius: Wheel radius (m)
            obstacle_avoidance: Enable obstacle avoidance
            safety_margin: Safety margin for obstacles (m)
        """
        self.horizon = horizon
        self.dt = dt
        self.wheelbase = wheelbase
        self.track = track
        self.wheel_radius = wheel_radius
        self.obstacle_avoidance = obstacle_avoidance
        self.safety_margin = safety_margin
        
        # Action bounds
        if action_bounds is None:
            self.action_lower = np.array([-1.0, -1.0, -1.0])
            self.action_upper = np.array([1.0, 1.0, 1.0])
        else:
            self.action_lower, self.action_upper = action_bounds
        
        # Cost weights
        self.Q = np.diag([10.0, 10.0, 1.0])  # State cost (x, y, theta)
        self.R = np.diag([1.0, 1.0, 1.0])     # Control cost
        self.Q_terminal = np.diag([100.0, 100.0, 10.0])  # Terminal cost
        
        # Obstacles (updated each step)
        self.obstacles = []
    
    def update_obstacles(self, obstacles: list):
        """Update obstacle positions"""
        self.obstacles = obstacles
    
    def get_action(
        self,
        observation: Dict[str, np.ndarray],
        goal: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Compute optimal action using MPC.
        
        Args:
            observation: Current observation with 'proprio' (position, velocity, orientation)
            goal: Goal position [x, y, theta] (optional)
            
        Returns:
            action: [linear_x, linear_y, angular_z]
        """
        # Extract state
        proprio = observation.get('proprio', observation)
        
        # State: [x, y, theta, vx, vy, omega]
        if len(proprio) >= 6:
            state = proprio[:6]
        else:
            # If only position available, assume zero velocity
            state = np.zeros(6)
            state[:3] = proprio[:3]
        
        # Set goal
        if goal is None:
            goal = observation.get('goal', np.array([5.0, 0.0, 0.0]))
        
        # Solve MPC problem
        try:
            action_sequence = self._solve_mpc(state, goal)
            action = action_sequence[0]
        except Exception as e:
            print(f"MPC solver failed: {e}, using fallback controller")
            action = self._fallback_controller(state, goal)
        
        return action
    
    def _solve_mpc(
        self,
        initial_state: np.ndarray,
        goal: np.ndarray,
    ) -> np.ndarray:
        """
        Solve MPC optimization problem.
        
        Uses CVXPY for convex optimization (linearized dynamics).
        """
        # Decision variables
        x = cp.Variable((self.horizon + 1, 3))  # [x, y, theta]
        u = cp.Variable((self.horizon, 3))      # [vx, vy, omega]
        
        # Objective
        cost = 0
        
        for t in range(self.horizon):
            # State cost
            state_error = x[t] - goal
            cost += cp.quad_form(state_error, self.Q)
            
            # Control cost
            cost += cp.quad_form(u[t], self.R)
        
        # Terminal cost
        terminal_error = x[self.horizon] - goal
        cost += cp.quad_form(terminal_error, self.Q_terminal)
        
        # Constraints
        constraints = []
        
        # Initial state
        constraints.append(x[0, 0] == initial_state[0])
        constraints.append(x[0, 1] == initial_state[1])
        constraints.append(x[0, 2] == initial_state[2])
        
        # Dynamics (linearized)
        for t in range(self.horizon):
            # Simplified kinematics: x_{t+1} = x_t + v_t * dt
            theta_t = initial_state[2]  # Use current orientation (linearization)
            
            constraints.append(
                x[t+1, 0] == x[t, 0] + self.dt * (
                    u[t, 0] * cp.cos(theta_t) - u[t, 1] * cp.sin(theta_t)
                )
            )
            constraints.append(
                x[t+1, 1] == x[t, 1] + self.dt * (
                    u[t, 0] * cp.sin(theta_t) + u[t, 1] * cp.cos(theta_t)
                )
            )
            constraints.append(
                x[t+1, 2] == x[t, 2] + self.dt * u[t, 2]
            )
        
        # Control bounds
        for t in range(self.horizon):
            constraints.append(u[t] >= self.action_lower)
            constraints.append(u[t] <= self.action_upper)
        
        # Obstacle avoidance (linearized constraints)
        if self.obstacle_avoidance:
            for obs in self.obstacles:
                obs_pos = obs['position'][:2]
                obs_radius = obs['radius']
                
                for t in range(self.horizon):
                    # Distance constraint: ||x[t] - obs_pos|| >= obs_radius + safety_margin
                    # Linearized: (x[t] - obs_pos)^T * (x_ref - obs_pos) >= (obs_radius + safety)^2
                    dist = np.linalg.norm(initial_state[:2] - obs_pos)
                    if dist < obs_radius + self.safety_margin + 1.0:
                        direction = (initial_state[:2] - obs_pos) / (dist + 1e-6)
                        min_dist = obs_radius + self.safety_margin
                        constraints.append(
                            direction[0] * (x[t, 0] - obs_pos[0]) +
                            direction[1] * (x[t, 1] - obs_pos[1]) >= min_dist
                        )
        
        # Solve
        problem = cp.Problem(cp.Minimize(cost), constraints)
        problem.solve(solver=cp.OSQP, verbose=False)
        
        if problem.status not in ["optimal", "optimal_inaccurate"]:
            raise ValueError(f"MPC solver failed with status: {problem.status}")
        
        return u.value
    
    def _fallback_controller(
        self,
        state: np.ndarray,
        goal: np.ndarray,
    ) -> np.ndarray:
        """
        Simple proportional controller as fallback.
        """
        # Position error
        error = goal[:2] - state[:2]
        distance = np.linalg.norm(error)
        
        if distance < 0.1:
            return np.zeros(3)
        
        # Desired velocity direction
        desired_direction = error / distance
        
        # Current orientation
        theta = state[2]
        current_direction = np.array([np.cos(theta), np.sin(theta)])
        
        # Velocity in robot frame
        vx = 0.5 * np.dot(desired_direction, current_direction)
        vy = 0.5 * np.dot(desired_direction, np.array([-np.sin(theta), np.cos(theta)]))
        
        # Angular velocity (proportional to heading error)
        desired_theta = np.arctan2(error[1], error[0])
        theta_error = np.arctan2(np.sin(desired_theta - theta), np.cos(desired_theta - theta))
        omega = 2.0 * theta_error
        
        # Clip
        action = np.array([vx, vy, omega])
        action = np.clip(action, self.action_lower, self.action_upper)
        
        return action


class NonlinearMPCController(MPCController):
    """
    Nonlinear MPC using scipy.optimize.
    
    More accurate but slower than linearized version.
    """
    
    def _solve_mpc(
        self,
        initial_state: np.ndarray,
        goal: np.ndarray,
    ) -> np.ndarray:
        """
        Solve MPC using nonlinear optimization.
        """
        # Decision variables: flatten [u_0, ..., u_{H-1}]
        u_init = np.zeros(self.horizon * 3)
        
        # Define cost function
        def cost_function(u_flat):
            u = u_flat.reshape(self.horizon, 3)
            
            # Simulate trajectory
            state = initial_state[:3].copy()
            cost = 0
            
            for t in range(self.horizon):
                # State cost
                state_error = state - goal
                cost += state_error @ self.Q @ state_error
                
                # Control cost
                cost += u[t] @ self.R @ u[t]
                
                # Update state (nonlinear dynamics)
                theta = state[2]
                state[0] += self.dt * (u[t, 0] * np.cos(theta) - u[t, 1] * np.sin(theta))
                state[1] += self.dt * (u[t, 0] * np.sin(theta) + u[t, 1] * np.cos(theta))
                state[2] += self.dt * u[t, 2]
            
            # Terminal cost
            terminal_error = state - goal
            cost += terminal_error @ self.Q_terminal @ terminal_error
            
            # Obstacle avoidance penalty
            if self.obstacle_avoidance:
                state = initial_state[:3].copy()
                for t in range(self.horizon):
                    for obs in self.obstacles:
                        obs_pos = obs['position'][:2]
                        obs_radius = obs['radius']
                        
                        dist = np.linalg.norm(state[:2] - obs_pos)
                        margin = obs_radius + self.safety_margin
                        
                        if dist < margin:
                            cost += 1000 * (margin - dist) ** 2
                    
                    # Update state
                    theta = state[2]
                    u_t = u[t]
                    state[0] += self.dt * (u_t[0] * np.cos(theta) - u_t[1] * np.sin(theta))
                    state[1] += self.dt * (u_t[0] * np.sin(theta) + u_t[1] * np.cos(theta))
                    state[2] += self.dt * u_t[2]
            
            return cost
        
        # Bounds
        bounds = []
        for _ in range(self.horizon):
            for i in range(3):
                bounds.append((self.action_lower[i], self.action_upper[i]))
        
        # Optimize
        result = minimize(
            cost_function,
            u_init,
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': 100, 'ftol': 1e-4}
        )
        
        if not result.success:
            raise ValueError(f"MPC optimization failed: {result.message}")
        
        u_opt = result.x.reshape(self.horizon, 3)
        return u_opt


class AdaptiveMPCController(NonlinearMPCController):
    """
    Adaptive MPC that adjusts parameters based on performance.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Adaptive parameters
        self.min_horizon = 5
        self.max_horizon = 30
        self.success_count = 0
        self.total_count = 0
    
    def get_action(
        self,
        observation: Dict[str, np.ndarray],
        goal: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Get action with adaptive horizon.
        """
        # Adjust horizon based on success rate
        if self.total_count > 10:
            success_rate = self.success_count / self.total_count
            
            if success_rate > 0.8:
                # Increase horizon for better performance
                self.horizon = min(self.horizon + 1, self.max_horizon)
            elif success_rate < 0.5:
                # Decrease horizon for faster computation
                self.horizon = max(self.horizon - 1, self.min_horizon)
        
        return super().get_action(observation, goal)
    
    def update_performance(self, success: bool):
        """Update performance statistics"""
        self.total_count += 1
        if success:
            self.success_count += 1
