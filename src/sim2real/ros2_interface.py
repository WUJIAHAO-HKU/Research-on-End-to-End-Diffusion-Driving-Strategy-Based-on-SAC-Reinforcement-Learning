"""
ROS2 Interface for Real Robot Deployment

Bridges trained policy with ROS2 topics/services on ROSOrin car.
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from sensor_msgs.msg import Image, PointCloud2, Imu
from nav_msgs.msg import Odometry
import torch
import numpy as np
from cv_bridge import CvBridge
import sensor_msgs_py.point_cloud2 as pc2
from typing import Dict, Optional
import threading


class PolicyNode(Node):
    """
    ROS2 node that subscribes to sensors and publishes control commands.
    """
    
    def __init__(
        self,
        agent,
        config: Dict,
        control_frequency: float = 10.0,
    ):
        super().__init__('policy_node')
        
        self.agent = agent
        self.config = config
        self.control_dt = 1.0 / control_frequency
        
        # CV Bridge for image conversion
        self.bridge = CvBridge()
        
        # Latest observations
        self.latest_obs = {
            'vision': None,
            'lidar': None,
            'proprio': None,
            'timestamp': None,
        }
        self.obs_lock = threading.Lock()
        
        # Subscribers
        self._create_subscribers()
        
        # Publishers
        self.cmd_vel_pub = self.create_publisher(
            Twist, '/cmd_vel', 10
        )
        
        # Control timer
        self.control_timer = self.create_timer(
            self.control_dt,
            self.control_callback
        )
        
        self.get_logger().info('Policy node initialized')
    
    def _create_subscribers(self):
        """Create ROS2 subscribers for sensors"""
        # RGB-D camera
        self.rgb_sub = self.create_subscription(
            Image,
            '/camera/color/image_raw',
            self.rgb_callback,
            10
        )
        
        self.depth_sub = self.create_subscription(
            Image,
            '/camera/depth/image_raw',
            self.depth_callback,
            10
        )
        
        # LiDAR
        self.lidar_sub = self.create_subscription(
            PointCloud2,
            '/scan_cloud',
            self.lidar_callback,
            10
        )
        
        # IMU
        self.imu_sub = self.create_subscription(
            Imu,
            '/imu/data',
            self.imu_callback,
            10
        )
        
        # Odometry
        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )
    
    def rgb_callback(self, msg: Image):
        """Handle RGB image"""
        try:
            # Convert ROS image to numpy
            rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
            rgb = rgb.astype(np.float32) / 255.0
            
            with self.obs_lock:
                if self.latest_obs['vision'] is None:
                    self.latest_obs['vision'] = np.zeros((4, *rgb.shape[:2]), dtype=np.float32)
                
                self.latest_obs['vision'][:3] = rgb.transpose(2, 0, 1)
                
        except Exception as e:
            self.get_logger().error(f'RGB callback error: {e}')
    
    def depth_callback(self, msg: Image):
        """Handle depth image"""
        try:
            # Convert depth image
            depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            depth = depth.astype(np.float32) / 1000.0  # mm to meters
            
            with self.obs_lock:
                if self.latest_obs['vision'] is None:
                    h, w = depth.shape
                    self.latest_obs['vision'] = np.zeros((4, h, w), dtype=np.float32)
                
                self.latest_obs['vision'][3] = depth
                
        except Exception as e:
            self.get_logger().error(f'Depth callback error: {e}')
    
    def lidar_callback(self, msg: PointCloud2):
        """Handle LiDAR point cloud"""
        try:
            # Extract points
            points_list = []
            for point in pc2.read_points(msg, field_names=('x', 'y', 'z'), skip_nans=True):
                points_list.append([point[0], point[1], point[2]])
            
            if len(points_list) > 0:
                points = np.array(points_list, dtype=np.float32)
                
                with self.obs_lock:
                    self.latest_obs['lidar'] = points
            
        except Exception as e:
            self.get_logger().error(f'LiDAR callback error: {e}')
    
    def imu_callback(self, msg: Imu):
        """Handle IMU data"""
        try:
            with self.obs_lock:
                if self.latest_obs['proprio'] is None:
                    self.latest_obs['proprio'] = np.zeros(8, dtype=np.float32)
                
                # Angular velocity
                self.latest_obs['proprio'][5:8] = [
                    msg.angular_velocity.x,
                    msg.angular_velocity.y,
                    msg.angular_velocity.z,
                ]
                
        except Exception as e:
            self.get_logger().error(f'IMU callback error: {e}')
    
    def odom_callback(self, msg: Odometry):
        """Handle odometry"""
        try:
            with self.obs_lock:
                if self.latest_obs['proprio'] is None:
                    self.latest_obs['proprio'] = np.zeros(8, dtype=np.float32)
                
                # Position (x, y)
                self.latest_obs['proprio'][0] = msg.pose.pose.position.x
                self.latest_obs['proprio'][1] = msg.pose.pose.position.y
                
                # Orientation (yaw)
                quat = msg.pose.pose.orientation
                yaw = np.arctan2(
                    2.0 * (quat.w * quat.z + quat.x * quat.y),
                    1.0 - 2.0 * (quat.y**2 + quat.z**2)
                )
                self.latest_obs['proprio'][2] = yaw
                
                # Linear velocity
                self.latest_obs['proprio'][3] = msg.twist.twist.linear.x
                self.latest_obs['proprio'][4] = msg.twist.twist.linear.y
                
        except Exception as e:
            self.get_logger().error(f'Odom callback error: {e}')
    
    def control_callback(self):
        """Main control loop"""
        try:
            # Get observations
            with self.obs_lock:
                obs = self.latest_obs.copy()
            
            # Check if observations are ready
            if any(v is None for v in obs.values() if v != obs['timestamp']):
                self.get_logger().warn('Waiting for sensor data...')
                return
            
            # Get action from policy
            action = self.agent.select_action(obs, eval_mode=True)
            
            # Publish control command
            self.publish_command(action)
            
        except Exception as e:
            self.get_logger().error(f'Control callback error: {e}')
            # Publish zero velocity for safety
            self.publish_command(np.zeros(3))
    
    def publish_command(self, action: np.ndarray):
        """
        Publish control command.
        
        Args:
            action: [linear_x, linear_y, angular_z]
        """
        cmd = Twist()
        cmd.linear.x = float(action[0])
        cmd.linear.y = float(action[1])
        cmd.angular.z = float(action[2])
        
        self.cmd_vel_pub.publish(cmd)
    
    def stop(self):
        """Send stop command"""
        self.publish_command(np.zeros(3))
        self.get_logger().info('Stop command sent')


class GoalPublisher(Node):
    """
    Node for publishing goal positions.
    """
    
    def __init__(self):
        super().__init__('goal_publisher')
        
        self.goal_pub = self.create_publisher(
            PoseStamped,
            '/goal_pose',
            10
        )
        
        self.get_logger().info('Goal publisher initialized')
    
    def publish_goal(self, x: float, y: float, theta: float = 0.0):
        """Publish goal pose"""
        goal = PoseStamped()
        goal.header.frame_id = 'map'
        goal.header.stamp = self.get_clock().now().to_msg()
        
        goal.pose.position.x = x
        goal.pose.position.y = y
        goal.pose.position.z = 0.0
        
        # Convert yaw to quaternion
        goal.pose.orientation.w = np.cos(theta / 2.0)
        goal.pose.orientation.z = np.sin(theta / 2.0)
        
        self.goal_pub.publish(goal)
        self.get_logger().info(f'Published goal: ({x}, {y}, {theta})')


def main():
    """Main function for ROS2 node"""
    rclpy.init()
    
    # Load agent (placeholder - implement actual loading)
    from src.models.sac.sac_agent import SACAgent
    agent = SACAgent(...)  # Load from checkpoint
    
    # Create node
    policy_node = PolicyNode(agent, config={})
    
    try:
        rclpy.spin(policy_node)
    except KeyboardInterrupt:
        policy_node.stop()
    finally:
        policy_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
