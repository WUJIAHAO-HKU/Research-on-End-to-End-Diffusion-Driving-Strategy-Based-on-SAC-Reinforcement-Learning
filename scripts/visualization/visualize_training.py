#!/usr/bin/env python3
"""
可视化测试脚本 - 观察ROSOrin机器人导航训练过程

在Isaac Sim GUI中可视化显示:
- 机器人运动
- 目标点位置
- 奖励实时变化
"""

import argparse
import torch
import time

# Isaac Lab
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="可视化ROSOrin导航训练")
parser.add_argument("--num_envs", type=int, default=2, help="并行环境数量(建议1-4)")

# AppLauncher (自动添加--headless等参数)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# 强制禁用headless模式以显示GUI
args_cli.headless = False

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# 导入Isaac Lab模块
from isaaclab.envs import ManagerBasedRLEnv
from rosorin_env_cfg import ROSOrinEnvCfg

def main():
    """可视化测试主函数"""
    
    print("\n" + "="*80)
    print("  🎬 ROSOrin 可视化导航测试")
    print("="*80)
    
    # 创建环境配置
    env_cfg = ROSOrinEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.episode_length_s = 20.0  # 20秒episodes
    
    print(f"\n📊 配置:")
    print(f"  环境数量: {env_cfg.scene.num_envs}")
    print(f"  Episode: {env_cfg.episode_length_s}秒")
    print(f"  奖励函数: 8项")
    print(f"  终止条件: 3项")
    
    # 创建环境
    print(f"\n🔧 创建环境 (GUI模式)...")
    print(f"  提示: 窗口将在几秒后打开...")
    
    env = ManagerBasedRLEnv(cfg=env_cfg)
    
    print(f"\n✅ 环境已创建! Isaac Sim窗口应该已显示")
    print(f"\n🎮 控制说明:")
    print(f"  - 鼠标左键拖动: 旋转视角")
    print(f"  - 鼠标中键拖动: 平移视角")
    print(f"  - 鼠标滚轮: 缩放")
    print(f"  - 按 'P' 键: 暂停/继续模拟")
    
    # Reset环境
    print(f"\n🔄 重置环境...")
    obs_dict, _ = env.reset()
    
    # 显示目标点信息
    if hasattr(env, 'goal_positions'):
        print(f"\n🎯 目标点已生成:")
        for i in range(env.num_envs):
            goal = env.goal_positions[i, :2]
            robot_pos = env.scene.articulations["robot"].data.root_pos_w[i, :2]
            distance = torch.norm(goal - robot_pos).item()
            print(f"  Env {i}: 目标({goal[0]:.2f}, {goal[1]:.2f}) | 距离: {distance:.2f}m")
    
    # 运行模拟
    print(f"\n🏃 开始模拟 (运行200步)...")
    print(f"  观察:")
    print(f"  - 机器人如何向目标点移动")
    print(f"  - 奖励值变化")
    print(f"  - 姿态稳定性")
    
    episode_rewards = torch.zeros(env.num_envs, device=env.device)
    step_count = 0
    max_steps = 200
    
    for step in range(max_steps):
        # 简单的导航策略：朝目标方向移动 (测试可视化)
        if hasattr(env, 'goal_positions'):
            # 获取机器人当前位置和朝向
            robot_pos = env.scene.articulations["robot"].data.root_pos_w[:, :2]  # (N, 2)
            robot_quat = env.scene.articulations["robot"].data.root_quat_w  # (N, 4)
            
            # 计算朝向目标的方向向量
            goal_vec = env.goal_positions[:, :2] - robot_pos  # (N, 2)
            goal_distance = torch.norm(goal_vec, dim=-1, keepdim=True)  # (N, 1)
            goal_dir = goal_vec / (goal_distance + 1e-6)  # (N, 2) 归一化
            
            # 提取机器人yaw角
            # quat = [x, y, z, w], yaw = atan2(2*(w*z + x*y), 1 - 2*(y^2 + z^2))
            yaw = torch.atan2(
                2.0 * (robot_quat[:, 3] * robot_quat[:, 2] + robot_quat[:, 0] * robot_quat[:, 1]),
                1.0 - 2.0 * (robot_quat[:, 1]**2 + robot_quat[:, 2]**2)
            )  # (N,)
            robot_dir = torch.stack([torch.cos(yaw), torch.sin(yaw)], dim=-1)  # (N, 2)
            
            # 计算转向误差 (点积判断是否对齐)
            alignment = (robot_dir * goal_dir).sum(dim=-1)  # (N,) 范围[-1, 1]
            cross = robot_dir[:, 0] * goal_dir[:, 1] - robot_dir[:, 1] * goal_dir[:, 0]  # 叉积判断左右
            
            # 构造简单的麦克纳姆轮控制 (4个轮子速度)
            # 前进速度: 基于距离和对齐度
            forward_vel = torch.clamp(goal_distance.squeeze(-1) * 0.3, 0, 1.0) * (alignment + 1.0) / 2.0
            # 转向速度: 基于叉积
            turn_vel = torch.clamp(cross * 0.8, -0.5, 0.5)
            
            # 麦克纳姆轮差速: [左前, 右前, 左后, 右后]
            actions = torch.stack([
                forward_vel + turn_vel,  # 左前
                forward_vel - turn_vel,  # 右前
                forward_vel + turn_vel,  # 左后
                forward_vel - turn_vel,  # 右后
            ], dim=-1)  # (N, 4)
        else:
            # 如果没有目标，使用更大的随机动作测试
            actions = (torch.rand(env.num_envs, 4, device=env.device) - 0.5) * 1.5  # 范围 [-0.75, 0.75]
        
        # 执行
        obs_dict, rewards, dones, truncated, infos = env.step(actions)
        
        episode_rewards += rewards
        step_count += 1
        
        # 每20步打印一次状态
        if (step + 1) % 20 == 0:
            print(f"\n  Step {step+1}/{max_steps}:")
            print(f"    平均奖励: {episode_rewards.mean().item():.3f}")
            print(f"    最大奖励: {episode_rewards.max().item():.3f}")
            
            # 显示距离变化
            if hasattr(env, 'goal_positions'):
                distances = torch.norm(
                    env.scene.articulations["robot"].data.root_pos_w[:, :2] - env.goal_positions[:, :2],
                    dim=-1
                )
                print(f"    平均距离目标: {distances.mean().item():.2f}m")
        
        # 检查终止
        if dones.any():
            done_envs = dones.nonzero(as_tuple=True)[0]
            print(f"\n  ⚠️  环境 {done_envs.tolist()} 终止 (可能到达目标或倾覆)")
            # Reset终止的环境
            env.reset(env_ids=done_envs)
        
        # 稍微延迟以便观察
        time.sleep(0.02)  # 20ms延迟
    
    # 最终统计
    print(f"\n" + "="*80)
    print(f"📊 模拟结束统计:")
    print(f"  总步数: {step_count}")
    print(f"  平均总奖励: {episode_rewards.mean().item():.3f}")
    print(f"  奖励范围: [{episode_rewards.min().item():.3f}, {episode_rewards.max().item():.3f}]")
    
    # 保持窗口打开
    print(f"\n💡 提示:")
    print(f"  - Isaac Sim窗口将保持打开")
    print(f"  - 可以继续手动操作相机查看场景")
    print(f"  - 按 Ctrl+C 退出")
    print(f"="*80 + "\n")
    
    # 等待用户关闭
    try:
        print("⏸️  模拟已暂停。按 Ctrl+C 退出...")
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n👋 用户中断，关闭环境...")
    
    # 关闭环境
    env.close()
    print("✅ 环境已关闭\n")


if __name__ == "__main__":
    # 运行可视化测试
    main()
    
    # 关闭模拟器
    simulation_app.close()
