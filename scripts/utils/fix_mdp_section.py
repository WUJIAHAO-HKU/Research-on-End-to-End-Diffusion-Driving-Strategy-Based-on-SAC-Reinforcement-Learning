#!/usr/bin/env python3
"""修复环境配置文件的MDP部分"""

# 读取原文件
with open('rosorin_env_cfg.py', 'r') as f:
    lines = f.readlines()

# 找到policy: PolicyCfg = PolicyCfg()这一行
policy_line_idx = None
for i, line in enumerate(lines):
    if 'policy: PolicyCfg = PolicyCfg()' in line:
        policy_line_idx = i
        break

if policy_line_idx is None:
    print("Error: 找不到policy定义行")
    exit(1)

#找到最后的@configclass class ROSOrinEnvCfg
env_cfg_idx = None
for i, line in enumerate(lines):
    if '@configclass' in line and i > policy_line_idx:
        if 'class ROSOrinEnvCfg' in lines[i+1]:
            env_cfg_idx = i
            break

if env_cfg_idx is None:
    print("Error: 找不到ROSOrinEnvCfg")
    exit(1)

print(f"policy定义在第 {policy_line_idx+1} 行")
print(f"ROSOrinEnvCfg在第 {env_cfg_idx+1} 行")

# 准备新的MDP配置部分
new_mdp_section = '''

@configclass
class RewardsCfg:
    """
    完整的奖励函数体系 for ROSOrin导航任务
    
    设计原则:
    1. 主要奖励: 导航进度 + 目标到达
    2. 辅助奖励: 速度控制 + 朝向对齐
    3. 惩罚项: 动作突变 + 姿态不稳 + 高度异常
    """
    
    # ========== 主要奖励 (Main Rewards) ==========
    # 向目标前进的进度奖励 (最重要)
    progress = RewTerm(
        func=rosorin_mdp.progress_reward,
        weight=10.0,  # 高权重，鼓励向目标前进
        params={"threshold": 0.01}  # 距离减少1cm以上才奖励
    )
    
    # 到达目标点的大额奖励
    goal_reached = RewTerm(
        func=rosorin_mdp.goal_reached_reward,
        weight=100.0,  # 成功到达给予巨大奖励
        params={"distance_threshold": 0.5}  # 0.5米内算到达
    )
    
    # ========== 辅助奖励 (Auxiliary Rewards) ==========
    # 速度跟踪奖励 (保持合适速度)
    velocity_tracking = RewTerm(
        func=rosorin_mdp.velocity_tracking_reward,
        weight=1.0,
        params={"target_vel": 0.3}  # 目标速度0.3m/s
    )
    
    # 朝向对齐奖励 (朝向目标点)
    orientation = RewTerm(
        func=rosorin_mdp.orientation_alignment_reward,
        weight=2.0
    )
    
    # 基础存活奖励 (保持运行)
    alive = RewTerm(
        func=mdp.is_alive,
        weight=0.1
    )
    
    # ========== 惩罚项 (Penalties) ==========
    # 动作平滑惩罚 (避免抖动)
    action_smoothness = RewTerm(
        func=rosorin_mdp.smooth_action_penalty,
        weight=0.5  # 轻度惩罚
    )
    
    # 姿态稳定惩罚 (避免倾覆)
    stability = RewTerm(
        func=rosorin_mdp.stability_penalty,
        weight=5.0,  # 中等惩罚
        params={
            "roll_threshold": 0.2,   # 允许±11.5度roll
            "pitch_threshold": 0.2   # 允许±11.5度pitch
        }
    )
    
    # 高度惩罚 (保持合理高度)
    height = RewTerm(
        func=rosorin_mdp.height_penalty,
        weight=2.0,
        params={
            "min_height": 0.05,  # 最低5cm
            "max_height": 0.25   # 最高25cm
        }
    )


@configclass
class TerminationsCfg:
    """
    终止条件配置
    
    Episode终止情况:
    1. 成功到达目标点
    2. 机器人倾覆
    3. 超时
    """
    
    # 成功到达目标 (SUCCESS)
    goal_reached = DoneTerm(
        func=rosorin_mdp.goal_reached_termination,
        params={"distance_threshold": 0.5}
    )
    
    # 机器人倾覆 (FAILURE)
    robot_fallen = DoneTerm(
        func=rosorin_mdp.robot_fallen_termination,
        params={
            "roll_threshold": 0.5,   # 约28.6度
            "pitch_threshold": 0.5
        }
    )
    
    # 超时 (TIMEOUT)
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class EventCfg:
    """
    环境随机化配置
    
    每次reset时:
    1. 随机化机器人起始位置和朝向
    2. 随机生成新的目标点
    """
    
    # 随机化机器人起始位置
    reset_robot_position = EventTerm(
        func=rosorin_mdp.reset_robot_to_random_position,
        mode="reset",
        params={
            "x_range": (-3.0, 3.0),      # ±3米范围
            "y_range": (-3.0, 3.0),
            "yaw_range": (-3.14, 3.14)   # 任意朝向
        }
    )
    
    # 随机生成目标点位置
    reset_goal_position = EventTerm(
        func=rosorin_mdp.reset_goal_position,
        mode="reset",
        params={
            "min_distance": 3.0,  # 至少3米远
            "max_distance": 8.0   # 最多8米远
        }
    )


'''

# 重新组装文件
new_lines = lines[:policy_line_idx+1] + [new_mdp_section] + lines[env_cfg_idx:]

# 写入文件
with open('rosorin_env_cfg.py', 'w') as f:
    f.writelines(new_lines)

print("✓ 配置文件已修复")
