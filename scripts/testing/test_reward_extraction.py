#!/usr/bin/env python3
"""
快速测试奖励提取系统是否正常工作

运行少量步骤，检查是否能正确提取各个奖励组件
"""

import torch
import sys
from pathlib import Path

# 添加项目路径到最前面，避免ROS的scripts包冲突
sys.path.insert(0, str(Path(__file__).parent))

# 导入环境配置（直接从当前目录导入）
from rosorin_env_cfg import ROSOrinEnvCfg
from isaaclab.envs import ManagerBasedRLEnv


def extract_reward_components(env):
    """从Isaac Lab环境的reward_manager中提取各个奖励项的值"""
    reward_dict = {}
    
    try:
        if hasattr(env.unwrapped, 'reward_manager'):
            manager = env.unwrapped.reward_manager
            if hasattr(manager, '_term_buffers'):
                for term_name, term_buffer in manager._term_buffers.items():
                    if isinstance(term_buffer, torch.Tensor):
                        reward_dict[term_name] = term_buffer.mean().item()
    except Exception as e:
        print(f"提取失败: {e}")
    
    return reward_dict


def main():
    """主测试函数"""
    print("=" * 80)
    print("测试奖励提取系统")
    print("=" * 80)
    
    # 创建环境
    print("\n1. 创建环境...")
    env_cfg = ROSOrinEnvCfg()
    env_cfg.scene.num_envs = 4  # 使用4个并行环境
    env = ManagerBasedRLEnv(cfg=env_cfg)
    print("   ✓ 环境创建成功")
    
    # 检查奖励管理器
    print("\n2. 检查奖励管理器...")
    if hasattr(env.unwrapped, 'reward_manager'):
        manager = env.unwrapped.reward_manager
        print(f"   ✓ 找到奖励管理器")
        print(f"   奖励项: {manager._term_names}")
    else:
        print("   ✗ 未找到奖励管理器！")
        return
    
    # 重置环境
    print("\n3. 重置环境...")
    obs_dict, _ = env.reset()
    obs = obs_dict["policy"]
    action_dim = env.action_space.shape[-1]
    print(f"   ✓ 观察维度: {obs.shape}")
    print(f"   ✓ 动作维度: {action_dim}")
    
    # 运行几步
    print("\n4. 运行测试步骤...")
    num_test_steps = 10
    
    for step in range(num_test_steps):
        # 随机动作
        actions = torch.rand(env.num_envs, action_dim, device=env.device) * 2 - 1
        
        # 执行
        next_obs_dict, rewards, terminated, truncated, infos = env.step(actions)
        
        # 提取奖励组件
        reward_components = extract_reward_components(env)
        
        # 显示
        print(f"\n   Step {step + 1}:")
        print(f"     总奖励 (mean): {rewards.mean().item():.4f}")
        print(f"     奖励组件:")
        
        if reward_components:
            for key, value in sorted(reward_components.items()):
                print(f"       {key:20s}: {value:8.4f}")
        else:
            print("       ⚠️ 未能提取奖励组件！")
        
        # 检查是否有环境终止
        dones = terminated | truncated
        if dones.any():
            num_done = dones.sum().item()
            print(f"     已终止环境数: {num_done}")
    
    # 关闭环境
    print("\n5. 关闭环境...")
    env.close()
    print("   ✓ 测试完成")
    
    print("\n" + "=" * 80)
    print("✓ 奖励提取系统测试成功！")
    print("=" * 80)


if __name__ == "__main__":
    main()
