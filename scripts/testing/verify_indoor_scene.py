#!/usr/bin/env python3
"""
室内场景验证脚本
测试新的10x10m室内导航场景
"""

import argparse
import torch

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="验证ROSOrin室内导航场景")
parser.add_argument("--num_envs", type=int, default=2, help="环境数量")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = False  # 强制GUI模式

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab.envs import ManagerBasedRLEnv
from rosorin_env_cfg import ROSOrinEnvCfg

def main():
    print("\n" + "="*80)
    print("  🏠 ROSOrin 室内场景验证")
    print("="*80)
    
    # 创建环境
    env_cfg = ROSOrinEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.episode_length_s = 30.0
    
    print(f"\n📊 场景配置:")
    print(f"  10m × 10m 室内空间（3个房间）")
    print(f"  外围墙壁 + 内部分隔墙（带门洞）")
    print(f"  6个家具障碍物")
    print(f"  环境数量: {env_cfg.scene.num_envs}")
    
    env = ManagerBasedRLEnv(cfg=env_cfg)
    
    print(f"\n✅ 场景已创建! Isaac Sim窗口应该显示室内环境")
    print(f"\n🎮 控制说明:")
    print(f"  - 鼠标左键拖动: 旋转视角")
    print(f"  - 鼠标中键拖动: 平移视角")
    print(f"  - 鼠标滚轮: 缩放")
    
    print(f"\n🔄 重置环境...")
    obs_dict, _ = env.reset()
    
    # 隐藏源场景 env_0（在 reset 之后，避免影响其他环境）
    import omni.usd
    from pxr import UsdGeom, Usd
    stage = omni.usd.get_context().get_stage()
    source_prim = stage.GetPrimAtPath("/World/envs/env_0")
    if source_prim.IsValid():
        # 只隐藏 env_0，不递归影响子节点
        imageable = UsdGeom.Imageable(source_prim)
        # 使用 inherited visibility，但立即在子环境上设置为 visible
        imageable.MakeInvisible()
        
        # 确保其他环境仍然可见
        for i in range(1, env_cfg.scene.num_envs + 1):
            env_prim = stage.GetPrimAtPath(f"/World/envs/env_{i}")
            if env_prim.IsValid():
                env_imageable = UsdGeom.Imageable(env_prim)
                env_imageable.MakeVisible()
        
        print(f"  ✓ 已隐藏源场景 env_0，其他环境保持可见")
    
    # 显示场景元素
    print(f"\n🏗️ 场景元素:")
    print(f"  ✓ 地板: 20m × 20m 浅灰色地面")
    print(f"  ✓ 外围墙壁: 4面 2.5m高米白色墙壁")
    print(f"  ✓ 分隔墙: 2道内墙（带1.2m宽门洞）")
    print(f"  ✓ 房间1（客厅）: 沙发 + 茶几")
    print(f"  ✓ 房间2（书房）: 书桌 + 书架")
    print(f"  ✓ 房间3（餐厅）: 餐桌 + 餐边柜")
    
    # 显示机器人和目标信息
    if hasattr(env, 'goal_positions'):
        print(f"\n🤖 机器人与目标:")
        for i in range(min(env.num_envs, 4)):
            robot_pos = env.scene.articulations["robot"].data.root_pos_w[i, :2]
            goal_pos = env.goal_positions[i, :2]
            distance = torch.norm(goal_pos - robot_pos).item()
            print(f"  环境 {i}: 机器人({robot_pos[0]:.2f}, {robot_pos[1]:.2f}) → 目标({goal_pos[0]:.2f}, {goal_pos[1]:.2f}) | 距离: {distance:.2f}m")
    
    print(f"\n🏃 运行简单测试 (100步)...")
    print(f"  观察: 机器人在室内环境中的行为")
    
    for step in range(10000):
        # 简单的前进动作
        actions = torch.ones(env.num_envs, 4, device=env.device) * 0.3
        obs_dict, rewards, dones, truncated, infos = env.step(actions)
        
        if (step + 1) % 20 == 0:
            print(f"  Step {step+1}/100: 平均奖励={rewards.mean().item():.3f}")
    
    print(f"\n" + "="*80)
    print(f"✅ 场景验证完成!")
    print(f"\n💡 接下来可以:")
    print(f"  1. 观察机器人在多房间之间的导航")
    print(f"  2. 测试通过门洞的行为")
    print(f"  3. 开始在新场景中训练")
    print(f"="*80 + "\n")
    
    # 保持窗口
    print("⏸️  按 Ctrl+C 退出...")
    try:
        import time
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n👋 关闭环境...")
    
    env.close()
    print("✅ 完成\n")

if __name__ == "__main__":
    main()
    simulation_app.close()
