"""
最小化测试 - 使用DirectRLEnv (无需observations/actions配置)
"""

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--headless", action="store_true")
args = parser.parse_args()

print("\n[1/4] 启动SimulationApp...")
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": args.headless})
print("✓ SimulationApp启动成功")

print("\n[2/4] 导入Isaac Lab模块...")
import torch
import gymnasium as gym
import isaaclab.sim as sim_utils
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.utils import configclass
print("✓ 模块导入成功")

print("\n[3/4] 创建最简环境...")

@configclass
class MinimalSceneCfg(InteractiveSceneCfg):
    """最小场景: 空场景"""
    pass
@configclass
class MinimalEnvCfg(DirectRLEnvCfg):
    """最小环境配置"""
    scene: MinimalSceneCfg = MinimalSceneCfg(num_envs=args.num_envs, env_spacing=2.0)
    
    # 基础参数
    decimation = 4
    episode_length_s = 5.0
    num_actions = 0  # 无动作
    num_observations = 1  # 最小1个观测
    num_states = 0
    
    # 定义空间
    observation_space = gym.spaces.Box(low=-1, high=1, shape=(1,))
    action_space = gym.spaces.Box(low=-1, high=1, shape=(0,))

class MinimalEnv(DirectRLEnv):
    cfg: MinimalEnvCfg
    
    def __init__(self, cfg: MinimalEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
    
    def _setup_scene(self):
        """设置场景"""
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[])
    
    def _pre_physics_step(self, actions: torch.Tensor):
        """物理步之前"""
        pass
    
    def _apply_action(self):
        """应用动作（必须实现）"""
        pass
    
    def _get_observations(self) -> dict:
        """获取观测"""
        return {"policy": torch.zeros(self.num_envs, 1, device=self.device)}
    
    def _get_rewards(self) -> torch.Tensor:
        """计算奖励"""
        return torch.zeros(self.num_envs, device=self.device)
    
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """检查终止"""
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return torch.zeros_like(time_out), time_out
    
    def _reset_idx(self, env_ids: torch.Tensor | None):
        """重置环境"""
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = torch.arange(self.num_envs, device=self.device)
        super()._reset_idx(env_ids)

# 创建环境
env_cfg = MinimalEnvCfg()
env = MinimalEnv(cfg=env_cfg)
print(f"✓ 环境创建成功 ({env.num_envs} envs, {env.device})")

print("\n[4/4] 运行500步仿真...")
count = 0
while simulation_app.is_running() and count < 500:
    with torch.inference_mode():
        if count % 100 == 0:
            obs, _ = env.reset()
            print(f"  Step {count}: 环境重置")
        else:
            obs, rew, terminated, truncated, info = env.step(torch.zeros(env.num_envs, 0, device=env.device))
        
        if count % 100 == 99:
            print(f"  Step {count}: OK")
        
        count += 1

print(f"\n✓ 仿真完成! 总步数: {count}\n")

print("="*70)
print("✅ 测试通过! Isaac Lab环境运行正常")
print("="*70 + "\n")

env.close()
simulation_app.close()
