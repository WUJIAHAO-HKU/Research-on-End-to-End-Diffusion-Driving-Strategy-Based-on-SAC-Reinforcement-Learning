"""
训练脚本模块

该目录包含所有强化学习算法的训练脚本：
- train_ppo.py: PPO算法训练
- train_sac_gaussian.py: SAC算法（高斯策略）训练
- train_sac_diffusion.py: SAC算法（扩散策略）训练
- train_bc.py: Behavioral Cloning训练
- train_td3.py: TD3算法训练
- train_dagger.py: DAgger算法训练

使用方法:
  ./isaaclab_runner.sh scripts/training/train_ppo.py --num_envs 8
"""
