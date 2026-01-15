"""
Collect Expert Demonstrations

Use MPC controller to collect expert demonstrations for behavior cloning.
"""

import hydra
from omegaconf import DictConfig
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.envs.isaac_lab.rosorin_car_env import ROSOrinDrivingEnv
from src.baselines.mpc_controller import MPCController, NonlinearMPCController
from src.data.demonstration_collector import MPCDemonstrationCollector


@hydra.main(config_path="../configs", config_name="collect_demonstrations", version_base="1.2")
def main(config: DictConfig):
    """Collect demonstrations using MPC"""
    
    # Create environment
    env = ROSOrinDrivingEnv(config.env)
    
    # Create MPC controller
    if config.mpc.nonlinear:
        mpc = NonlinearMPCController(**config.mpc.params)
    else:
        mpc = MPCController(**config.mpc.params)
    
    # Create collector
    collector = MPCDemonstrationCollector(
        env=env,
        save_dir=Path(config.save_dir),
        mpc_controller=mpc,
        max_episodes=config.max_episodes,
        save_format=config.save_format,
        save_video=config.save_video,
    )
    
    # Collect demonstrations
    collector.collect(num_episodes=config.num_episodes)
    
    # Save
    collector.save(filename=config.output_filename)
    
    # Print statistics
    stats = collector.get_statistics()
    print("\n" + "="*60)
    print("DEMONSTRATION COLLECTION SUMMARY")
    print("="*60)
    for key, value in stats.items():
        print(f"{key}: {value}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
