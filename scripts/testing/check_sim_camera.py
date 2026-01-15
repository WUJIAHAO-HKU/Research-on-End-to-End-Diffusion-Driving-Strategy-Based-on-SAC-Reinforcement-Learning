#!/usr/bin/env python3
"""Sim camera sanity check.

这个脚本用于验证 Isaac Lab 仿真环境里的 RGB-D 相机是否:
- 正常出图（非全 0、非全 NaN/Inf）
- 在 step 后持续更新
- 深度值范围合理（与 camera clipping_range 一致）

输出:
- experiments/debug/camera_check/<timestamp>/
  - rgb_env0_step000.ppm
  - depth_env0_step000.npy
  - depth_env0_step000.pgm  (可直接用常见图片查看器打开)

使用:
  ./isaaclab_runner.sh scripts/testing/check_sim_camera.py --num_envs 1 --steps 3
"""

import argparse
import os
from pathlib import Path
from datetime import datetime

import numpy as np
import torch

from isaaclab.app import AppLauncher


def _write_ppm(path: Path, rgb_u8: np.ndarray) -> None:
    """Write RGB image as binary PPM (P6)."""
    if rgb_u8.dtype != np.uint8:
        raise ValueError(f"PPM expects uint8, got {rgb_u8.dtype}")
    if rgb_u8.ndim != 3 or rgb_u8.shape[-1] != 3:
        raise ValueError(f"PPM expects HxWx3, got {rgb_u8.shape}")

    h, w, _ = rgb_u8.shape
    header = f"P6\n{w} {h}\n255\n".encode("ascii")
    with open(path, "wb") as f:
        f.write(header)
        f.write(rgb_u8.tobytes(order="C"))


def _write_pgm16(path: Path, gray_u16: np.ndarray) -> None:
    """Write 16-bit grayscale as binary PGM (P5).

    Note: For maxval > 255, PGM uses big-endian 16-bit samples.
    """
    if gray_u16.dtype != np.uint16:
        raise ValueError(f"PGM16 expects uint16, got {gray_u16.dtype}")
    if gray_u16.ndim != 2:
        raise ValueError(f"PGM16 expects HxW, got {gray_u16.shape}")

    h, w = gray_u16.shape
    header = f"P5\n{w} {h}\n65535\n".encode("ascii")
    with open(path, "wb") as f:
        f.write(header)
        f.write(gray_u16.astype(">u2", copy=False).tobytes(order="C"))


def _stats(name: str, arr: torch.Tensor) -> None:
    arr_f = arr.float()
    nan = torch.isnan(arr_f).float().mean().item() * 100.0
    inf = torch.isinf(arr_f).float().mean().item() * 100.0

    arr_safe = torch.nan_to_num(arr_f, nan=0.0, posinf=0.0, neginf=0.0)
    print(
        f"  {name}: shape={tuple(arr.shape)} dtype={arr.dtype} "
        f"min={arr_safe.min().item():.3f} max={arr_safe.max().item():.3f} mean={arr_safe.mean().item():.3f} "
        f"NaN%={nan:.2f} Inf%={inf:.2f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Check Isaac Lab sim RGB-D camera output")
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--steps", type=int, default=3, help="Step count before saving")
    parser.add_argument("--save_env_id", type=int, default=0)
    parser.add_argument("--out_dir", type=str, default="experiments/debug/camera_check")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    args.headless = True
    args.enable_cameras = True

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    # Import project helpers after AppLauncher
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    from env_factory import create_ppo_env_cfg
    from isaaclab.envs import ManagerBasedRLEnv

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Sim RGB-D Camera Sanity Check")
    print("=" * 80)
    print(f"num_envs={args.num_envs} steps={args.steps} save_env_id={args.save_env_id}")
    print(f"out_dir={out_dir}")

    env_cfg = create_ppo_env_cfg(num_envs=args.num_envs)
    env = ManagerBasedRLEnv(cfg=env_cfg)

    obs_dict, _ = env.reset()
    _ = obs_dict  # not used; we read sensor outputs directly

    device = env.device
    action_dim = env.action_space.shape[-1]
    actions = torch.zeros((args.num_envs, action_dim), device=device)

    # Step a few times to ensure sensor buffers are populated/updated.
    for step_idx in range(max(args.steps, 1)):
        _, _, _, _, _ = env.step(actions)

        rgb = env.scene.sensors["camera"].data.output["rgb"]
        depth = env.scene.sensors["camera"].data.output["distance_to_image_plane"]

        print(f"\n[step {step_idx:03d}]")
        _stats("rgb", rgb)
        _stats("depth", depth)

        # Save first step only (most useful for quick check)
        if step_idx == 0:
            env_id = int(args.save_env_id)
            if env_id < 0 or env_id >= args.num_envs:
                raise ValueError(f"save_env_id out of range: {env_id}")

            rgb0 = rgb[env_id].detach().cpu()
            depth0 = depth[env_id].detach().cpu()

            # RGB may be float[0,1] or uint8; normalize to uint8 for PPM.
            if rgb0.dtype != torch.uint8:
                rgb0_u8 = (torch.clamp(rgb0.float(), 0.0, 1.0) * 255.0).to(torch.uint8)
            else:
                rgb0_u8 = rgb0

            rgb0_np = rgb0_u8.numpy()
            # Depth is typically HxWx1; squeeze to HxW for easier inspection.
            depth0_np = depth0.float().squeeze(-1).numpy()

            rgb_path = out_dir / f"rgb_env{env_id}_step{step_idx:03d}.ppm"
            depth_npy_path = out_dir / f"depth_env{env_id}_step{step_idx:03d}.npy"
            depth_pgm_path = out_dir / f"depth_env{env_id}_step{step_idx:03d}.pgm"

            _write_ppm(rgb_path, rgb0_np)
            np.save(depth_npy_path, depth0_np)

            # Depth visualization: clip to [0, 10]m and invert so nearer is brighter.
            depth_clip = np.clip(depth0_np, 0.0, 10.0)
            depth_vis = (1.0 - depth_clip / 10.0) * 65535.0
            depth_vis_u16 = depth_vis.astype(np.uint16)
            _write_pgm16(depth_pgm_path, depth_vis_u16)

            print("\nSaved:")
            print(f"  {rgb_path}")
            print(f"  {depth_npy_path}")
            print(f"  {depth_pgm_path}")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
