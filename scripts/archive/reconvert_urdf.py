#!/usr/bin/env python3
"""重新转换URDF为USD，确保articulation根在正确位置"""

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

import omni
from omni.importer.urdf import _urdf as urdf_importer
import carb

urdf_path = "/home/wujiahao/ROSORIN_CAR and Reasearch/Research on End-to-End Diffusion Driving Strategy Based on SAC Reinforcement Learning/rosorin_ws/rosorin_full.urdf"
output_path = "/home/wujiahao/ROSORIN_CAR and Reasearch/Research on End-to-End Diffusion Driving Strategy Based on SAC Reinforcement Learning/data/assets/rosorin/rosorin_fixed.usd"

print(f"\n正在转换URDF: {urdf_path}")
print(f"输出USD: {output_path}")

# 配置URDF导入器
import_config = urdf_importer.ImportConfig()
import_config.merge_fixed_joints = False  # 保留固定关节以便后续设置fix_root_link
import_config.convex_decomp = False  # 使用原始网格作为碰撞
import_config.import_inertia_tensor = True
import_config.fix_base = False  # 关键！不要固定base
import_config.distance_scale = 1.0
import_config.density = 0.0  # 使用URDF中的质量
import_config.default_drive_type = urdf_importer.UrdfJointTargetType.JOINT_DRIVE_VELOCITY

print("\n导入配置:")
print(f"  fix_base: {import_config.fix_base}")
print(f"  merge_fixed_joints: {import_config.merge_fixed_joints}")
print(f"  default_drive_type: VELOCITY")

# 执行导入
result, prim_path = omni.kit.commands.execute(
    "URDFParseAndImportFile",
    urdf_path=urdf_path,
    import_config=import_config,
    dest_path="/World/Robot"
)

if result:
    print(f"\n✓ URDF导入成功")
    print(f"  Prim路径: {prim_path}")
    
    # 保存为USD
    stage = omni.usd.get_context().get_stage()
    stage.Export(output_path)
    print(f"\n✓ USD已保存: {output_path}")
else:
    print("\n✗ URDF导入失败")

simulation_app.close()
