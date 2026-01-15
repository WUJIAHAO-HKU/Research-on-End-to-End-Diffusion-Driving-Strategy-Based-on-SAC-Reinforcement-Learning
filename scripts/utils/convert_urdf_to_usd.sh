#!/bin/bash
# ROSOrin URDF → USD 转换脚本
# 使用Isaac Sim的Python脚本进行转换

set -e

URDF_FILE="/home/wujiahao/ROSORIN_CAR and Reasearch/Research on End-to-End Diffusion Driving Strategy Based on SAC Reinforcement Learning/rosorin_ws/rosorin_full.urdf"
OUTPUT_USD="/home/wujiahao/ROSORIN_CAR and Reasearch/Research on End-to-End Diffusion Driving Strategy Based on SAC Reinforcement Learning/data/assets/rosorin/rosorin.usd"

echo "========================================"
echo "  ROSOrin URDF → USD 转换"
echo "========================================"
echo ""
echo "输入: $URDF_FILE"
echo "输出: $OUTPUT_USD"
echo ""

# 创建输出目录
mkdir -p "$(dirname "$OUTPUT_USD")"

# 使用isaacsim Python环境运行转换
conda run -n isaaclab_pip python3 << 'PYTHON_SCRIPT'
import sys
sys.path.insert(0, '/home/wujiahao/miniconda3/envs/isaaclab_pip/lib/python3.11/site-packages')

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

import omni.kit.commands
from pxr import Usd, UsdGeom
import carb

urdf_file = "/home/wujiahao/ROSORIN_CAR and Reasearch/Research on End-to-End Diffusion Driving Strategy Based on SAC Reinforcement Learning/rosorin_ws/rosorin_full.urdf"
output_usd = "/home/wujiahao/ROSORIN_CAR and Reasearch/Research on End-to-End Diffusion Driving Strategy Based on SAC Reinforcement Learning/data/assets/rosorin/rosorin.usd"

print(f"\n[1/3] 启用URDF导入扩展...")
import omni.kit.app
ext_manager = omni.kit.app.get_app().get_extension_manager()

# 尝试可能的URDF扩展名称
urdf_extensions = [
    "omni.importer.urdf",
    "omni.isaac.urdf", 
    "omni.kit.asset_converter"
]

enabled = False
for ext_name in urdf_extensions:
    try:
        ext_manager.set_extension_enabled_immediate(ext_name, True)
        print(f"  ✓ 已启用: {ext_name}")
        enabled = True
        break
    except Exception as e:
        print(f"  × {ext_name}: {e}")
        continue

if not enabled:
    print("\n[ERROR] 无可用的URDF导入扩展！")
    print("\n可用扩展列表:")
    for ext_id in ext_manager.get_extensions():
        if 'urdf' in ext_id.lower() or 'import' in ext_id.lower():
            print(f"  - {ext_id}")
    simulation_app.close()
    sys.exit(1)

print(f"\n[2/3] 导入URDF: {urdf_file}")

try:
    # 尝试使用URDFParseAndImportFile命令
    result = omni.kit.commands.execute(
        "URDFParseAndImportFile",
        urdf_path=urdf_file,
        import_inertia_tensor=True,
        fix_base=False,
        merge_fixed_joints=False,
    )
    
    if result:
        print(f"  ✓ URDF导入成功")
    else:
        print(f"  × URDF导入失败")
        simulation_app.close()
        sys.exit(1)
        
except Exception as e:
    print(f"\n[ERROR] URDF导入异常: {e}")
    print("\n尝试备用方法...")
    
    # 备用方法：使用资产转换器
    try:
        import omni.kit.asset_converter as converter
        # TODO: 实现备用转换逻辑
        print("  使用资产转换器...")
    except:
        pass
    
    simulation_app.close()
    sys.exit(1)

print(f"\n[3/3] 保存USD: {output_usd}")

try:
    stage = omni.usd.get_context().get_stage()
    stage.Export(output_usd)
    print(f"  ✓ USD文件已保存")
except Exception as e:
    print(f"  × 保存失败: {e}")
    simulation_app.close()
    sys.exit(1)

print(f"\n{'='*40}")
print(f"✓ 转换完成！")
print(f"  USD: {output_usd}")
print(f"{'='*40}\n")

simulation_app.close()
PYTHON_SCRIPT

echo ""
echo "✓ 转换完成"
