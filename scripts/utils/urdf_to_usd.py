"""
ROSOrin URDF → USD 转换脚本

使用Isaac Sim的URDF导入功能将ROSOrin机器人转换为USD格式
"""

import sys
import argparse
from pathlib import Path

# Isaac Sim导入必须在最前面
from isaacsim import SimulationApp

# 配置
URDF_PATH = Path("/home/wujiahao/ROSORIN_CAR and Reasearch/Research on End-to-End Diffusion Driving Strategy Based on SAC Reinforcement Learning/rosorin_ws/rosorin_full.urdf")
OUTPUT_PATH = Path("/home/wujiahao/ROSORIN_CAR and Reasearch/Research on End-to-End Diffusion Driving Strategy Based on SAC Reinforcement Learning/data/assets/rosorin/rosorin.usd")

# 启动Isaac Sim (headless模式)
simulation_app = SimulationApp({"headless": True})

# 现在可以导入其他模块
import omni.kit.commands
import omni.kit.app
import omni.usd
from pxr import Usd
import carb

def convert_urdf_to_usd():
    """执行URDF到USD的转换"""
    
    print(f"\n{'='*70}")
    print(f"  ROSOrin URDF → USD 转换")
    print(f"{'='*70}\n")
    
    # 检查输入文件
    if not URDF_PATH.exists():
        print(f"[ERROR] URDF文件不存在: {URDF_PATH}")
        return False
    
    print(f"[1/4] URDF文件: {URDF_PATH}")
    print(f"      大小: {URDF_PATH.stat().st_size} bytes")
    
    # 创建输出目录
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n[2/4] 输出路径: {OUTPUT_PATH}")
    
    # 启用URDF扩展
    print(f"\n[3/4] 启用URDF导入扩展...")
    ext_manager = omni.kit.app.get_app().get_extension_manager()
    
    # 列出所有包含urdf的扩展
    all_extensions = ext_manager.get_extensions()
    if isinstance(all_extensions, dict):
        urdf_exts = [ext_id for ext_id in all_extensions.keys() if 'urdf' in str(ext_id).lower()]
    else:
        urdf_exts = [ext for ext in all_extensions if 'urdf' in str(ext).lower()]
    
    if urdf_exts:
        print(f"      找到URDF相关扩展:")
        for ext in urdf_exts:
            print(f"        - {ext}")
    
    # 尝试启用URDF导入扩展
    urdf_extension_names = [
        "omni.importer.urdf",
        "omni.isaac.urdf",
    ]
    
    extension_enabled = False
    for ext_name in urdf_extension_names:
        try:
            ext_manager.set_extension_enabled_immediate(ext_name, True)
            print(f"      ✓ 已启用: {ext_name}")
            extension_enabled = True
            break
        except Exception as e:
            print(f"      × {ext_name}: 不可用")
    
    if not extension_enabled:
        print(f"\n[ERROR] 无法启用URDF导入扩展!")
        print(f"\n解决方案:")
        print(f"  1. Isaac Lab pip版本不支持URDF导入")
        print(f"  2. 需要使用完整的Isaac Sim安装")
        print(f"  3. 或使用在线转换工具")
        return False
    
    # 导入URDF
    print(f"\n[4/4] 导入URDF...")
    
    try:
        # Isaac Sim 5.1.0的URDFParseAndImportFile命令参数
        result = omni.kit.commands.execute(
            "URDFParseAndImportFile",
            urdf_path=str(URDF_PATH),
            # 基本参数 (不同版本参数名可能不同)
            import_config=None,  # 使用默认配置
        )
        
        if result:
            print(f"      ✓ URDF导入成功")
        else:
            print(f"      × URDF导入返回False")
            return False
            
    except Exception as e:
        print(f"      × URDF导入失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 保存USD
    print(f"\n[5/5] 保存USD文件...")
    
    try:
        stage = omni.usd.get_context().get_stage()
        if stage:
            stage.Export(str(OUTPUT_PATH))
            print(f"      ✓ USD已保存: {OUTPUT_PATH}")
            print(f"      大小: {OUTPUT_PATH.stat().st_size} bytes")
        else:
            print(f"      × 无法获取USD stage")
            return False
            
    except Exception as e:
        print(f"      × 保存失败: {e}")
        return False
    
    print(f"\n{'='*70}")
    print(f"✓ 转换完成!")
    print(f"{'='*70}\n")
    
    return True


if __name__ == "__main__":
    try:
        success = convert_urdf_to_usd()
        exit_code = 0 if success else 1
    except Exception as e:
        print(f"\n[FATAL ERROR] {e}")
        import traceback
        traceback.print_exc()
        exit_code = 1
    finally:
        simulation_app.close()
        sys.exit(exit_code)
