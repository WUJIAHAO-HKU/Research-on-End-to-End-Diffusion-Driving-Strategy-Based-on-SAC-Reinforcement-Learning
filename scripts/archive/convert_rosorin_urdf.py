"""
将ROSOrin URDF转换为USD格式

Isaac Lab需要USD格式的机器人模型
"""

import argparse
from pathlib import Path

# 解析参数
parser = argparse.ArgumentParser(description="转换ROSOrin URDF到USD")
parser.add_argument(
    "--urdf_path",
    type=str,
    default="/home/wujiahao/ROSORIN_CAR and Reasearch/ROSOrin智能视觉小车/1 教程资料/9 Gazebo仿真/资源文件/3 功能包文件/rosorin_description",
    help="ROSOrin描述包路径"
)
parser.add_argument(
    "--output_dir",
    type=str,
    default="data/assets/rosorin",
    help="输出USD文件目录"
)
parser.add_argument("--headless", action="store_true", help="Run in headless mode")
args = parser.parse_args()

# 启动Isaac Sim (直接使用SimulationApp)
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})  # URDF转换使用headless模式

# 导入Isaac Sim模块 (必须在SimulationApp之后)
import omni
import omni.kit.commands
import carb
from pxr import Usd
import isaaclab.sim as sim_utils

def convert_urdf_to_usd():
    """转换URDF到USD"""
    
    print(f"\n{'='*70}")
    print(f"  ROSOrin URDF → USD 转换")
    print(f"{'='*70}\n")
    
    # URDF路径 - 使用实际的URDF文件而不是xacro
    urdf_dir = Path(args.urdf_path) / "urdf"
    
    # 查找URDF文件
    urdf_files = list(urdf_dir.glob("*.urdf"))
    if not urdf_files:
        # 如果没有.urdf，尝试使用rosorin_full.urdf
        urdf_path = Path("/home/wujiahao/ROSORIN_CAR and Reasearch/Research on End-to-End Diffusion Driving Strategy Based on SAC Reinforcement Learning/rosorin_ws/rosorin_full.urdf")
        if not urdf_path.exists():
            print(f"[ERROR] 找不到URDF文件")
            print(f"  查找路径1: {urdf_dir}/*.urdf")
            print(f"  查找路径2: {urdf_path}")
            return False
    else:
        urdf_path = urdf_files[0]
    
    print(f"[1/5] 找到URDF文件: {urdf_path}")
    
    # 输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_usd = output_dir / "rosorin.usd"
    
    print(f"[2/5] 输出路径: {output_usd}")
    
    try:
        # 方法1: 尝试使用Isaac Sim的URDF导入器
        print(f"[3/5] 尝试导入URDF扩展...")
        
        import omni.kit.app
        ext_manager = omni.kit.app.get_app().get_extension_manager()
        
        # 尝试启用URDF导入扩展 (可能不存在)
        urdf_ext_names = [
            "isaacsim.asset.importer.urdf",  # 新版本
            "omni.importer.urdf",            # 旧版本
        ]
        
        urdf_ext_enabled = False
        for ext_name in urdf_ext_names:
            try:
                ext_manager.set_extension_enabled_immediate(ext_name, True)
                print(f"✓ 启用扩展: {ext_name}")
                urdf_ext_enabled = True
                break
            except Exception as e:
                print(f"  - 扩展 {ext_name} 不可用")
                continue
        
        if not urdf_ext_enabled:
            print(f"\n[WARNING] URDF导入扩展不可用")
            print(f"  源码构建版本可能缺少URDF导入功能")
            print(f"  建议:")
            print(f"    1. 等待二进制版本下载完成")
            print(f"    2. 或手动创建USD文件")
            print(f"    3. 或使用Isaac Lab的机器人配置API")
            return False
        
        print(f"[4/5] 导入URDF...")
        
        # 尝试导入 - 使用最简单的参数
        result = omni.kit.commands.execute(
            "URDFParseAndImportFile",
            urdf_path=str(urdf_path),
        )
        
        if not result or not result[0]:
            print(f"[ERROR] URDF导入失败")
            return False
        
        print(f"✓ URDF导入成功")
        
        # 保存USD
        print(f"[5/5] 保存USD文件...")
        stage = omni.usd.get_context().get_stage()
        stage.Export(str(output_usd))
        
        print(f"\n{'='*70}")
        print(f"✓ 转换成功!")
        print(f"  USD文件: {output_usd}")
        print(f"  大小: {output_usd.stat().st_size / 1024:.1f} KB")
        print(f"{'='*70}\n")
        
        return True
        
    except Exception as e:
        print(f"\n[ERROR] 转换失败: {e}")
        import traceback
        traceback.print_exc()
        
        print(f"\n{'='*70}")
        print(f"备选方案:")
        print(f"{'='*70}")
        print(f"1. 使用二进制版本的Isaac Sim (推荐)")
        print(f"   - 完整的URDF导入支持")
        print(f"")
        print(f"2. 手动创建USD机器人模型")
        print(f"   - 使用Isaac Lab的ArticulationCfg")
        print(f"   - 定义关节和连杆")
        print(f"")
        print(f"3. 使用现有的机器人模型")
        print(f"   - 先用Carter/Turtlebot测试环境")
        print(f"   - 稍后替换为ROSOrin")
        print(f"{'='*70}\n")
        
        return False


if __name__ == "__main__":
    try:
        success = convert_urdf_to_usd()
        exit(0 if success else 1)
    finally:
        simulation_app.close()
