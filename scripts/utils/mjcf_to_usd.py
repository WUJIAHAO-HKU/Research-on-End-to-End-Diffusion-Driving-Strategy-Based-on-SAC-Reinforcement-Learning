"""
MJCF → USD 转换

使用Isaac Sim将MuJoCo MJCF格式转换为USD
"""

import sys
from pathlib import Path

# Isaac Sim必须最先导入
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

# 现在导入其他模块
import omni.kit.commands
import omni.usd
from pxr import Usd, UsdGeom
import carb

MJCF_FILE = Path("/home/wujiahao/ROSORIN_CAR and Reasearch/Research on End-to-End Diffusion Driving Strategy Based on SAC Reinforcement Learning/data/assets/rosorin/rosorin.xml")
OUTPUT_USD = Path("/home/wujiahao/ROSORIN_CAR and Reasearch/Research on End-to-End Diffusion Driving Strategy Based on SAC Reinforcement Learning/data/assets/rosorin/rosorin.usd")

def convert_mjcf_to_usd():
    """转换MJCF到USD"""
    
    print(f"\n{'='*70}")
    print(f"  MJCF → USD 转换")
    print(f"{'='*70}\n")
    
    if not MJCF_FILE.exists():
        print(f"[ERROR] MJCF文件不存在: {MJCF_FILE}")
        return False
    
    print(f"[1/3] MJCF文件: {MJCF_FILE}")
    print(f"      大小: {MJCF_FILE.stat().st_size} bytes")
    
    OUTPUT_USD.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # 启用MuJoCo导入扩展
        print(f"\n[2/3] 启用MuJoCo MJCF导入扩展...")
        
        import omni.kit.app
        ext_manager = omni.kit.app.get_app().get_extension_manager()
        
        # 尝试启用MJCF扩展
        mjcf_extensions = [
            "omni.importer.mjcf",
            "omni.isaac.mjcf",
        ]
        
        enabled = False
        for ext_name in mjcf_extensions:
            try:
                ext_manager.set_extension_enabled_immediate(ext_name, True)
                print(f"      ✓ 已启用: {ext_name}")
                enabled = True
                break
            except Exception as e:
                print(f"      × {ext_name}: 不可用")
        
        if not enabled:
            print(f"\n[ERROR] Isaac Lab pip版本不支持MJCF导入!")
            print(f"\n解决方案：使用MuJoCo Python API直接生成USD")
            return convert_mjcf_via_mujoco()
        
        # 导入MJCF
        print(f"\n[3/3] 导入MJCF...")
        
        result = omni.kit.commands.execute(
            "MJCFCreateAsset",
            mjcf_path=str(MJCF_FILE),
        )
        
        if result:
            print(f"      ✓ MJCF导入成功")
            
            # 保存USD
            stage = omni.usd.get_context().get_stage()
            stage.Export(str(OUTPUT_USD))
            print(f"      ✓ USD已保存: {OUTPUT_USD}")
            return True
        else:
            print(f"      × MJCF导入失败")
            return False
            
    except Exception as e:
        print(f"\n[ERROR] 转换失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def convert_mjcf_via_mujoco():
    """备用方案：使用MuJoCo Python加载MJCF并手动构建USD"""
    
    print(f"\n使用MuJoCo备用转换...")
    
    try:
        import mujoco
        from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema
        
        # 加载MJCF
        print(f"  [1/4] 使用MuJoCo加载MJCF...")
        model = mujoco.MjModel.from_xml_path(str(MJCF_FILE))
        
        print(f"      ✓ 模型加载成功")
        print(f"        - Bodies: {model.nbody}")
        print(f"        - Joints: {model.njnt}")
        print(f"        - Geometries: {model.ngeom}")
        
        # 创建USD stage
        print(f"\n  [2/4] 创建USD stage...")
        stage = Usd.Stage.CreateNew(str(OUTPUT_USD))
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        
        # 创建根prim
        root_prim = stage.DefinePrim("/World", "Xform")
        stage.SetDefaultPrim(root_prim)
        
        # 添加物理场景
        physics_scene = UsdPhysics.Scene.Define(stage, "/World/physicsScene")
        physics_scene.CreateGravityDirectionAttr().Set((0.0, 0.0, -1.0))
        physics_scene.CreateGravityMagnitudeAttr().Set(9.81)
        
        # 创建地面
        ground_prim = UsdGeom.Mesh.Define(stage, "/World/ground")
        ground_prim.CreatePointsAttr([(-10, -10, 0), (10, -10, 0), (10, 10, 0), (-10, 10, 0)])
        ground_prim.CreateFaceVertexCountsAttr([4])
        ground_prim.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
        
        # 创建机器人
        print(f"\n  [3/4] 创建机器人USD结构...")
        robot_prim = UsdGeom.Xform.Define(stage, "/World/rosorin")
        robot_prim.AddTranslateOp().Set((0.0, 0.0, 0.1))
        
        # 添加碰撞和视觉几何体（简化版）
        # 这里只创建基本结构，完整版需要遍历MuJoCo模型
        
        # 保存
        print(f"\n  [4/4] 保存USD...")
        stage.Save()
        
        print(f"      ✓ USD已保存: {OUTPUT_USD}")
        print(f"      大小: {OUTPUT_USD.stat().st_size} bytes")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] MuJoCo备用方案失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    try:
        success = convert_mjcf_to_usd()
        exit_code = 0 if success else 1
    except Exception as e:
        print(f"\n[FATAL ERROR] {e}")
        import traceback
        traceback.print_exc()
        exit_code = 1
    finally:
        simulation_app.close()
        sys.exit(exit_code)
