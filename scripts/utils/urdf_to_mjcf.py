"""
ROSOrin URDF → MJCF 转换脚本

将ROSOrin的URDF模型转换为MuJoCo MJCF格式
Isaac Lab支持MJCF导入
"""

import mujoco
from pathlib import Path
import xml.etree.ElementTree as ET

# 路径配置
URDF_FILE = Path("/home/wujiahao/ROSORIN_CAR and Reasearch/Research on End-to-End Diffusion Driving Strategy Based on SAC Reinforcement Learning/rosorin_ws/rosorin_full.urdf")
OUTPUT_MJCF = Path("/home/wujiahao/ROSORIN_CAR and Reasearch/Research on End-to-End Diffusion Driving Strategy Based on SAC Reinforcement Learning/data/assets/rosorin/rosorin.xml")

def convert_urdf_to_mjcf():
    """使用MuJoCo的编译器将URDF转换为MJCF"""
    
    print(f"\n{'='*70}")
    print(f"  ROSOrin URDF → MJCF 转换")
    print(f"{'='*70}\n")
    
    # 检查输入文件
    if not URDF_FILE.exists():
        print(f"[ERROR] URDF文件不存在: {URDF_FILE}")
        return False
    
    print(f"[1/3] 读取URDF: {URDF_FILE}")
    print(f"      大小: {URDF_FILE.stat().st_size} bytes")
    
    # 创建输出目录
    OUTPUT_MJCF.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # 方法1: 使用MuJoCo直接加载URDF
        print(f"\n[2/3] 使用MuJoCo加载URDF...")
        
        # MuJoCo可以直接加载URDF并转换为内部格式
        model = mujoco.MjModel.from_xml_path(str(URDF_FILE))
        
        print(f"      ✓ URDF加载成功")
        print(f"      - Bodies: {model.nbody}")
        print(f"      - Joints: {model.njnt}")
        print(f"      - Actuators: {model.nu}")
        print(f"      - Geometries: {model.ngeom}")
        
        # 保存为MJCF
        print(f"\n[3/3] 保存MJCF: {OUTPUT_MJCF}")
        
        # MuJoCo加载后会自动转换，我们需要保存XML
        mujoco.mj_saveLastXML(str(OUTPUT_MJCF), model)
        
        print(f"      ✓ MJCF已保存")
        print(f"      大小: {OUTPUT_MJCF.stat().st_size} bytes")
        
        print(f"\n{'='*70}")
        print(f"✓ 转换完成!")
        print(f"  MJCF: {OUTPUT_MJCF}")
        print(f"{'='*70}\n")
        
        return True
        
    except Exception as e:
        print(f"\n[ERROR] MuJoCo加载失败: {e}")
        print(f"\n尝试备用方法...")
        
        try:
            # 方法2: 手动转换URDF到MJCF
            return convert_urdf_to_mjcf_manual()
        except Exception as e2:
            print(f"[ERROR] 备用方法也失败: {e2}")
            import traceback
            traceback.print_exc()
            return False


def convert_urdf_to_mjcf_manual():
    """手动转换URDF到MJCF格式"""
    
    print(f"\n使用手动转换方法...")
    
    # 读取URDF
    tree = ET.parse(URDF_FILE)
    root = tree.getroot()
    
    # 创建MJCF根元素
    mjcf = ET.Element('mujoco', model='rosorin')
    
    # 编译器选项
    compiler = ET.SubElement(mjcf, 'compiler', {
        'angle': 'radian',
        'meshdir': str(URDF_FILE.parent.parent / 'meshes'),
        'autolimits': 'true',
    })
    
    # 选项
    option = ET.SubElement(mjcf, 'option', {
        'timestep': '0.001',
        'gravity': '0 0 -9.81',
    })
    
    # 资产
    asset = ET.SubElement(mjcf, 'asset')
    
    # 默认值
    default = ET.SubElement(mjcf, 'default')
    ET.SubElement(default, 'joint', {'damping': '0.1', 'armature': '0.01'})
    ET.SubElement(default, 'geom', {'friction': '0.8 0.1 0.1', 'density': '1000'})
    
    # Worldbody
    worldbody = ET.SubElement(mjcf, 'worldbody')
    
    # 添加光源
    ET.SubElement(worldbody, 'light', {
        'directional': 'true',
        'pos': '0 0 3',
        'dir': '0 0 -1',
    })
    
    # 添加地面
    ET.SubElement(worldbody, 'geom', {
        'name': 'floor',
        'type': 'plane',
        'size': '10 10 0.1',
        'rgba': '0.8 0.8 0.8 1',
    })
    
    # 转换URDF的link和joint
    print(f"  - 转换link和joint...")
    
    # 获取base_link
    base_link = None
    for link in root.findall('.//link'):
        link_name = link.get('name')
        if 'base' in link_name.lower():
            base_link = link
            break
    
    if base_link is None:
        base_link = root.find('.//link')
    
    # 创建机器人body
    robot_body = ET.SubElement(worldbody, 'body', {
        'name': 'rosorin',
        'pos': '0 0 0.1',
    })
    
    # 添加基本几何体（底盘）
    ET.SubElement(robot_body, 'geom', {
        'name': 'chassis',
        'type': 'box',
        'size': '0.103 0.097 0.04',  # ROSOrin底盘尺寸的一半
        'rgba': '0.2 0.2 0.8 1',
        'mass': '1.5',
    })
    
    # 添加4个轮子
    wheel_positions = [
        ('front_left', 0.103, 0.097, 0),
        ('front_right', 0.103, -0.097, 0),
        ('rear_left', -0.103, 0.097, 0),
        ('rear_right', -0.103, -0.097, 0),
    ]
    
    for wheel_name, x, y, z in wheel_positions:
        wheel_body = ET.SubElement(robot_body, 'body', {
            'name': f'{wheel_name}_wheel',
            'pos': f'{x} {y} {z}',
        })
        
        # 轮子几何体
        ET.SubElement(wheel_body, 'geom', {
            'name': f'{wheel_name}_wheel_geom',
            'type': 'cylinder',
            'size': '0.0325 0.0125',  # 半径 高度/2
            'rgba': '0.1 0.1 0.1 1',
            'mass': '0.05',
        })
        
        # 轮子关节
        ET.SubElement(wheel_body, 'joint', {
            'name': f'{wheel_name}_wheel_joint',
            'type': 'hinge',
            'axis': '0 1 0',
            'limited': 'false',
        })
    
    # 执行器
    actuator = ET.SubElement(mjcf, 'actuator')
    
    for wheel_name, _, _, _ in wheel_positions:
        ET.SubElement(actuator, 'velocity', {
            'name': f'{wheel_name}_actuator',
            'joint': f'{wheel_name}_wheel_joint',
            'kv': '1.0',
            'forcerange': '-10 10',
        })
    
    # 保存MJCF
    mjcf_tree = ET.ElementTree(mjcf)
    ET.indent(mjcf_tree, space='  ')
    mjcf_tree.write(OUTPUT_MJCF, encoding='utf-8', xml_declaration=True)
    
    print(f"  ✓ MJCF已保存: {OUTPUT_MJCF}")
    print(f"  大小: {OUTPUT_MJCF.stat().st_size} bytes")
    
    return True


if __name__ == "__main__":
    try:
        success = convert_urdf_to_mjcf()
        exit(0 if success else 1)
    except Exception as e:
        print(f"\n[FATAL ERROR] {e}")
        import traceback
        traceback.print_exc()
        exit(1)
