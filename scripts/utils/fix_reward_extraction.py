"""
快速修复所有baseline训练脚本的奖励提取逻辑

此脚本会：
1. 在每个env.step()后立即调用extract_reward_components()
2. 将奖励组件保存到episode结束时
3. 更新打印输出以显示即时奖励细节
"""

import re
import os
from pathlib import Path

# 定义要修复的文件列表
SCRIPTS_TO_FIX = [
    "train_ppo.py",
    "train_td3.py",
    "train_sac_gaussian.py",
    "train_dagger.py",
    "train_sac_diffusion_simple.py"
]

SCRIPTS_DIR = Path(__file__).parent

# 要插入的代码片段（在env.step()后）
EXTRACTION_CODE = """
        # 提取当前步的奖励细节（在step之后立即提取）
        current_reward_components = extract_reward_components(env)
"""

def fix_file(filepath):
    """修复单个文件的奖励提取逻辑"""
    print(f"\n处理文件: {filepath.name}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 检查是否已经有extract_reward_components函数
    if 'def extract_reward_components' not in content:
        print(f"  ⚠️ 缺少extract_reward_components函数，跳过")
        return False
    
    # 查找env.step()的模式
    # 匹配: next_obs_dict, rewards, terminated, truncated, infos = env.step(actions)
    step_pattern = re.compile(
        r'([ \t]*)next_obs_dict,\s*rewards,\s*terminated,\s*truncated,\s*infos\s*=\s*env\.step\(actions\)\s*\n'
        r'([ \t]*)next_obs\s*=\s*next_obs_dict\["policy"\]\s*\n'
        r'([ \t]*)next_obs\s*=\s*torch\.nan_to_num\(.*?\)\s*\n'
    )
    
    matches = list(step_pattern.finditer(content))
    
    if not matches:
        print(f"  ⚠️ 未找到env.step()模式，跳过")
        return False
    
    print(f"  找到 {len(matches)} 个env.step()调用")
    
    # 从后向前替换，避免索引变化
    for match in reversed(matches):
        # 检查下一行是否已经是extract_reward_components调用
        next_line_start = match.end()
        next_lines = content[next_line_start:next_line_start+200]
        
        if 'current_reward_components = extract_reward_components(env)' in next_lines:
            print(f"    已有提取代码，跳过此处")
            continue
        
        # 插入提取代码
        indent = match.group(1)
        insertion = f"\n{indent}# 提取当前步的奖励细节（在step之后立即提取）\n{indent}current_reward_components = extract_reward_components(env)\n"
        content = content[:match.end()] + insertion + content[match.end():]
        print(f"    ✓ 插入了奖励提取代码")
    
    # 现在修复episode结束时的奖励保存逻辑
    # 查找: if dones[i]: 或 if dones[i] > 0:
    # 然后找到其中的"提取奖励细节"部分并替换
    
    old_extraction_pattern = re.compile(
        r'([ \t]*)# 提取奖励细节\s*\n'
        r'\1if "log" in infos:.*?'
        r'(?=\n\1(?:# 记录成功率|episode_reward))',
        re.DOTALL
    )
    
    new_extraction = '''# 保存当前episode的奖励组件
                for key, value in current_reward_components.items():
                    if key in reward_components:
                        reward_components[key].append(value)
                
                # 记录成功率（基于是否到达目标）
                if episode_reward[i] > 50:  # 如果总奖励超过50（说明获得了goal_reached奖励）
                    episode_successes.append(1)
                else:
                    episode_successes.append(0)
                
                '''
    
    if old_extraction_pattern.search(content):
        content = old_extraction_pattern.sub(lambda m: m.group(1) + new_extraction, content)
        print(f"  ✓ 更新了奖励保存逻辑")
    
    # 保存修改
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"  ✓ 文件修复完成")
    return True

def main():
    """主函数"""
    print("=" * 80)
    print("开始修复所有baseline训练脚本的奖励提取逻辑")
    print("=" * 80)
    
    fixed_count = 0
    for script_name in SCRIPTS_TO_FIX:
        filepath = SCRIPTS_DIR / script_name
        if not filepath.exists():
            print(f"\n⚠️ 文件不存在: {script_name}")
            continue
        
        if fix_file(filepath):
            fixed_count += 1
    
    print("\n" + "=" * 80)
    print(f"修复完成！成功修复 {fixed_count}/{len(SCRIPTS_TO_FIX)} 个文件")
    print("=" * 80)

if __name__ == "__main__":
    main()
