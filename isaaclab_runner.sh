#!/bin/bash

##
# Isaac Lab Script Runner - 源码版本兼容
# 
# 使用方法:
#   ./isaaclab_runner.sh scripts/test_simple_env.py --num_envs 4
#   ./isaaclab_runner.sh scripts/run_rosorin_env.py --headless
##

# Isaac Sim 源码构建路径
ISAAC_SIM_PATH="/home/wujiahao/IsaacSim/_build/linux-x86_64/release"
ISAAC_LAB_PATH="/home/wujiahao/IsaacLab"

# 检查路径
if [ ! -f "$ISAAC_SIM_PATH/python.sh" ]; then
    echo "❌ 错误: Isaac Sim python.sh 未找到"
    echo "   路径: $ISAAC_SIM_PATH/python.sh"
    exit 1
fi

if [ ! -d "$ISAAC_LAB_PATH" ]; then
    echo "❌ 错误: Isaac Lab 路径未找到"
    echo "   路径: $ISAAC_LAB_PATH"
    exit 1
fi

# 设置环境变量
export PYTHONPATH="${ISAAC_LAB_PATH}/source:${PYTHONPATH}"

# 运行脚本
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Isaac Lab 脚本运行器 (源码版本兼容)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📂 Isaac Sim: $ISAAC_SIM_PATH"
echo "📂 Isaac Lab: $ISAAC_LAB_PATH"
echo "🚀 运行脚本: $@"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# 执行
"$ISAAC_SIM_PATH/python.sh" "$@"

EXIT_CODE=$?
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ 脚本执行成功"
else
    echo "❌ 脚本执行失败 (退出码: $EXIT_CODE)"
fi
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

exit $EXIT_CODE
