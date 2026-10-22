#!/bin/bash
# add0123 一键构建 + 测试脚本
# 用法: bash scripts/run_test.sh [SOC_VERSION] [CANN路径]
set -e
script_dir=$(cd "$(dirname "$0")" && pwd)
root_dir=$(dirname "$script_dir")

echo "=========================================="
echo "== [1/2] 编译算子工程"
echo "=========================================="
bash "$root_dir/build.sh" "$@"

echo ""
echo "=========================================="
echo "== [2/2] 运行 Python 单元测试"
echo "=========================================="
python3 "$script_dir/test_add0123.py"

echo ""
echo "全部通过: 构建成功 + 单元测试通过"
