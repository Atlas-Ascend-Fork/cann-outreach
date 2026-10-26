#!/usr/bin/env bash
# =============================================================================
# Challenge 10 · ExpCustom —— 自验证脚本
#
# 【怎么用】（等真实 Ascend 950 / NPU 环境就绪后）
#   1. 在 Ascend 环境（网页控制台或已装 CANN toolkit 的机器）里，
#      下面两行整个复制粘贴执行（-b 不能省：本 PR 的文件只在
#      challenge10-hansonzhang 分支上，默认分支 master 没有它们）：
#
#        git clone -b challenge10-hansonzhang https://gitcode.com/HansonZhang/cann-outreach.git cann-outreach
#        bash cann-outreach/2026/CANN-Code-Detective/Challenge10-ExpCustom/PR_code/personal_HansonZhang_10/verify_hidevlab.sh
#
#   2. 看到 PASS 后截图（含 npu-smi 设备型号 + PASS 字样 + 日期），
#      存到本仓库 verify_env/runs/challenge10/pass.png
#      下次 agent 唤醒时会自动发现并补发验证报告，无需通知 agent
#
# 注：本期是 Ascend C 自定义算子工程（C++），需要 CANN toolkit（含 msopgen/cmake）。
#     第八/九期的 Python 验证脚本不适用于本期。
# =============================================================================
set -uo pipefail

REPO="cann-outreach"
DIR="2026/CANN-Code-Detective/Challenge10-ExpCustom/PR_code/personal_HansonZhang_10"
OP_DIR="$DIR/code/expcustom"

echo "================================================"
echo " Challenge 10 · Exp · 自验证"
echo "================================================"
echo "[环境] $(date -Is)"
uname -a

echo; echo "[1/4] 设备与工具链探测"
if command -v npu-smi >/dev/null 2>&1; then
  npu-smi info | head -15
else
  echo "✗ 无 npu-smi，本环境不是 Ascend"; exit 1
fi

if [ -f /usr/local/Ascend/cann/set_env.sh ]; then
  # shellcheck disable=SC1091
  source /usr/local/Ascend/cann/set_env.sh
  echo "  ✓ CANN 环境已加载"
else
  echo "  ✗ 缺少 /usr/local/Ascend/cann/set_env.sh"
  exit 1
fi

echo; echo "[2/4] 构建工具探测"
missing=0
for t in cmake g++ msopgen; do
  if command -v "$t" >/dev/null 2>&1; then
    echo "  ✓ $t ($(command -v $t))"
  else
    echo "  ✗ $t 缺失"; missing=1
  fi
done
if [ $missing -eq 1 ]; then
  echo
  echo "  自定义算子工程需要 CANN toolkit（含 msopgen）。"
  echo "  若该环境未预装，可改用 CANN 官方社区环境，或告知 agent 换验证方案。"
  exit 2
fi

echo; echo "[3/4] 构建算子工程"
if [ ! -d "$REPO/$OP_DIR" ]; then
  echo "✗ 找不到算子目录 $REPO/$OP_DIR"
  echo "  确认 clone 时带了 -b challenge10-hansonzhang"
  exit 1
fi
cd "$REPO/$OP_DIR" || exit 1
if [ ! -f ./build.sh ]; then
  echo "✗ 未找到 build.sh"
  exit 2
fi
# 注意：裸的 `bash build.sh` 只编译、不跑任何测试，输出里不会有 PASS。
# -u 跑算子 UT，-e 跑 aclnn 调用示例（需 NPU）。
# 模板里的 .sh 都是 100644（无执行位），build.sh 内部用 ./run.sh 调用，
# 不先补执行位会直接 Permission denied。
chmod +x tests/ut/run.sh examples/run.sh 2>/dev/null || true
bash ./build.sh -j8
BUILD_RC=$?
[ $BUILD_RC -eq 0 ] || { echo "✗ 构建失败，请把完整日志反馈给 agent"; exit 3; }
echo "  ✓ 构建完成"

echo; echo "[4/4] 运行 aclnn 调用示例（官方验证路径）"
bash ./build.sh -e
EX_RC=$?
echo
if [ $EX_RC -eq 0 ]; then echo "--- aclnn 示例结果: PASS"; else echo "--- aclnn 示例结果: FAIL exit=$EX_RC"; fi

echo
echo "================================================"
if [ $EX_RC -eq 0 ]; then
  echo " 结果: PASS  ← 请截图此区域"
else
  echo " 结果: FAIL ← 请把上方完整输出反馈给 agent"
  echo " 提示 1：算子定义里 COMPUTE_UNIT=ascend910b（沿用模板，未改动）。"
  echo "         若在 Ascend 950 环境因 SoC 名不匹配而构建失败，请反馈 npu-smi 的产品名。"
  echo " 提示 2：examples/test_aclnn_exp.cpp 是官方验证脚本，若它失败，"
  echo "         请把完整 stdout 一并反馈（可能是输入 shape/精度或环境问题）。"
fi
echo " 设备: $(npu-smi info 2>/dev/null | grep -m1 'Product Name' || echo unknown)"
echo " 时间: $(date -Is)"
echo "================================================"
exit $EX_RC
