#!/usr/bin/env bash
# =============================================================================
# Challenge 08 · TanhCustom —— 自验证脚本
#
# 【怎么用】
#   1. 在 HiDevLab 打开 Ascend 环境（网页控制台）
#   2. 下面两行整个复制粘贴执行（-b 不能省：本 PR 的文件只在
#      challenge08-hansonzhang 分支上，默认分支 master 没有它们）：
#
#        git clone -b challenge08-hansonzhang https://gitcode.com/HansonZhang/cann-outreach.git cann-outreach
#        bash cann-outreach/2026/CANN-Code-Detective/Challenge08-TanhCustom/PR_code/personal_HansonZhang_08/verify_hidevlab.sh
#
#   3. 看到 PASS 后截图，存到本仓库 verify_env/runs/challenge08/pass.png
#
# 注：本期是 Ascend C 自定义算子工程（C++），需要 msopgen/CMake 构建，
#     比 Challenge 09 的纯 Python 复杂。若环境未装 CANN toolkit，
#     脚本会给出明确的缺失项提示。
# =============================================================================
set -uo pipefail

REPO="cann-outreach"
DIR="2026/CANN-Code-Detective/Challenge08-TanhCustom/PR_code/personal_HansonZhang_08"

echo "================================================"
echo " Challenge 08 · TanhCustom · 自验证"
echo "================================================"
echo "[环境] $(date -Is)"

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
cd "$REPO/$DIR/code/challenge08_tanhcustom" || exit 1
if [ ! -f ./build.sh ]; then
  echo "未找到 build.sh，按任务书 README 的构建步骤手动执行"
  exit 2
fi
# 注意：裸的 `bash build.sh` 只编译、不跑任何测试，输出里不会有 PASS。
# -u 跑算子 UT（op_host + kernel + 精度比对），-e 跑 aclnn 调用示例（需 NPU）。
# 模板里的 .sh 都是 100644（无执行位），build.sh 内部用 `./run.sh` 调用，
# 不先补执行位会直接 Permission denied。
chmod +x tests/ut/run.sh examples/run.sh 2>/dev/null || true
bash ./build.sh -j8
[ $? -eq 0 ] || { echo "✗ 构建失败，请把完整日志反馈给 agent"; exit 3; }

echo; echo "[4/4] 运行单元测试与调用示例"
bash ./build.sh -u
UT_RC=$?
echo
if [ $UT_RC -eq 0 ]; then echo "--- UT 结果: PASS"; else echo "--- UT 结果: FAIL exit=$UT_RC"; fi

bash ./build.sh -e
EX_RC=$?
echo
if [ $EX_RC -eq 0 ]; then echo "--- aclnn 示例结果: PASS"; else echo "--- aclnn 示例结果: FAIL exit=$EX_RC"; fi

echo
echo "================================================"
if [ $UT_RC -eq 0 ] && [ $EX_RC -eq 0 ]; then
  echo " 结果: PASS  ← 请截图此区域"
else
  echo " 结果: FAIL ← 请把上方完整输出反馈给 agent"
  echo " 提示：工程模板给定 COMPUTE_UNIT=ascend910b（未改动模板）。"
  echo "       若在 Ascend 950 环境因 SoC 不匹配而构建失败，请反馈产品名。"
fi
echo " 设备: $(npu-smi info 2>/dev/null | grep -m1 'Product Name' || echo unknown)"
echo " 时间: $(date -Is)"
echo "================================================"