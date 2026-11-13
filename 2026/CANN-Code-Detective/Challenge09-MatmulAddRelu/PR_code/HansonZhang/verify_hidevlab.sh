#!/usr/bin/env bash
# =============================================================================
# Challenge 09 · CATLASS MatmulAddRelu —— 自验证脚本
#
# 【怎么用】
#   1. 等 950 NPU 资源审批通过，在 HiDevLab 打开你的 Ascend 环境（网页控制台）
#   2. 下面两行代码整个复制粘贴进控制台执行（-b 不能省：本 PR 的文件
#      只在 challenge09-hansonzhang 分支上，默认分支 master 没有它们）：
#
#        git clone -b challenge09-hansonzhang https://gitcode.com/HansonZhang/cann-outreach.git cann-outreach
#        bash cann-outreach/2026/CANN-Code-Detective/Challenge09-MatmulAddRelu/PR_code/HansonZhang/verify_hidevlab.sh
#
#   3. 看到 PASS 后截图，存到本仓库 verify_env/runs/challenge09/pass.png
#      下次 agent 唤醒时会自动发现并补发验证报告，无需通知 agent
#
# 【截图需包含】设备型号(npu-smi) + PASS 字样 + 日期
# =============================================================================
set -uo pipefail

REPO="cann-outreach"
DIR="2026/CANN-Code-Detective/Challenge09-MatmulAddRelu/PR_code/HansonZhang"

echo "================================================"
echo " Challenge 09 · MatmulAddRelu · 自验证"
echo "================================================"
echo "[环境] $(date -Is)"
uname -a

# ---- 1. 确认设备与工具链 -------------------------------------------------
echo; echo "[1/5] 设备探测"
command -v npu-smi >/dev/null 2>&1 && npu-smi info || { echo "✗ 无 npu-smi，本环境不是 Ascend"; exit 1; }

echo; echo "[2/5] 加载 CANN 环境"
if [ -f /usr/local/Ascend/cann/set_env.sh ]; then
  # shellcheck disable=SC1091
  source /usr/local/Ascend/cann/set_env.sh
  echo "  ✓ 已加载 /usr/local/Ascend/cann/set_env.sh"
else
  echo "  ✗ 找不到 set_env.sh，请确认环境为 Ascend 950 + CANN >= 9.1.0"
  exit 1
fi

# ---- 3. 安装依赖（版本严格按任务书，不要改）------------------------------
# 任务书「环境准备」：ascend-catlass-dsl==2.1.0, torch==2.9.0+cpu,
#                    torch-npu==2.9.0.post6, PyYAML；CANN >= 9.1.0
echo; echo "[3/5] 依赖检查"
python3 - <<'PY'
import importlib.util as u, sys
need = {"catlass": "ascend-catlass-dsl", "torch": "torch",
        "torch_npu": "torch-npu", "yaml": "PyYAML"}
miss = []
for mod, pkg in need.items():
    ok = u.find_spec(mod) is not None
    print(f"  {'✓' if ok else '✗'} {pkg} ({mod})")
    if not ok:
        miss.append(pkg)
if miss:
    print("\n  缺失:", ", ".join(miss))
    print("  安装命令（与任务书 README「环境准备」一致，x86_64）：")
    print("    python -m pip install --index-url https://pypi.org/simple \\")
    print("      --find-links https://mirrors.aliyun.com/pytorch-wheels/cpu/ \"torch==2.9.0+cpu\"")
    print("    python -m pip install --index-url https://repo.huaweicloud.com/repository/pypi/simple \\")
    print("      \"torch-npu==2.9.0.post6\" \"ascend-catlass-dsl==2.1.0\" \"PyYAML>=6,<7\"")
    sys.exit(2)
PY
[ $? -eq 0 ] || { echo "依赖不齐，先按上方命令安装后重试"; exit 2; }

# ---- 4. 准备 BC 模板（任务书 README 明确要求，缺了必抛 RuntimeError）-----
echo; echo "[4/5] 生成 Catlass BC 模板"
python3 -m catlass.bc_compile || {
  echo "✗ bc_compile 失败。这一步按任务书 README 是必需的："
  echo "  它生成数据搬运/矩阵乘的 LLVM bitcode 模板，"
  echo "  缺失时编译会抛 'BC not compiled or version mismatch'。"
  exit 3
}

# ---- 5. 跑官方验证脚本 ---------------------------------------------------
echo; echo "[5/5] 执行官方验证脚本"
cd "$REPO" || exit 1
python3 "$DIR/test_matmul_add_relu.py" --device 0
rc=$?

echo
echo "================================================"
if [ $rc -eq 0 ]; then
  echo " 结果: PASS  ← 请截图此区域"
else
  echo " 结果: FAIL (exit=$rc) ← 请把失败输出一并反馈给 agent"
fi
echo " 设备: $(npu-smi info 2>/dev/null | grep -m1 'Product Name' || echo unknown)"
echo " 时间: $(date -Is)"
echo "================================================"
exit $rc