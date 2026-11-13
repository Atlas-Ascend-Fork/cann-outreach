#!/bin/bash
set -e

export BASE_PATH=$(
  cd "$(dirname $0)"
  pwd
)
export BUILD_PATH="${BASE_PATH}/build"
export BUILD_OUT_PATH="${BASE_PATH}/build_out"

CORE_NUMS=$(cat /proc/cpuinfo | grep "processor" | wc -l)
if [ ${CORE_NUMS} -gt 8 ]; then
  CORE_NUMS=8
fi

usage() {
    cat <<EOF
Usage: bash build.sh [options]
  --soc=<soc>            # 目标 SoC，默认 ascend910b（可选 ascend910b / ascend910_93 / ascend950）
  --soc <soc>            # 同上（双参数写法，等价）
  --clean                # 清理构建产物
  --run-ut               # 构建完成后运行 UT
  --run-example          # 构建完成后运行 example
  -j<N>                  # 指定并行编译线程数
  -h | --help            # 显示帮助
EOF
}

clean_build() {
  if [ -d "${BUILD_PATH}" ]; then
    echo "Cleaning build directory..."
    rm -rf ${BUILD_PATH}/*
  fi
}

clean_build_out() {
  if [ -d "${BUILD_OUT_PATH}" ]; then
    echo "Cleaning build_out directory..."
    rm -rf ${BUILD_OUT_PATH}/*
  fi
}

THREAD_NUM=${CORE_NUMS}
COMPUTE_UNIT="ascend910b"
ENABLE_CLEAN=FALSE
RUN_UT=FALSE
RUN_EXAMPLE=FALSE

while [[ $# -gt 0 ]]; do
  case "$1" in
    --soc=*)
      COMPUTE_UNIT="${1#*=}"
      shift
      ;;
    --soc)
      if [[ -z "${2:-}" || "${2:0:2}" == "--" ]]; then
        echo "[ERROR] --soc requires a value (e.g. --soc=ascend910_93)" >&2
        usage
        exit 1
      fi
      COMPUTE_UNIT="$2"
      shift 2
      ;;
    --clean|--make_clean)
      ENABLE_CLEAN=TRUE
      shift
      ;;
    --run-ut|-u|--ut)
      RUN_UT=TRUE
      shift
      ;;
    --run-example|-e|--example)
      RUN_EXAMPLE=TRUE
      shift
      ;;
    -j*)
      THREAD_NUM="${1:2}"
      if [ -z "$THREAD_NUM" ]; then
        THREAD_NUM=${CORE_NUMS}
      fi
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    -*)
      echo "[ERROR] Invalid option: $1" >&2
      usage
      exit 1
      ;;
    *)
      echo "[ERROR] Unexpected argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

# 校验 soc 合法性
case "${COMPUTE_UNIT}" in
  ascend910b|ascend910_93|ascend950) ;;
  *)
    echo "[ERROR] Unsupported --soc=${COMPUTE_UNIT}" >&2
    usage
    exit 1
    ;;
esac

echo "[INFO] COMPUTE_UNIT=${COMPUTE_UNIT}"

if [ "$ENABLE_CLEAN" = "TRUE" ]; then
  clean_build
  clean_build_out
  exit 0
fi

if [ "$RUN_UT" = "TRUE" ]; then
  echo "[INFO] Running UT tests..."
  cd "${BASE_PATH}/tests/ut"
  ./run.sh
  UT_RESULT=$?
  if [ $UT_RESULT -ne 0 ]; then
    echo "[ERROR] UT tests failed"
    exit 1
  fi
  echo "[INFO] UT tests passed!"
  exit 0
fi

CMAKE_ARGS="-DASCEND_COMPUTE_UNIT=$COMPUTE_UNIT"

if [ ! -d "${BUILD_PATH}" ]; then
  mkdir -p "${BUILD_PATH}"
fi

[ -f "${BUILD_PATH}/CMakeCache.txt" ] && rm -f ${BUILD_PATH}/CMakeCache.txt

echo "----------------------------------------------------------------"
echo "[INFO] Configuring project..."
echo "[INFO] CMAKE_ARGS: ${CMAKE_ARGS}"
cd "${BUILD_PATH}" && cmake ${CMAKE_ARGS} ..

echo "----------------------------------------------------------------"
echo "[INFO] Building project with ${THREAD_NUM} threads..."
cmake --build . --target all binary package install -- -j ${THREAD_NUM}

KERNEL_O=$(find ${BUILD_PATH}/op_kernel/ascendc_kernels/binary/${COMPUTE_UNIT} -name "*.o" 2>/dev/null | head -1)
if [ -z "$KERNEL_O" ]; then
    echo "[ERROR] Kernel binary not found"
    exit 1
fi

PKG_PATH=$(ls "${BUILD_PATH}"/custom_opp_*.run 2>/dev/null | head -n 1)
if [ -z "$PKG_PATH" ] || [ ! -f "$PKG_PATH" ] || [ ! -s "$PKG_PATH" ]; then
    echo "[ERROR] Package not found or empty"
    exit 1
fi

echo "----------------------------------------------------------------"
echo "[INFO] Build completed successfully!"
echo "[INFO] Kernel binary: ${KERNEL_O}"
echo "[INFO] Package: ${PKG_PATH}"

if [ "$RUN_EXAMPLE" = "TRUE" ]; then
  echo "----------------------------------------------------------------"
  echo "[INFO] Running examples..."
  cd "${BASE_PATH}/examples"
  ./run.sh
  EXAMPLE_RESULT=$?
  cd - > /dev/null
  if [ $EXAMPLE_RESULT -ne 0 ]; then
    echo "[ERROR] Example execution failed"
    exit 1
  fi
  echo "[INFO] Example completed successfully!"
fi
