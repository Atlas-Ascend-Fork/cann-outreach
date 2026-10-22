// Copyright (C) 2026.
// Gcd123 Host 侧实现：
//   - namespace ge     : InferShape（broadcast 形状推导）/ InferDataType
//   - namespace optiling: TilingFunc（计算 TilingData）
//   - namespace ops    : 算子原型注册（由 msOpGen 自动生成，保持生成内容不变）
//
// 使用方式（推荐）：
//   1) msopgen gen -i gcd123.json -c ai_core-<soc_version> -lan cpp -out Gcd123
//      生成工程骨架（含 op_host/gcd123.cpp 的 namespace ops 注册块、CMake、build.sh 等）。
//   2) 用本文件中 namespace ge / namespace optiling 的实现替换骨架中对应块。
//   3) 用本仓库 op_kernel/gcd123_tiling.h、op_kernel/gcd123.cpp 覆盖骨架同名文件。
//   4) ./build.sh 编译，再安装 build_out 下的 .run。
//
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"

#include "../op_kernel/gcd123_tiling.h"

namespace optiling {

constexpr uint32_t GCD123_NUM_BLOCKS = 20;   // 使用的 AI Core 数，按目标芯片核数调整
constexpr uint32_t GCD123_CHUNK_LEN = 1024;  // 每行内分块元素数

static ge::graphStatus TilingFunc(gert::TilingContext *context) {
    const ge::TensorDesc *selfDesc = context->GetInputShape(0);
    const ge::TensorDesc *otherDesc = context->GetInputShape(1);
    const ge::TensorDesc *outDesc = context->GetOutputShape(0);
    if (selfDesc == nullptr || otherDesc == nullptr || outDesc == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const ge::Shape &sShape = selfDesc->GetOriginShape();
    const ge::Shape &oShape = otherDesc->GetOriginShape();
    const ge::Shape &rShape = outDesc->GetOriginShape(); // 已由 InferShape 推导为 broadcast 形状

    Gcd123TilingData *tiling = context->GetTilingData<Gcd123TilingData>();
    tiling->selfN4 = (uint32_t)sShape.GetDimSize(0);
    tiling->selfN3 = (uint32_t)sShape.GetDimSize(1);
    tiling->selfN2 = (uint32_t)sShape.GetDimSize(2);
    tiling->selfN1 = (uint32_t)sShape.GetDimSize(3);
    tiling->otherN4 = (uint32_t)oShape.GetDimSize(0);
    tiling->otherN3 = (uint32_t)oShape.GetDimSize(1);
    tiling->otherN2 = (uint32_t)oShape.GetDimSize(2);
    tiling->otherN1 = (uint32_t)oShape.GetDimSize(3);
    tiling->outN4 = (uint32_t)rShape.GetDimSize(0);
    tiling->outN3 = (uint32_t)rShape.GetDimSize(1);
    tiling->outN2 = (uint32_t)rShape.GetDimSize(2);
    tiling->outN1 = (uint32_t)rShape.GetDimSize(3);

    tiling->totalSelf = tiling->selfN4 * tiling->selfN3 * tiling->selfN2 * tiling->selfN1;
    tiling->totalOther = tiling->otherN4 * tiling->otherN3 * tiling->otherN2 * tiling->otherN1;
    tiling->totalOut = tiling->outN4 * tiling->outN3 * tiling->outN2 * tiling->outN1;
    tiling->totalLines = tiling->outN4 * tiling->outN3 * tiling->outN2;
    tiling->chunkLen = GCD123_CHUNK_LEN;

    context->SetSimdNumBlocks(GCD123_NUM_BLOCKS);

    size_t *workspace = context->GetWorkspaceSizes(1);
    workspace[0] = 0; // 本算子不需要额外 workspace

    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

namespace ge {

namespace {
// 右对齐 broadcast：a、b 逐维需满足相等或其中一方为 1，否则返回 false。
bool TryBroadcastShape(const ge::Shape &a, const ge::Shape &b, ge::Shape &out) {
    const int32_t dimsA = a.GetDimensions();
    const int32_t dimsB = b.GetDimensions();
    const int32_t dims = (dimsA > dimsB) ? dimsA : dimsB;
    out = ge::Shape(dims);
    const int32_t offA = dims - dimsA;
    const int32_t offB = dims - dimsB;
    for (int32_t i = 0; i < dims; ++i) {
        const int64_t da = (i < offA) ? 1 : a.GetDimSize(i - offA);
        const int64_t db = (i < offB) ? 1 : b.GetDimSize(i - offB);
        if (da != db && da != 1 && db != 1) {
            return false;
        }
        out.SetDimSize(i, da > db ? da : db);
    }
    return true;
}
} // namespace

// 输出形状 = self 与 other 的 broadcast 形状。
ge::graphStatus InferShape(OpDesc *op) {
    const TensorDesc *selfDesc = op->GetInputTensorDesc(0);
    const TensorDesc *otherDesc = op->GetInputTensorDesc(1);
    TensorDesc *outDesc = op->GetOutputTensorDesc(0);
    if (selfDesc == nullptr || otherDesc == nullptr || outDesc == nullptr) {
        return ge::GRAPH_FAILED;
    }
    ge::Shape outShape;
    if (!TryBroadcastShape(selfDesc->GetOriginShape(), otherDesc->GetOriginShape(), outShape)) {
        return ge::GRAPH_FAILED;
    }
    outDesc->SetOriginShape(outShape);
    return ge::GRAPH_SUCCESS;
}

// 输出数据类型 = 输入数据类型（本算子仅支持 float16）。
ge::graphStatus InferDataType(OpDesc *op) {
    const TensorDesc *selfDesc = op->GetInputTensorDesc(0);
    TensorDesc *outDesc = op->GetOutputTensorDesc(0);
    if (selfDesc == nullptr || outDesc == nullptr) {
        return ge::GRAPH_FAILED;
    }
    outDesc->SetDataType(selfDesc->GetDataType());
    return ge::GRAPH_SUCCESS;
}
} // namespace ge

namespace ops {
// ============================================================================
// 算子原型注册：此块由 msOpGen 依据 gcd123.json 自动生成。
// 请勿手工修改/手写 —— 直接保留 msOpGen 在 op_host/gcd123.cpp 中生成的 namespace ops
// 注册代码（包含 Gcd123 的原型、输入输出、InferShape/InferDataType/TilingFunc 绑定等）。
// 上方 namespace ge / namespace optiling 为需要实现/替换的部分。
// ============================================================================
} // namespace ops
