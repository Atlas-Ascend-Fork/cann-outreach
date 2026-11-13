// Host侧Tiling实现
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/mul_tiling.h"
#include "../op_kernel/tiling_key_mul.h"

#include <algorithm>

namespace {
    constexpr int64_t ALIGN_BYTES = 32;    // GM<->UB 搬运的最小对齐粒度：32 Byte
    constexpr int64_t BUFFER_NUM = 2;      // 双缓冲，与 kernel 侧一致
    constexpr int64_t UB_RESERVED = 1024;  // 预留 UB，避免踩到框架占用区

    inline int64_t CeilDiv(int64_t a, int64_t b) { return (a + b - 1) / b; }
    inline int64_t AlignUp(int64_t v, int64_t align) { return CeilDiv(v, align) * align; }
    inline int64_t AlignDown(int64_t v, int64_t align) { return (v / align) * align; }
}  // namespace

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context) {
    // 1. 平台信息
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    int32_t coreNum = platform.GetCoreNumAiv();   // 纯 Vector 算子用 AIV 核
    if (coreNum <= 0) {
        coreNum = 1;
    }
    uint64_t ubSize = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);

    // 2. 输入信息
    const gert::Tensor *tensor_x = context->GetRequiredInputTensor(0);
    const gert::Tensor *tensor_y = context->GetRequiredInputTensor(1);
    if (tensor_x == nullptr || tensor_y == nullptr) {
        return ge::GRAPH_FAILED;
    }
    ge::DataType dtype = tensor_x->GetDataType();
    int64_t dtypeSize = ge::GetSizeByDataType(dtype);          // float=4, float16=2
    uint32_t totalLength = static_cast<uint32_t>(tensor_x->GetShapeSize());
    int64_t alignNum = ALIGN_BYTES / dtypeSize;                // 32B 对应元素个数

    // 3. tiling key：决定 kernel 实例化哪个 dtype 模板
    ASCENDC_TPL_SEL_PARAM(context, static_cast<uint32_t>(dtype));

    MulTilingData *tiling = context->GetTilingData<MulTilingData>();

    // 4. 空张量保护
    if (totalLength == 0) {
        tiling->totalLength = 0;
        tiling->blockLength = 0;
        tiling->tileLength = 0;
        tiling->tileNum = 0;
        context->SetBlockDim(1);
        context->GetWorkspaceSizes(1)[0] = 0;
        return ge::GRAPH_SUCCESS;
    }

    // 5. 按核切分
    uint32_t blockLength =
        static_cast<uint32_t>(AlignUp(CeilDiv(static_cast<int64_t>(totalLength), coreNum), alignNum));

    // 6. 按 UB 容量决定单次处理量（3 队列 × 双缓冲）
    int64_t bytesPerElem = BUFFER_NUM * 3 * dtypeSize;
    int64_t maxTileLength = AlignDown(static_cast<int64_t>(ubSize - UB_RESERVED) / bytesPerElem, alignNum);
    if (maxTileLength <= 0) {
        maxTileLength = alignNum;
    }
    uint32_t tileLength = static_cast<uint32_t>(std::min(static_cast<int64_t>(blockLength), maxTileLength));
    uint32_t tileNum = static_cast<uint32_t>(CeilDiv(static_cast<int64_t>(blockLength), tileLength));

    // 7. 回填 tiling
    tiling->totalLength = totalLength;
    tiling->blockLength = blockLength;
    tiling->tileLength = tileLength;
    tiling->tileNum = tileNum;

    // 8. 启动配置
    context->SetBlockDim(coreNum);
    context->GetWorkspaceSizes(1)[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static graphStatus InferShape(gert::InferShapeContext *context) {
    const gert::Shape *xShape = context->GetInputShape(0);
    gert::Shape *zShape = context->GetOutputShape(0);
    if (xShape == nullptr || zShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    *zShape = *xShape;
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context) {
    const ge::DataType xDtype = context->GetInputDataType(0);
    context->SetOutputDataType(0, xDtype);
    return ge::GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class Mul : public OpDef {
public:
    explicit Mul(const char *name) : OpDef(name) {
        this->Input("x").ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("y").ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("z").ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend910b");
    }
};
OP_ADD(Mul);
}  // namespace ops
