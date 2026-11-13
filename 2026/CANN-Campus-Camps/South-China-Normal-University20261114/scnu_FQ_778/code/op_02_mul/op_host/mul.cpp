#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/mul_tiling.h"
#include "../op_kernel/tiling_key_mul.h"

namespace optiling {

constexpr uint32_t BLOCK_DIM = 8;
constexpr uint32_t TILE_NUM = 8;
constexpr int64_t ALIGN_BYTES = 32;
constexpr int64_t MAX_TOTAL_LENGTH = (1LL << 40);
constexpr uint64_t DEFAULT_UB_SIZE = 192ULL * 1024ULL;

static ge::graphStatus MulTilingFunc(gert::TilingContext *context)
{
    auto inputDesc = context->GetInputDesc(0);
    if (inputDesc == nullptr) {
        return ge::GRAPH_FAILED;
    }
    uint32_t DT_X = static_cast<uint32_t>(inputDesc->GetDataType());
    ASCENDC_TPL_SEL_PARAM(context, DT_X);

    MulTilingData *tiling = context->GetTilingData<MulTilingData>();
    if (tiling == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const gert::StorageShape *x1_shape = context->GetInputShape(0);
    if (x1_shape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    int64_t totalLength = 1;
    for (size_t i = 0; i < x1_shape->GetStorageShape().GetDimNum(); i++) {
        int64_t dim = x1_shape->GetStorageShape().GetDim(i);
        if (dim <= 0) {
            return ge::GRAPH_FAILED;
        }
        if (totalLength > MAX_TOTAL_LENGTH / dim) {
            return ge::GRAPH_FAILED;
        }
        totalLength *= dim;
    }
    if (totalLength <= 0 || totalLength > MAX_TOTAL_LENGTH) {
        return ge::GRAPH_FAILED;
    }

    uint64_t ubSize = DEFAULT_UB_SIZE;
    auto platformInfoPtr = context->GetPlatformInfo();
    if (platformInfoPtr != nullptr) {
        platform_ascendc::PlatformAscendC ascendcPlatform(platformInfoPtr);
        uint64_t queriedUb = 0;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, queriedUb);
        if (queriedUb > 0) {
            ubSize = queriedUb;
        }
    }

    int64_t dtypeSize = (DT_X == 1) ? 2 : 4;
    int64_t alignElems = ALIGN_BYTES / dtypeSize;
    if (alignElems < 1) {
        alignElems = 1;
    }

    int64_t base = totalLength / static_cast<int64_t>(BLOCK_DIM);
    if (base < TILE_NUM) {
        base = TILE_NUM;
    }
    int64_t ubBudget = static_cast<int64_t>(ubSize) / 2;
    int64_t maxByUb = ubBudget / (3 * 2 * dtypeSize);
    int64_t cand = base / static_cast<int64_t>(TILE_NUM);
    if (cand > maxByUb) {
        cand = maxByUb;
    }
    int64_t tileLength = (cand / alignElems) * alignElems;
    if (tileLength < alignElems) {
        tileLength = alignElems;
    }

    tiling->totalLength = totalLength;
    tiling->tileLength = tileLength;

    context->SetBlockDim(BLOCK_DIM);

    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    if (currentWorkspace == nullptr) {
        return ge::GRAPH_FAILED;
    }
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}

}  // namespace optiling

namespace ge {
    static graphStatus InferShape(gert::InferShapeContext *context) {
        const gert::Shape *x1_shape = context->GetInputShape(0);
        if (x1_shape == nullptr) {
            return ge::GRAPH_FAILED;
        }
        gert::Shape *z_shape = context->GetOutputShape(0);
        if (z_shape == nullptr) {
            return ge::GRAPH_FAILED;
        }
        *z_shape = *x1_shape;
        return GRAPH_SUCCESS;
    }
    static graphStatus InferDataType(gert::InferDataTypeContext *context) {
        const auto inputDataType = context->GetInputDataType(0);
        context->SetOutputDataType(0, inputDataType);
        return ge::GRAPH_SUCCESS;
    }
}  // namespace ge

namespace ops {
    class Mul : public OpDef {
    public:
        explicit Mul(const char *name) : OpDef(name) {
            this->Input("x")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Input("y")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Output("z")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
            this->AICore()
                .SetTiling(optiling::MulTilingFunc)
                .AddConfig("ascend910b");
        }
    };
    OP_ADD(Mul);
}  // namespace ops

