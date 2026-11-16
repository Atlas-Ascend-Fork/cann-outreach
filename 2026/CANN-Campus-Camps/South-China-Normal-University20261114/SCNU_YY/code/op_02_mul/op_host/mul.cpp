// Host侧Tiling实现
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/mul_tiling.h"
#include "../op_kernel/tiling_key_mul.h"

#include <algorithm>
#include <cstdint>

namespace optiling {
constexpr uint32_t ALIGN_NUM = 256;

static ge::graphStatus TilingFunc(gert::TilingContext *context) {
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    int32_t max_cores_aiv = platform.GetCoreNumAiv();
    if (max_cores_aiv <= 0) {
        return ge::GRAPH_FAILED;
    }

    uint64_t ub_size = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);

    const gert::Tensor* tensor_x = context->GetRequiredInputTensor(0);
    const gert::Tensor* tensor_y = context->GetRequiredInputTensor(1);
    if (tensor_x == nullptr || tensor_y == nullptr) {
        return ge::GRAPH_FAILED;
    }

    ge::DataType dtype_x = tensor_x->GetDataType();
    ge::DataType dtype_y = tensor_y->GetDataType();
    if (dtype_x != dtype_y) {
        return ge::GRAPH_FAILED;
    }
    if (dtype_x != ge::DT_FLOAT && dtype_x != ge::DT_FLOAT16) {
        return ge::GRAPH_FAILED;
    }

    uint32_t length_x = tensor_x->GetShapeSize();
    uint32_t length_y = tensor_y->GetShapeSize();
    if (length_x == 0 || length_x != length_y) {
        return ge::GRAPH_FAILED;
    }

    int dtype_size_x = ge::GetSizeByDataType(dtype_x);
    if (dtype_size_x <= 0) {
        return ge::GRAPH_FAILED;
    }

    // x/y/z 三块 UB buffer，留一半余量
    uint32_t max_tile_by_ub = static_cast<uint32_t>((ub_size / 2) / (3 * dtype_size_x));
    max_tile_by_ub = max_tile_by_ub / ALIGN_NUM * ALIGN_NUM;
    if (max_tile_by_ub < ALIGN_NUM) {
        max_tile_by_ub = ALIGN_NUM;
    }

    // 本题 shape=(8,2048)，length=16384。
    // tile_length 取 4096 时，4 核各处理 4096 个元素，无尾块。
    uint32_t tile_length = std::min<uint32_t>(max_tile_by_ub, 4096);
    if (tile_length < ALIGN_NUM) {
        tile_length = ALIGN_NUM;
    }

    uint32_t used_cores = (length_x + tile_length - 1) / tile_length;
    if (used_cores == 0) {
        used_cores = 1;
    }
    if (used_cores > static_cast<uint32_t>(max_cores_aiv)) {
        used_cores = static_cast<uint32_t>(max_cores_aiv);
    }

    uint32_t per_core_length = (length_x + used_cores - 1) / used_cores;
    per_core_length = (per_core_length + ALIGN_NUM - 1) / ALIGN_NUM * ALIGN_NUM;

    used_cores = (length_x + per_core_length - 1) / per_core_length;
    if (used_cores == 0) {
        used_cores = 1;
    }
    if (used_cores > static_cast<uint32_t>(max_cores_aiv)) {
        used_cores = static_cast<uint32_t>(max_cores_aiv);
    }

    uint32_t tile_num = (per_core_length + tile_length - 1) / tile_length;

    MulTilingData* tiling = context->GetTilingData<MulTilingData>();
    if (tiling == nullptr) {
        return ge::GRAPH_FAILED;
    }
    tiling->length = length_x;
    tiling->perCoreLength = per_core_length;
    tiling->tileLength = tile_length;
    tiling->tileNum = tile_num;

    // 保持初始模板的 tiling key 配置方式，float32/float16 由模板参数区分
    uint32_t DT_X = static_cast<uint32_t>(dtype_x);
    ASCENDC_TPL_SEL_PARAM(context, DT_X);

    // 必须与 perCoreLength 的计算保持一致，不能继续用 num_cores_aiv
    context->SetBlockDim(used_cores);

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
    const gert::Shape* x_shape = context->GetInputShape(0);
    gert::Shape* z_shape = context->GetOutputShape(0);
    if (x_shape == nullptr || z_shape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    *z_shape = *x_shape;
    return ge::GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context) {
    ge::DataType dtype = context->GetInputDataType(0);
    context->SetOutputDataType(0, dtype);
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
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend910b");
    }
};
OP_ADD(Mul);
}  // namespace ops