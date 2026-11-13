// Host侧Tiling实现
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/mul_tiling.h"
#include "../op_kernel/tiling_key_mul.h"

namespace optiling {

static inline uint32_t AlignUp(uint32_t value, uint32_t align) {
    return (value + align - 1) / align * align;
}

static ge::graphStatus TilingFunc(gert::TilingContext *context) {
    // 获取平台信息
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    int32_t num_cores_aiv = platform.GetCoreNumAiv();
    uint64_t ub_size;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);

    // 获取算子输入信息
    const gert::Tensor* tensor_x = context->GetRequiredInputTensor(0);
    ge::DataType dtype_x = tensor_x->GetDataType();
    uint32_t dtype_size_x = ge::GetSizeByDataType(dtype_x);
    uint32_t length_x = tensor_x->GetShapeSize();

    // 配置tiling key，kernel侧按数据类型区分
    uint32_t DT_X = static_cast<uint32_t>(dtype_x);
    ASCENDC_TPL_SEL_PARAM(context, DT_X);

    // 计算tiling方案
    uint32_t align_elems = 32 / dtype_size_x;  // 32字节对齐的元素个数
    // 以每核约8KB数据量为目标估算所需核数：本算子数据量较小，
    // 核数过多会放大每核固定的调度/同步开销，适度的核数反而更快
    constexpr uint32_t PER_CORE_BYTES = 8 * 1024;
    uint32_t per_core_elems = PER_CORE_BYTES / dtype_size_x;
    int32_t block_dim = static_cast<int32_t>((length_x + per_core_elems - 1) / per_core_elems);
    if (block_dim > num_cores_aiv) {
        block_dim = num_cores_aiv;
    }
    if (block_dim < 1) {
        block_dim = 1;
    }
    uint32_t block_factor = AlignUp((length_x + static_cast<uint32_t>(block_dim) - 1) /
                                        static_cast<uint32_t>(block_dim),
                                    align_elems);
    // 实际使用核数：保证每核数据量不超过UB容量(预留一部分给搬运与对齐补齐)
    uint32_t ub_elem_cap = static_cast<uint32_t>(ub_size / 4 / dtype_size_x);
    if (block_factor > ub_elem_cap) {
        block_factor = (ub_elem_cap / align_elems) * align_elems;
        block_dim = (static_cast<int32_t>(length_x) + static_cast<int32_t>(block_factor) - 1) /
                    static_cast<int32_t>(block_factor);
    }
    if (block_dim < 1) {
        block_dim = 1;
    }

    MulTilingData* tiling = context->GetTilingData<MulTilingData>();
    tiling->length = length_x;
    tiling->block_factor = block_factor;
    context->SetBlockDim(block_dim);

    // 配置workspace大小
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
    static graphStatus InferShape(gert::InferShapeContext *context) {
        // 输出z与输入x形状一致
        const gert::Shape *x_shape = context->GetInputShape(0);
        gert::Shape *z_shape = context->GetOutputShape(0);
        *z_shape = *x_shape;
        return GRAPH_SUCCESS;
    }
    static graphStatus InferDataType(gert::InferDataTypeContext *context) {
        // 输出z与输入x数据类型一致
        context->SetOutputDataType(0, context->GetInputDataType(0));
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
