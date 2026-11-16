// Host侧Tiling实现
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/mul_tiling.h"
#include "../op_kernel/tiling_key_mul.h"

namespace optiling {
    static ge::graphStatus TilingFunc(gert::TilingContext *context) {
        // 获取平台信息
        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        int32_t num_cores_aiv = platform.GetCoreNumAiv();
        uint64_t ub_size;
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);
        // 获取算子输入数组信息
        const gert::Tensor* tensor_x = context->GetRequiredInputTensor(0);
        ge::DataType dtype_x = tensor_x->GetDataType();      // 获取数据类型
        int dtype_size_x = ge::GetSizeByDataType(dtype_x);   // 获取数据类型的字长
        uint32_t length_x = tensor_x->GetShapeSize();        // 获取元素个数
        uint32_t size_x = tensor_x->GetSize();               // 获取内存大小
        (void)size_x;
        (void)ub_size;

        // 配置 tiling key：按输入 dtype 区分 kernel 侧实例（float / half）
        uint32_t DT_X = static_cast<uint32_t>(dtype_x);
        ASCENDC_TPL_SEL_PARAM(context, DT_X);

        // 填充 tiling 结构体：只传总长度，每核的切分方式由 kernel 侧按 blockNum 推导
        MulTilingData* tiling = context->GetTilingData<MulTilingData>();
        tiling->length = length_x;

        // 启动核数取硬件 AIV 核数，kernel 侧 GetBlockNum() 与本值一致
        context->SetBlockDim(num_cores_aiv);

        // 本算子不需要额外 workspace
        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = 0;
        return ge::GRAPH_SUCCESS;
    }
}  // namespace optiling

namespace ge {
    // 输出 z 与输入 x 形状、数据类型完全一致
    static graphStatus InferShape(gert::InferShapeContext *context) {
        const gert::Shape *xShape = context->GetInputShape(0);
        if (xShape == nullptr) {
            return ge::GRAPH_FAILED;
        }
        gert::Shape *zShape = context->GetOutputShape(0);
        if (zShape == nullptr) {
            return ge::GRAPH_FAILED;
        }
        *zShape = *xShape;
        return GRAPH_SUCCESS;
    }
    static graphStatus InferDataType(gert::InferDataTypeContext *context) {
        const ge::DataType dtype_x = context->GetInputDataType(0);
        context->SetOutputDataType(0, dtype_x);
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
