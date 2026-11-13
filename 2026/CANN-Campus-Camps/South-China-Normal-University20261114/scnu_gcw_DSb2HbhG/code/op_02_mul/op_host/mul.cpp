// Host侧Tiling实现
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/mul_tiling.h"
#include "../op_kernel/tiling_key_mul.h"

namespace optiling {
    static ge::graphStatus TilingFunc(gert::TilingContext *context) {
        // 获取平台信息：本平台可用的 AI Vector Core 个数
        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        int32_t num_cores_aiv = platform.GetCoreNumAiv();
        if (num_cores_aiv <= 0) {
            num_cores_aiv = 1;
        }

        // 获取第一个输入 x 的元素个数和数据类型
        const gert::Tensor *tensor_x = context->GetRequiredInputTensor(0);
        uint32_t length_x = static_cast<uint32_t>(tensor_x->GetShapeSize());
        ge::DataType dtype_x = tensor_x->GetDataType();

        // 配置 TilingKey：Kernel 侧据此实例化 float / half 两种版本
        uint32_t DT_X = static_cast<uint32_t>(dtype_x);
        ASCENDC_TPL_SEL_PARAM(context, DT_X);

        // 填充 Tiling 数据。Kernel 只需知道总元素个数，
        // 每个核处理多少、怎么对齐、怎么分 tile 都在 Kernel 内部计算。
        MulTilingData *tiling = context->GetTilingData<MulTilingData>();
        tiling->length = length_x;

        // 启动所有 Vector Core
        context->SetBlockDim(static_cast<uint32_t>(num_cores_aiv));

        // 本算子不需要额外 workspace
        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = 0;
        return ge::GRAPH_SUCCESS;
    }
}  // namespace optiling

namespace ge {
    static graphStatus InferShape(gert::InferShapeContext *context) {
        // z 的 shape 与 x 完全一致
        const gert::Shape *xShape = context->GetInputShape(0);
        gert::Shape *zShape = context->GetOutputShape(0);
        if (xShape == nullptr || zShape == nullptr) {
            return ge::GRAPH_FAILED;
        }
        *zShape = *xShape;
        return GRAPH_SUCCESS;
    }
    static graphStatus InferDataType(gert::InferDataTypeContext *context) {
        // z 的 dtype 与 x 一致
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