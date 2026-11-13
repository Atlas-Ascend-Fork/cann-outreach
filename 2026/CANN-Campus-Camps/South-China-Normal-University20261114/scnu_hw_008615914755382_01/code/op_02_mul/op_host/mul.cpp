// Host侧Tiling实现
#include <cstdint>
#include <limits>

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/mul_tiling.h"
#include "../op_kernel/tiling_key_mul.h"

namespace optiling {
    constexpr uint32_t BLOCK_DIM = 8;

    static ge::graphStatus TilingFunc(gert::TilingContext *context)
    {
        const gert::Tensor *tensor_x =
            context->GetRequiredInputTensor(0);
        const gert::Tensor *tensor_y =
            context->GetRequiredInputTensor(1);

        if (tensor_x == nullptr || tensor_y == nullptr) {
            return ge::GRAPH_FAILED;
        }

        // 输入必须使用相同的数据类型
        const auto dtype = tensor_x->GetDataType();
        if (dtype != tensor_y->GetDataType() ||
            (dtype != ge::DT_FLOAT && dtype != ge::DT_FLOAT16)) {
            return ge::GRAPH_FAILED;
        }

        // 获取总元素数，避免转换到uint32_t时溢出
        const auto totalLength = tensor_x->GetShapeSize();
        if (totalLength <= 0 ||
            static_cast<uint64_t>(totalLength) >
                std::numeric_limits<uint32_t>::max() ||
            totalLength != tensor_y->GetShapeSize()) {
            return ge::GRAPH_FAILED;
        }

        // 按输入类型选择Kernel模板实例
        uint32_t DT_X = static_cast<uint32_t>(dtype);
        ASCENDC_TPL_SEL_PARAM(context, DT_X);

        MulTilingData *tiling =
            context->GetTilingData<MulTilingData>();
        if (tiling == nullptr) {
            return ge::GRAPH_FAILED;
        }

        // 启动8个核，并传递分块参数
        context->SetBlockDim(BLOCK_DIM);
        tiling->length = static_cast<uint32_t>(totalLength);

        // 不申请额外workspace
        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = 0;

        return ge::GRAPH_SUCCESS;
    }
}  // namespace optiling

namespace ge {
    static graphStatus InferShape(gert::InferShapeContext *context)
    {
        const gert::Shape *xShape = context->GetInputShape(0);
        const gert::Shape *yShape = context->GetInputShape(1);
        gert::Shape *zShape = context->GetOutputShape(0);

        if (xShape == nullptr || yShape == nullptr ||
            zShape == nullptr) {
            return GRAPH_FAILED;
        }

        // 本实现要求两个输入形状一致
        if (xShape->GetDimNum() != yShape->GetDimNum()) {
            return GRAPH_FAILED;
        }
        for (size_t i = 0; i < xShape->GetDimNum(); ++i) {
            if (xShape->GetDim(i) != yShape->GetDim(i)) {
                return GRAPH_FAILED;
            }
        }

        // 输出形状与输入相同
        *zShape = *xShape;
        return GRAPH_SUCCESS;
    }

    static graphStatus InferDataType(
        gert::InferDataTypeContext *context)
    {
        const auto dtypeX = context->GetInputDataType(0);
        const auto dtypeY = context->GetInputDataType(1);

        if (dtypeX != dtypeY ||
            (dtypeX != DT_FLOAT && dtypeX != DT_FLOAT16)) {
            return GRAPH_FAILED;
        }

        // 输出类型与输入相同
        return context->SetOutputDataType(0, dtypeX);
    }
}  // namespace ge

namespace ops {
    class Mul : public OpDef {
    public:
        explicit Mul(const char *name) : OpDef(name)
        {
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

            this->SetInferShape(ge::InferShape)
                .SetInferDataType(ge::InferDataType);

            this->AICore()
                .SetTiling(optiling::TilingFunc)
                .AddConfig("ascend910b");
        }
    };

    OP_ADD(Mul);
}  // namespace ops