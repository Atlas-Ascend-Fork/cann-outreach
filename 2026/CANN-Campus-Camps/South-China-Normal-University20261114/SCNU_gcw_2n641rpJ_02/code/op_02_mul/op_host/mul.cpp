// Host侧Tiling实现
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/mul_tiling.h"
#include "../op_kernel/tiling_key_mul.h"

namespace optiling {
    const uint32_t TILE_NUM = 1;
const uint32_t BLOCK_DIM = 4;

    static ge::graphStatus TilingFunc(gert::TilingContext *context)
    {
        // 获取输入数据类型，用于选择对应的Kernel模板
        const gert::Tensor *tensorX = context->GetRequiredInputTensor(0);
        const gert::Tensor *tensorY = context->GetRequiredInputTensor(1);
        (void)tensorY;

        uint32_t dtX = static_cast<uint32_t>(tensorX->GetDataType());
        ASCENDC_TPL_SEL_PARAM(context, dtX);

        // 输入形状固定为(8, 2048)，总元素数为16384
        const int64_t totalLength =
            context->GetInputShape(0)->GetStorageShape().GetShapeSize();

        MulTilingData *tiling = context->GetTilingData<MulTilingData>();
        tiling->totalLength = static_cast<uint32_t>(totalLength);
        tiling->tileNum = TILE_NUM;

        // 使用8个AI Vector Core，每个核处理2048个元素
        context->SetBlockDim(BLOCK_DIM);

        // 本算子不需要额外workspace
        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = 0;

        return ge::GRAPH_SUCCESS;
    }
}  // namespace optiling

namespace ge {
    static graphStatus InferShape(gert::InferShapeContext *context)
    {
        // z的形状与x相同，均为(8, 2048)
        const gert::Shape *xShape = context->GetInputShape(0);
        gert::Shape *zShape = context->GetOutputShape(0);
        *zShape = *xShape;

        return GRAPH_SUCCESS;
    }

    static graphStatus InferDataType(gert::InferDataTypeContext *context)
    {
        // z的数据类型与x相同
        context->SetOutputDataType(0, context->GetInputDataType(0));
        return ge::GRAPH_SUCCESS;
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