// Host侧Tiling实现
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/mul_tiling.h"
#include "../op_kernel/tiling_key_mul.h"

namespace optiling {
    const uint32_t TILE_NUM = 8;

    static ge::graphStatus TilingFunc(gert::TilingContext *context) {
        // 1. 获取平台信息
        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        int32_t num_cores_aiv = platform.GetCoreNumAiv();

        // 2. 获取输入信息
        const gert::Tensor *tensor_x = context->GetRequiredInputTensor(0);
        ge::DataType dtype_x = tensor_x->GetDataType();
        uint32_t totalLength = static_cast<uint32_t>(tensor_x->GetShapeSize());

        // 3. 配置 tiling key（区分 fp16 / fp32）
        uint32_t DT_X = static_cast<uint32_t>(dtype_x);
        ASCENDC_TPL_SEL_PARAM(context, DT_X);

        // 4. 选一个能整除总长度的核数，避免末尾元素被漏算
        uint32_t blockDim = 1;
        for (uint32_t d = static_cast<uint32_t>(num_cores_aiv); d >= 1; --d) {
            if (totalLength % d == 0 && totalLength / d >= TILE_NUM * 2) {
                blockDim = d;
                break;
            }
        }

        // 5. 填充 tiling 结构体
        MulTilingData *tiling = context->GetTilingData<MulTilingData>();
        tiling->totalLength = totalLength;
        tiling->tileNum = TILE_NUM;

        // 6. 配置启动核数与 workspace
        context->SetBlockDim(blockDim);
        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = 0;
        return ge::GRAPH_SUCCESS;
    }
}  // namespace optiling

namespace ge {
    static graphStatus InferShape(gert::InferShapeContext *context) {
        const gert::Shape *xShape = context->GetInputShape(0);
        gert::Shape *zShape = context->GetOutputShape(0);
        *zShape = *xShape;   // 输出形状与输入 x 一致
        return GRAPH_SUCCESS;
    }

    static graphStatus InferDataType(gert::InferDataTypeContext *context) {
        const ge::DataType xDtype = context->GetInputDataType(0);
        context->SetOutputDataType(0, xDtype);   // 输出类型与输入一致
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