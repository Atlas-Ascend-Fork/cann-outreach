// Host侧Tiling实现
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/mul_tiling.h"
#include "../op_kernel/tiling_key_mul.h"

namespace optiling {
    const uint32_t TILE_NUM = 8;
    const uint32_t BUFFER_NUM = 2;
    static ge::graphStatus TilingFunc(gert::TilingContext *context) {
        // 示例: 获取算子输入数组信息
        const gert::Tensor *tensor_x = context->GetRequiredInputTensor(0);
        const gert::Tensor *tensor_y = context->GetRequiredInputTensor(1);
        uint32_t DT_X = static_cast<uint32_t>(tensor_x->GetDataType());
        ASCENDC_TPL_SEL_PARAM(context, DT_X);
        MulTilingData *tiling = context->GetTilingData<MulTilingData>();

        // 1. 计算输入/输出的总元素个数（x/y/z 形状一致）
        const gert::Shape &xShape = context->GetInputShape(0)->GetStorageShape();
        uint32_t totalLength = 1;
        for (size_t i = 0; i < xShape.GetDimNum(); ++i) {
            totalLength *= static_cast<uint32_t>(xShape.GetDim(i));
        }
        tiling->totalLength = totalLength;
        tiling->tileNum = TILE_NUM;

        // 2. 多核切分：尽量用满平台的 AIV 核数，同时保证每个核的数据能被
        //    TILE_NUM * BUFFER_NUM 整除（kernel 侧按 tileNum * BUFFER_NUM 分块循环）
        platform_ascendc::PlatformAscendC ascendcPlatform(context->GetPlatformInfo());
        uint32_t blockDim = ascendcPlatform.GetCoreNumAiv();
        if (blockDim == 0) {
            blockDim = 1;
        }
        const uint32_t alignUnit = TILE_NUM * BUFFER_NUM;
        if (blockDim > totalLength / alignUnit) {
            blockDim = totalLength / alignUnit;
            if (blockDim == 0) {
                blockDim = 1;
            }
        }
        while (blockDim > 1 && (totalLength % (blockDim * alignUnit) != 0)) {
            --blockDim;
        }
        context->SetBlockDim(blockDim);

        // 配置workspace大小
        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = 0;
        return ge::GRAPH_SUCCESS;
    }
}  // namespace optiling

namespace ge {
    static graphStatus InferShape(gert::InferShapeContext *context) {
        // 输出 z 与输入 x 形状一致
        const gert::Shape *xShape = context->GetInputShape(0);
        gert::Shape *zShape = context->GetOutputShape(0);
        *zShape = *xShape;
        return GRAPH_SUCCESS;
    }
    static graphStatus InferDataType(gert::InferDataTypeContext *context) {
        // 输出 z 与输入 x 数据类型一致（float32 / float16）
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
