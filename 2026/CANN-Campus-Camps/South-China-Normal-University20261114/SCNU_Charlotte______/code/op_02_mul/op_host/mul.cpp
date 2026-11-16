// Host侧Tiling实现
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/mul_tiling.h"
#include "../op_kernel/tiling_key_mul.h"

namespace optiling {
    const uint32_t TILE_NUM = 8;
    const uint32_t BLOCK_DIM = 8;
    static ge::graphStatus TilingFunc(gert::TilingContext *context) {
        // 示例: 获取算子输入数组信息
        const gert::Tensor *tensor_x = context->GetRequiredInputTensor(0);
        const gert::Tensor *tensor_y = context->GetRequiredInputTensor(1);
        uint32_t DT_X = static_cast<uint32_t>(tensor_x->GetDataType());
        ASCENDC_TPL_SEL_PARAM(context, DT_X);
        MulTilingData *tiling = context->GetTilingData<MulTilingData>();

        // 1. 计算总元素数：形状 (8, 2048) -> 16384（x/y/z 形状一致，以 x 为准）
        int64_t totalLength = context->GetInputShape(0)->GetOriginShape().GetShapeSize();
        if (totalLength <= 0) {
            return ge::GRAPH_FAILED;
        }

        // 2. 多核切分：启动 BLOCK_DIM 个核，每核处理 totalLength / BLOCK_DIM 个元素
        //    若总长度不能被 BLOCK_DIM 整除，则退化为可整除的核数，避免尾部数据丢失
        uint32_t blockDim = BLOCK_DIM;
        while (blockDim > 1 && (static_cast<uint64_t>(totalLength) % blockDim) != 0) {
            blockDim /= 2;
        }
        context->SetBlockDim(blockDim);

        // 3. 填充 tiling 数据，供 kernel 侧计算 blockLength / tileLength
        tiling->totalLength = static_cast<uint32_t>(totalLength);
        tiling->tileNum = TILE_NUM;

        // 4. 登记 tiling data 的实际大小（保证 kernel 侧 GET_TILING_DATA_WITH_STRUCT 取到完整结构体）
        auto rawTilingData = context->GetRawTilingData();
        if (rawTilingData != nullptr) {
            rawTilingData->SetDataSize(sizeof(MulTilingData));
        }

        // 配置workspace大小
        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = 0;
        return ge::GRAPH_SUCCESS;
    }
}  // namespace optiling

namespace ge {
    static graphStatus InferShape(gert::InferShapeContext *context) {
        // z 的形状与 x 完全一致：shape(z) = shape(x)
        const gert::Shape *xShape = context->GetInputShape(0);
        gert::Shape *zShape = context->GetOutputShape(0);
        *zShape = *xShape;
        return GRAPH_SUCCESS;
    }
    static graphStatus InferDataType(gert::InferDataTypeContext *context) {
        // z 的数据类型与 x 一致（fp32 / fp16）
        const ge::DataType inputDataType = context->GetInputDataType(0);
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
                .SetTiling(optiling::TilingFunc)
                .AddConfig("ascend910b");
        }
    };
    OP_ADD(Mul);
}  // namespace ops