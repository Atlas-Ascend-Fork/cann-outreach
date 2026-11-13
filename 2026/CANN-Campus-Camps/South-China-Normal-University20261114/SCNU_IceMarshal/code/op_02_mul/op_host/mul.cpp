// Host侧Tiling实现
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/mul_tiling.h"
#include "../op_kernel/tiling_key_mul.h"

namespace optiling {
    constexpr uint32_t BLOCK_SIZE = 32;  // UB 与 GM 间搬运的基本块长度，单位字节
    constexpr uint32_t TILE_NUM = 8;     // 单核内的数据块个数，与 kernel 侧保持一致

    static ge::graphStatus TilingFunc(gert::TilingContext *context) {
        // 获取平台信息
        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        int32_t num_cores_aiv = platform.GetCoreNumAiv();
        // 获取算子输入信息
        const gert::Tensor* tensor_x = context->GetRequiredInputTensor(0);
        ge::DataType dtype_x = tensor_x->GetDataType();
        int dtype_size_x = ge::GetSizeByDataType(dtype_x);
        uint32_t length_x = tensor_x->GetShapeSize();
        // 配置tiling key，实现kernel侧不同数据类型的区分
        uint32_t DT_X = static_cast<uint32_t>(dtype_x);
        ASCENDC_TPL_SEL_PARAM(context, DT_X);
        // 计算启动核数：单核数据长度须整除为 TILE_NUM 块，且每块字节数为 BLOCK_SIZE 的整数倍
        uint32_t alignNum = BLOCK_SIZE / static_cast<uint32_t>(dtype_size_x);
        uint32_t blockStep = TILE_NUM * alignNum;
        uint32_t blockDim = static_cast<uint32_t>(num_cores_aiv);
        uint32_t maxBlockDim = length_x / blockStep;
        if (blockDim > maxBlockDim) {
            blockDim = maxBlockDim;
        }
        if (blockDim == 0) {
            blockDim = 1;
        }
        while (blockDim > 1 && ((length_x % blockDim != 0) || ((length_x / blockDim) % blockStep != 0))) {
            blockDim--;
        }
        // 填充tiling结构体
        MulTilingData* tiling = context->GetTilingData<MulTilingData>();
        tiling->length = length_x;
        context->SetBlockDim(blockDim);
        // 配置workspace大小
        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = 0;
        return ge::GRAPH_SUCCESS;
    }
}  // namespace optiling

namespace ge {
    static graphStatus InferShape(gert::InferShapeContext *context) {
        const gert::Shape* xShape = context->GetInputShape(0);
        gert::Shape* zShape = context->GetOutputShape(0);
        *zShape = *xShape;
        return GRAPH_SUCCESS;
    }
    static graphStatus InferDataType(gert::InferDataTypeContext *context) {
        auto dtype_x = context->GetInputDataType(0);
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
