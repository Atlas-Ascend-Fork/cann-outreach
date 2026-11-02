// Host侧Tiling实现
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/mul_tiling.h"
#include "../op_kernel/tiling_key_mul.h"

namespace optiling {
    static ge::graphStatus TilingFunc(gert::TilingContext *context) {
        // 获取输入张量及其数据类型
        const gert::Tensor *tensor_x = context->GetRequiredInputTensor(0);
        uint32_t DT_X = static_cast<uint32_t>(tensor_x->GetDataType());
        // 配置 tiling key：按数据类型选择对应的 kernel 实例
        ASCENDC_TPL_SEL_PARAM(context, DT_X);

        // 按输入 shape 计算元素总个数
        const gert::StorageShape *x_shape = context->GetInputShape(0);
        uint32_t totalLength = 1;
        for (int i = 0; i < x_shape->GetStorageShape().GetDimNum(); ++i) {
            totalLength *= static_cast<uint32_t>(x_shape->GetStorageShape().GetDim(i));
        }

        // 选择 2 的幂的核数：不超过真实 AIV 核数，且能整除元素个数，保证各核等长
        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        int32_t num_cores_aiv = platform.GetCoreNumAiv();
        uint32_t blockDim = 8;
        while (blockDim > 1 &&
               (static_cast<int32_t>(blockDim) > num_cores_aiv || totalLength % blockDim != 0)) {
            blockDim >>= 1;
        }

        // 每核内以 128 元素为单位分块，配合双缓冲流水
        uint32_t blockLength = totalLength / blockDim;
        uint32_t tileNum = blockLength / (128u * 2u);
        if (tileNum < 1) {
            tileNum = 1;
        }

        // 计算 tiling 方案并写回 tiling 结构体
        MulTilingData *tiling = context->GetTilingData<MulTilingData>();
        tiling->totalLength = totalLength;
        tiling->tileNum = tileNum;

        // 配置启动核数
        context->SetBlockDim(blockDim);
        // 配置workspace大小
        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = 0;
        return ge::GRAPH_SUCCESS;
    }
}  // namespace optiling

namespace ge {
    static graphStatus InferShape(gert::InferShapeContext *context) {
        // 逐元素算子：输出 z 的 shape 与输入 x 一致
        const gert::Shape *x_shape = context->GetInputShape(0);
        gert::Shape *z_shape = context->GetOutputShape(0);
        *z_shape = *x_shape;
        return GRAPH_SUCCESS;
    }
    static graphStatus InferDataType(gert::InferDataTypeContext *context) {
        // 输出 z 的 dtype 与输入一致
        const auto inputDataType = context->GetInputDataType(0);
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
