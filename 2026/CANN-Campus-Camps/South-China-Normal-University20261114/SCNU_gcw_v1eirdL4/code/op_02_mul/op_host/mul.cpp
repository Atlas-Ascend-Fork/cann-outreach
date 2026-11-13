// Host侧Tiling实现
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/mul_tiling.h"
#include "../op_kernel/tiling_key_mul.h"

namespace optiling {
    const uint32_t TILE_NUM = 8;      // 单核分块数，与kernel侧保持一致
    const uint32_t BUFFER_NUM = 2;    // 队列缓冲块数，与kernel侧保持一致
    static ge::graphStatus TilingFunc(gert::TilingContext *context) {
        // 示例: 获取平台信息
        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        int32_t num_cores_aiv = platform.GetCoreNumAiv();
        // 示例: 获取算子输入数组信息
        const gert::Tensor* tensor_x = context->GetRequiredInputTensor(0);
        const gert::Tensor* tensor_y = context->GetRequiredInputTensor(1);
        ge::DataType dtype_x = tensor_x->GetDataType(); // 获取数据类型
        int dtype_size_x = ge::GetSizeByDataType(dtype_x); // 获取数据类型的字长
        uint32_t length_x = tensor_x->GetShapeSize(); // 获取元素个数
        // 示例: 配置tiling key, 从而实现kernel侧不同数据类型/算法的区分
        uint32_t DT_X = static_cast<uint32_t>(dtype_x);
        ASCENDC_TPL_SEL_PARAM(context, DT_X);
        // 示例: 计算tiling方案并填充tiling结构体
        MulTilingData* tiling = context->GetTilingData<MulTilingData>();
        tiling->length = length_x;
        // 配置启动核数：优先用满硬件AIV核数，同时保证
        // 1) 总长度能被核数整除；2) 每核长度能被 TILE_NUM * BUFFER_NUM 均分；
        // 3) 每个分块的字节数不小于32Byte且为32Byte的整数倍，满足DataCopy的对齐要求。
        const uint32_t tileUnit = TILE_NUM * BUFFER_NUM;
        uint32_t coreNum = 1;
        for (uint32_t c = static_cast<uint32_t>(num_cores_aiv > 0 ? num_cores_aiv : 1); c > 1; --c) {
            if (length_x % c != 0) {
                continue;
            }
            uint32_t perCore = length_x / c;
            if (perCore % tileUnit != 0) {
                continue;
            }
            uint32_t tileBytes = (perCore / tileUnit) * static_cast<uint32_t>(dtype_size_x);
            if (tileBytes < 32 || tileBytes % 32 != 0) {
                continue;
            }
            coreNum = c;
            break;
        }
        context->SetBlockDim(coreNum);
        // 配置workspace大小
        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = 0;
        return ge::GRAPH_SUCCESS;
    }
}  // namespace optiling

namespace ge {
    static graphStatus InferShape(gert::InferShapeContext *context) {
        // Mul为逐元素运算，输出shape与输入x完全一致
        const gert::Shape* x_shape = context->GetInputShape(0);
        gert::Shape* z_shape = context->GetOutputShape(0);
        *z_shape = *x_shape;
        return GRAPH_SUCCESS;
    }
    static graphStatus InferDataType(gert::InferDataTypeContext *context) {
        // 输出数据类型与输入x保持一致
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
