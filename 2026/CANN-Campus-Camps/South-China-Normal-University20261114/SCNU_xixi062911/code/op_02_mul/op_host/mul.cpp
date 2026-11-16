// Host侧Tiling实现
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/mul_tiling.h"
#include "../op_kernel/tiling_key_mul.h"

namespace optiling {
    static ge::graphStatus TilingFunc(gert::TilingContext *context) {
        // 1. 获取平台信息：AI Vector 核数 / UB 大小
        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        int32_t num_cores_aiv = platform.GetCoreNumAiv();
        uint64_t ub_size = 0;
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);

        // 2. 获取输入信息（x/y/z 形状一致，取 x 即可）
        const gert::Tensor* tensor_x = context->GetRequiredInputTensor(0);
        ge::DataType dtype_x = tensor_x->GetDataType();      // 获取数据类型
        int32_t dtype_size = ge::GetSizeByDataType(dtype_x);  // 数据类型字长（字节）
        uint32_t length = static_cast<uint32_t>(tensor_x->GetShapeSize());  // 元素总个数

        // 3. 配置 tiling key，kernel 侧据此实例化不同数据类型的模板
        uint32_t DT_X = static_cast<uint32_t>(dtype_x);
        ASCENDC_TPL_SEL_PARAM(context, DT_X);

        // 4. 启动核数：block 数过多时每块都有固定的调度/启动开销（实测约 0.5us/块），
        //    对本赛题这种小数据量（16384 元素）应"少核 + 每核多搬"，故按每核至少 2048 个元素来定核数
        const uint32_t MIN_ELEMS_PER_CORE = 2048;
        uint32_t align_elems = 32 / static_cast<uint32_t>(dtype_size);
        uint32_t cores_by_data = length / MIN_ELEMS_PER_CORE;
        uint32_t block_dim = static_cast<uint32_t>(num_cores_aiv);
        if (cores_by_data < block_dim) {
            block_dim = cores_by_data;   // 数据量小则少开核，控制在 1 个调度批次内
        }
        uint32_t total_blocks = length / align_elems;
        if (total_blocks < block_dim) {
            block_dim = total_blocks;
        }
        if (block_dim < 1) {
            block_dim = 1;
        }


        // 5. 填充 tiling 结构体
        MulTilingData* tiling = context->GetTilingData<MulTilingData>();
        tiling->length = length;

        // 6. 配置启动核数
        context->SetBlockDim(block_dim);
        // 7. 配置workspace大小
        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = 0;
        return ge::GRAPH_SUCCESS;
    }
}  // namespace optiling

namespace ge {
    static graphStatus InferShape(gert::InferShapeContext *context) {
        // 逐元素算子：输出 z 的 shape 与输入 x 完全一致
        const gert::Shape* x_shape = context->GetInputShape(0);
        gert::Shape* z_shape = context->GetOutputShape(0);
        *z_shape = *x_shape;
        return GRAPH_SUCCESS;
    }
    static graphStatus InferDataType(gert::InferDataTypeContext *context) {
        // 输出数据类型与输入保持一致（float32 / float16）
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
