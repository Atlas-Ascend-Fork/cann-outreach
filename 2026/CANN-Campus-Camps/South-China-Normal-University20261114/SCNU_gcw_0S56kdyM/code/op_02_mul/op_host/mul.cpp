// // Host侧Tiling实现
// #include "register/op_def_registry.h"
// #include "tiling/platform/platform_ascendc.h"

// #include "../op_kernel/mul_tiling.h"
// #include "../op_kernel/tiling_key_mul.h"

// namespace optiling {
//     static ge::graphStatus TilingFunc(gert::TilingContext *context) {
//         // 示例: 获取平台信息
//         auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
//         int32_t num_cores_aiv = platform.GetCoreNumAiv();
//         uint64_t ub_size;
//         platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);
//         // 示例: 获取算子输入数组信息
//         const gert::Tensor* tensor_x = context->GetRequiredInputTensor(0);
//         const gert::Tensor* tensor_y = context->GetRequiredInputTensor(1);
//         ge::DataType dtype_x = tensor_x->GetDataType(); // 获取数据类型
//         int dtype_size_x = ge::GetSizeByDataType(dtype_x); // 获取数据类型的字长
//         uint32_t length_x = tensor_x->GetShapeSize(); // 获取元素个数
//         uint32_t size_x = tensor_x->GetSize(); // 获取内存大小
//         // 示例: 配置tiling key, 从而实现kernel侧不同数据类型/算法的区分
//         uint32_t DT_X = static_cast<uint32_t>(dtype_x);
//         ASCENDC_TPL_SEL_PARAM(context, DT_X);
//         // 示例: 计算tiling方案并填充tiling结构体
//         MulTilingData* tiling = context->GetTilingData<MulTilingData>();
//         tiling->length = length_x;
//         // 配置启动核数
//         context->SetBlockDim(num_cores_aiv);
//         // 配置workspace大小
//         size_t *currentWorkspace = context->GetWorkspaceSizes(1);
//         currentWorkspace[0] = 0;
//         return ge::GRAPH_SUCCESS;
//     }
// }  // namespace optiling

// namespace ge {
//     static graphStatus InferShape(gert::InferShapeContext *context) {
//         return GRAPH_SUCCESS;
//     }
//     static graphStatus InferDataType(gert::InferDataTypeContext *context) {
//         return ge::GRAPH_SUCCESS;
//     }
// }  // namespace ge

// namespace ops {
//     class Mul : public OpDef {
//     public:
//         explicit Mul(const char *name) : OpDef(name) {
//             this->Input("x")
//                 .ParamType(REQUIRED)
//                 .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
//                 .Format({ge::FORMAT_ND, ge::FORMAT_ND});
//             this->Input("y")
//                 .ParamType(REQUIRED)
//                 .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
//                 .Format({ge::FORMAT_ND, ge::FORMAT_ND});
//             this->Output("z")
//                 .ParamType(REQUIRED)
//                 .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
//                 .Format({ge::FORMAT_ND, ge::FORMAT_ND});
//             this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
//             this->AICore()
//                 .SetTiling(optiling::TilingFunc)
//                 .AddConfig("ascend910b");
//         }
//     };
//     OP_ADD(Mul);
// }  // namespace ops

// Host侧Tiling实现
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/mul_tiling.h"
#include "../op_kernel/tiling_key_mul.h"

namespace optiling {
    // 与 Kernel 侧保持一致，必须相同
    const uint32_t TILE_NUM = 8;
    const uint32_t BUFFER_NUM = 2;

    static ge::graphStatus TilingFunc(gert::TilingContext *context) {
        // 获取平台信息
        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        int32_t num_cores_aiv = platform.GetCoreNumAiv();

        // 获取算子输入数组信息
        const gert::Tensor* tensor_x = context->GetRequiredInputTensor(0);
        const gert::Tensor* tensor_y = context->GetRequiredInputTensor(1);
        if (tensor_x == nullptr || tensor_y == nullptr) {
            return ge::GRAPH_FAILED;
        }

        ge::DataType dtype_x = tensor_x->GetDataType(); // 获取数据类型

        // 配置tiling key，从而实现kernel侧不同数据类型/算法的区分
        uint32_t DT_X = static_cast<uint32_t>(dtype_x);
        ASCENDC_TPL_SEL_PARAM(context, DT_X);

        // 计算tiling方案并填充tiling结构体
        MulTilingData* tiling = context->GetTilingData<MulTilingData>();
        if (tiling == nullptr) {
            return ge::GRAPH_FAILED;
        }

        // 通过 StorageShape 逐维相乘计算元素总个数（与 Add 保持一致，兼容动态 shape）
        const gert::StorageShape* x_shape = context->GetInputShape(0);
        if (x_shape == nullptr) {
            return ge::GRAPH_FAILED;
        }
        int64_t total_length = 1;
        for (int i = 0; i < x_shape->GetStorageShape().GetDimNum(); i++) {
            total_length *= x_shape->GetStorageShape().GetDim(i);
        }
        tiling->length = static_cast<uint32_t>(total_length);

        // ---- 关键：自适应 blockDim，保证 blockLength 能被 TILE_NUM*BUFFER_NUM 整除 ----
        // Kernel 侧：blockLength = totalLength / blockDim
        //            tileLength  = blockLength / TILE_NUM / BUFFER_NUM
        // 为避免整除丢数据，blockDim 需满足：
        //   totalLength 能被 blockDim 整除，
        //   且 blockLength 能被 TILE_NUM * BUFFER_NUM 整除。
        const uint32_t unit = TILE_NUM * BUFFER_NUM; // 每核最少处理元素数
        uint32_t block_dim = static_cast<uint32_t>(num_cores_aiv);

        // 核数不能超过元素总数
        if (block_dim > static_cast<uint32_t>(total_length)) {
            block_dim = static_cast<uint32_t>(total_length);
        }
        // 保证每个核至少能分到 unit 个元素，否则减半核数
        while (block_dim > 1 && (total_length / block_dim) < unit) {
            block_dim /= 2;
        }
        // 再保证 blockLength 能被 unit 整除（向下调整核数直到整除）
        while (block_dim > 1 && ((total_length / block_dim) % unit) != 0) {
            block_dim /= 2;
        }
        // 兜底：至少 1 个核
        if (block_dim == 0) {
            block_dim = 1;
        }

        // 配置启动核数
        context->SetBlockDim(block_dim);

        // 配置workspace大小
        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        if (currentWorkspace == nullptr) {
            return ge::GRAPH_FAILED;
        }
        currentWorkspace[0] = 0;

        return ge::GRAPH_SUCCESS;
    }
}  // namespace optiling

namespace ge {
    static graphStatus InferShape(gert::InferShapeContext *context) {
        // 获取第0号输入 x 的 shape
        const gert::Shape* x_shape = context->GetInputShape(0);
        // 获取第0号输出 z 的 shape 可写指针
        gert::Shape* z_shape = context->GetOutputShape(0);
        // Mul 逐元素算子：输出 shape 与输入 shape 完全一致
        *z_shape = *x_shape;
        return GRAPH_SUCCESS;
    }

    static graphStatus InferDataType(gert::InferDataTypeContext *context) {
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