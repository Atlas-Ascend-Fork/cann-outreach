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
//         // MulTilingData* tiling = context->GetTilingData<MulTilingData>();
//         // tiling->length = length_x;
//         // // 配置启动核数
//         // context->SetBlockDim(num_cores_aiv);
//         // 计算tiling方案并填充tiling结构体
//         uint32_t typeSize   = ge::GetSizeByDataType(dtype_x);
//         uint32_t alignNum   = 32 / typeSize;                                   // 一个 32B block 能放几个元素
//         uint32_t alignLen   = (length_x + alignNum - 1) / alignNum * alignNum; // 对齐后的总长度
//         uint32_t coreNum    = num_cores_aiv > 0 ? num_cores_aiv : 1;
//         uint32_t perCore    = (alignLen + coreNum - 1) / coreNum;
//         perCore = (perCore + alignNum - 1) / alignNum * alignNum;              // 每核处理量再做 32B 对齐
//         if (perCore == 0) perCore = alignNum;
//         coreNum = (alignLen + perCore - 1) / perCore;                          // 实际需要的核数

//         MulTilingData* tiling = context->GetTilingData<MulTilingData>();
//         tiling->length = length_x;
//         // 配置启动核数
//         context->SetBlockDim(coreNum);

//         // 配置workspace大小
//         size_t *currentWorkspace = context->GetWorkspaceSizes(1);
//         currentWorkspace[0] = 0;
//         return ge::GRAPH_SUCCESS;
//     }
// }  // namespace optiling

// // namespace ge {
// //     static graphStatus InferShape(gert::InferShapeContext *context) {
// //         return GRAPH_SUCCESS;
// //     }
// //     static graphStatus InferDataType(gert::InferDataTypeContext *context) {
// //         return ge::GRAPH_SUCCESS;
// //     }
// // }  // namespace ge

// namespace ge {
// static graphStatus InferShape(gert::InferShapeContext *context) {
//     const gert::Shape *xShape = context->GetInputShape(0);
//     gert::Shape *zShape = context->GetOutputShape(0);
//     if (xShape == nullptr || zShape == nullptr) return ge::GRAPH_FAILED;
//     *zShape = *xShape;              // 逐元素乘：形状完全一致
//     return ge::GRAPH_SUCCESS;
// }
// static graphStatus InferDataType(gert::InferDataTypeContext *context) {
//     auto xDtype = context->GetInputDataType(0);
//     context->SetOutputDataType(0, xDtype);   // 输出类型跟 x 一致
//     return ge::GRAPH_SUCCESS;
// }
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
    static constexpr uint32_t MIN_ELE_PER_CORE = 2048;   // ← 新增这一行
    static ge::graphStatus TilingFunc(gert::TilingContext *context) {
        // 示例: 获取平台信息
        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        int32_t num_cores_aiv = platform.GetCoreNumAiv();
        //uint64_t ub_size;
        //platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);
        // 示例: 获取算子输入数组信息
        const gert::Tensor* tensor_x = context->GetRequiredInputTensor(0);
        const gert::Tensor* tensor_y = context->GetRequiredInputTensor(1);
        ge::DataType dtype_x = tensor_x->GetDataType(); // 获取数据类型
        int dtype_size_x = ge::GetSizeByDataType(dtype_x); // 获取数据类型的字长
        uint32_t length_x = tensor_x->GetShapeSize(); // 获取元素个数
        uint32_t size_x = tensor_x->GetSize(); // 获取内存大小
        // 示例: 配置tiling key, 从而实现kernel侧不同数据类型/算法的区分
        uint32_t DT_X = static_cast<uint32_t>(dtype_x);
        ASCENDC_TPL_SEL_PARAM(context, DT_X);
        // 示例: 计算tiling方案并填充tiling结构体
        MulTilingData* tiling = context->GetTilingData<MulTilingData>();
        tiling->length = length_x;
        // 配置启动核数
        //context->SetBlockDim(num_cores_aiv);
        // 按数据量收缩核数：16384个元素 => 8个核，正好每核一行(2048)
        uint32_t need_cores = (length_x + MIN_ELE_PER_CORE - 1) / MIN_ELE_PER_CORE;
        if (need_cores == 0) {
            need_cores = 1;
        }
        if (need_cores > static_cast<uint32_t>(num_cores_aiv)) {
            need_cores = static_cast<uint32_t>(num_cores_aiv);
        }
        context->SetBlockDim(need_cores);

        // 配置workspace大小
        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = 0;
        return ge::GRAPH_SUCCESS;
    }
}  // namespace optiling

namespace ge {
    static graphStatus InferShape(gert::InferShapeContext *context) {
        return GRAPH_SUCCESS;
    }
    static graphStatus InferDataType(gert::InferDataTypeContext *context) {
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



