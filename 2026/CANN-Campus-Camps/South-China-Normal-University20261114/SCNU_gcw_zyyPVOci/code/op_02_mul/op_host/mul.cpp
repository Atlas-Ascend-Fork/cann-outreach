#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/mul_tiling.h"
#include "../op_kernel/tiling_key_mul.h"

namespace optiling {
    static ge::graphStatus TilingFunc(gert::TilingContext *context) {
        // 获取平台信息
        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        int32_t num_cores_aiv = platform.GetCoreNumAiv();
        uint64_t ub_size;
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);

        // 获取输入数组信息
        const gert::Tensor* tensor_x = context->GetRequiredInputTensor(0);
        const gert::Tensor* tensor_y = context->GetRequiredInputTensor(1);
        ge::DataType dtype_x = tensor_x->GetDataType();
        int dtype_size_x = ge::GetSizeByDataType(dtype_x);
        uint32_t length_x = tensor_x->GetShapeSize();
        uint32_t size_x = tensor_x->GetSize();

        // 按数据类型选 kernel 分支（float32 / float16）
        uint32_t DT_X = static_cast<uint32_t>(dtype_x);
        ASCENDC_TPL_SEL_PARAM(context, DT_X);

        // 填充 tiling 结构体
        MulTilingData* tiling = context->GetTilingData<MulTilingData>();
        tiling->length = length_x;

        // ---- 切分计算 ----
        // 1) 选核数：从 num_cores_aiv 往下找第一个能整除 length_x 的核数，
        //    避免尾块长度不对齐。16384=2^14，910B 有 20 个 AIV 核，20 不能整除 → 退到 16。
        uint32_t use_core_num = static_cast<uint32_t>(num_cores_aiv);
        while (use_core_num > 1 && (length_x % use_core_num != 0)) {
            use_core_num--;
        }
        uint32_t core_length = length_x / use_core_num;

        // 2) 算单次搬进 UB 的元素数。
        //    UB 里放 x/y/z 三块，double buffer 再 ×2 = 6 块，保守取 UB 的 1/8。
        uint32_t tile_length = static_cast<uint32_t>(ub_size / (dtype_size_x * 6 * 8));
        // DataCopy 要求 32 字节对齐：fp32 时 8 元素对齐，fp16 时 16 元素对齐
        uint32_t align_elems = 32 / static_cast<uint32_t>(dtype_size_x);
        tile_length = (tile_length / align_elems) * align_elems;
        if (tile_length == 0) {
            tile_length = align_elems;
        }
        // 保证 core_length 能被 tile_length 整除（kernel 侧不做尾块处理）
        while (tile_length > core_length || (core_length % tile_length != 0)) {
            if (tile_length <= align_elems) {
                tile_length = align_elems;
                break;
            }
            tile_length -= align_elems;
        }
        tiling->tile_length = tile_length;

        // 配置启动核数
        context->SetBlockDim(use_core_num);

        // 配置 workspace 大小
        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = 0;
        return ge::GRAPH_SUCCESS;
    }
}  // namespace optiling

namespace ge {
    static graphStatus InferShape(gert::InferShapeContext *context) {
        const gert::Shape* x_shape = context->GetInputShape(0);
        gert::Shape* out_shape = context->GetOutputShape(0);
        out_shape->SetDimNum(x_shape->GetDimNum());
        for (size_t i = 0; i < x_shape->GetDimNum(); i++) {
            out_shape->SetDim(i, x_shape->GetDim(i));
        }
        return GRAPH_SUCCESS;
    }

    static graphStatus InferDataType(gert::InferDataTypeContext *context) {
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