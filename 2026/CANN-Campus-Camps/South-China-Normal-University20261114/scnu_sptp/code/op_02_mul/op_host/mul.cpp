// Host侧Tiling实现
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/mul_tiling.h"
#include "../op_kernel/tiling_key_mul.h"

namespace optiling {
    constexpr uint32_t BUFFER_NUM = 2;    // 与 kernel 侧一致（双缓冲）
    constexpr uint32_t TENSOR_NUM = 3;    // x、y、z 三个 tensor
    constexpr uint32_t MAX_CORES = 16;    // 实测最优核数上限
    constexpr uint32_t MIN_TILE_NUM = 4;  // 最小分块数，保证流水深度
    constexpr uint32_t ALIGN_ELEM = 32;   // 32 元素对齐

    static ge::graphStatus TilingFunc(gert::TilingContext *context) {
        // 获取平台信息
        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        int64_t max_cores = static_cast<int64_t>(platform.GetCoreNumAiv());
        uint64_t ub_size = 0;
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);

        // 获取算子输入信息，配置 tiling key
        const gert::Tensor *tensor_x = context->GetRequiredInputTensor(0);
        uint32_t DT_X = static_cast<uint32_t>(tensor_x->GetDataType());
        ASCENDC_TPL_SEL_PARAM(context, DT_X);
        int64_t dtype_size = ge::GetSizeByDataType(tensor_x->GetDataType());
        if (dtype_size <= 0) {
            dtype_size = 4;
        }

        // 总元素个数
        const gert::StorageShape* x1_shape = context->GetInputShape(0);
        int64_t totalLength = 1;
        for (int i = 0; i < x1_shape->GetStorageShape().GetDimNum(); i++) {
            totalLength *= x1_shape->GetStorageShape().GetDim(i);
        }

        // 1) 核数：上限 MAX_CORES（16），并取 totalLength 的约数保证各核等长
        int64_t coreNum = (max_cores > 0 && max_cores < MAX_CORES) ? max_cores : MAX_CORES;
        while (coreNum > 1 && (totalLength % coreNum) != 0) {
            --coreNum;
        }
        if (coreNum <= 0) {
            coreNum = 1;
        }
        int64_t blockLength = totalLength / coreNum;

        // 2) 单块长度 UB 上限（3 个 tensor × 双缓冲，留 1/4 余量）
        int64_t maxTileLen = static_cast<int64_t>(ub_size) * 3 / 4 /
                             (TENSOR_NUM * BUFFER_NUM) / dtype_size;
        maxTileLen = maxTileLen / ALIGN_ELEM * ALIGN_ELEM;
        if (maxTileLen < ALIGN_ELEM) {
            maxTileLen = ALIGN_ELEM;
        }

        // 3) tileNum：从 MIN_TILE_NUM 起取最小可行值（块尽量大），保证整除且不超 UB
        int64_t tileNum = 1;
        for (int64_t tn = MIN_TILE_NUM; tn * BUFFER_NUM <= blockLength; ++tn) {
            if (blockLength % (tn * BUFFER_NUM) != 0) {
                continue;
            }
            int64_t tileLen = blockLength / (tn * BUFFER_NUM);
            if (tileLen > maxTileLen) {
                continue;
            }
            tileNum = tn;
            break;
        }

        // 填充 tiling 结构体
        MulTilingData *tiling = context->GetTilingData<MulTilingData>();
        tiling->totalLength = static_cast<uint32_t>(totalLength);
        tiling->tileNum = static_cast<uint32_t>(tileNum);
        // 启动核数
        context->SetBlockDim(coreNum);
        // workspace 大小
        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = 0;
        return ge::GRAPH_SUCCESS;
    }
}  // namespace optiling

namespace ge {
    static graphStatus InferShape(gert::InferShapeContext *context) {
        // 获取第0号输入 x 的 shape
        const gert::Shape* x1_shape = context->GetInputShape(0);
        // 获取第0号输出 z 的 shape 可写指针
        gert::Shape* z_shape = context->GetOutputShape(0);
        // Mul 逐元素算子：输出 shape 与输入 shape 完全一致
        *z_shape = *x1_shape;
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
