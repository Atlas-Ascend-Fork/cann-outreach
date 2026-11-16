// Host侧Tiling实现
#include <cstdint>
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/mul_tiling.h"
#include "../op_kernel/tiling_key_mul.h"

namespace optiling {
    constexpr uint32_t BUFFER_NUM  = 2;
    constexpr uint32_t ALIGN_BYTES = 32;   // DataCopy 的 32B 对齐要求

    static ge::graphStatus TilingFunc(gert::TilingContext *context) {
        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        uint32_t coresAiv = static_cast<uint32_t>(platform.GetCoreNumAiv());
        uint64_t ubSize = 0;
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);

        const gert::Tensor *tensor_x = context->GetRequiredInputTensor(0);
        uint32_t length    = static_cast<uint32_t>(tensor_x->GetShapeSize());
        ge::DataType dtype = tensor_x->GetDataType();
        uint32_t dtypeSize = static_cast<uint32_t>(ge::GetSizeByDataType(dtype));

        uint32_t DT_X = static_cast<uint32_t>(dtype);
        ASCENDC_TPL_SEL_PARAM(context, DT_X);

        MulTilingData *tiling = context->GetTilingData<MulTilingData>();
        tiling->length = length;

        if (length == 0 || dtypeSize == 0) {
            tiling->blockLength = 0;
            tiling->tileLength  = 1;
            context->SetBlockDim(1);
            size_t *ws0 = context->GetWorkspaceSizes(1);
            ws0[0] = 0;
            return ge::GRAPH_SUCCESS;
        }

        // ① 以 32B 为最小分配单位，保证 GM/UB 搬运对齐
        uint32_t alignNum   = ALIGN_BYTES / dtypeSize;                 // 每个 32B 里的元素数
        uint32_t totalUnits = (length + alignNum - 1) / alignNum;      // 总 32B 单元数

        // ② 核数不能超过单元数（否则出现空核/整除截断）
        uint32_t blockDim = (coresAiv < totalUnits) ? coresAiv : totalUnits;
        if (blockDim == 0) blockDim = 1;

        // ③ 每核按 32B 单元向上均分，最后一核少处理，不丢尾巴
        uint32_t unitsPerCore = (totalUnits + blockDim - 1) / blockDim;
        uint32_t blockLength  = unitsPerCore * alignNum;

        // ④ 每块长度：依据 UB 容量算，3 个队列 × BUFFER_NUM 份，留 10% 余量
        uint64_t budget = ubSize / 10 * 9 / (3 * BUFFER_NUM);
        budget = budget / ALIGN_BYTES * ALIGN_BYTES;
        uint32_t tileLength = static_cast<uint32_t>(budget / dtypeSize);
        if (tileLength > blockLength) tileLength = blockLength;
        tileLength = tileLength / alignNum * alignNum;                 // 保持 32B 对齐
        if (tileLength == 0) tileLength = alignNum;

        tiling->blockLength = blockLength;
        tiling->tileLength  = tileLength;

        context->SetBlockDim(blockDim);      // 注意：这里是算出来的 blockDim，不是 coresAiv
        size_t *ws = context->GetWorkspaceSizes(1);
        ws[0] = 0;
        return ge::GRAPH_SUCCESS;
    }
}  // namespace optiling

namespace ge {
    static graphStatus InferShape(gert::InferShapeContext *context) {
        *context->GetOutputShape(0) = *context->GetInputShape(0);
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