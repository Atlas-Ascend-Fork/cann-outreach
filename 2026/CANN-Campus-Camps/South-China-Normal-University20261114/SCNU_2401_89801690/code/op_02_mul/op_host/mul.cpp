// Host侧Tiling实现
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/mul_tiling.h"
#include "../op_kernel/tiling_key_mul.h"

namespace {
// UB / GM 之间 DataCopy 的对齐粒度（字节）
constexpr uint32_t ALIGN_BYTES = 32;
// 每个队列的 buffer 块数，需与 kernel 侧保持一致（单缓冲）
constexpr uint32_t BUFFER_NUM = 1;
// 每个核承担的数据量目标：一块 32KB，让每次 DataCopy 都是大块搬运，同时避免为小张量拉起几十个核
constexpr uint32_t TARGET_BYTES_PER_CORE = 32 * 1024;
// 单个 tile 的字节上限
constexpr uint32_t MAX_TILE_BYTES = TARGET_BYTES_PER_CORE;
}  // namespace
namespace optiling {
    static ge::graphStatus TilingFunc(gert::TilingContext *context) {
        // 获取平台信息
        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        int32_t num_cores_aiv = platform.GetCoreNumAiv();
        uint64_t ub_size = 0;
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);

        // 获取算子输入数组信息
        const gert::Tensor* tensor_x = context->GetRequiredInputTensor(0);
        ge::DataType dtype_x = tensor_x->GetDataType(); // 获取数据类型
        int dtype_size_x = ge::GetSizeByDataType(dtype_x); // 获取数据类型的字长
        uint32_t length_x = static_cast<uint32_t>(tensor_x->GetShapeSize()); // 获取元素个数

        // 配置tiling key, 从而实现kernel侧不同数据类型/算法的区分
        uint32_t DT_X = static_cast<uint32_t>(dtype_x);
        ASCENDC_TPL_SEL_PARAM(context, DT_X);

        MulTilingData* tiling = context->GetTilingData<MulTilingData>();
        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = 0;

        // 空张量：不需要启动核
        if (length_x == 0 || dtype_size_x <= 0) {
            tiling->length = 0;
            tiling->blockLength = 0;
            tiling->tileLength = 0;
            context->SetBlockDim(1);
            return ge::GRAPH_SUCCESS;
        }

        // 32B 对齐所需的元素个数：float -> 8，half -> 16
        const uint32_t alignElems = ALIGN_BYTES / static_cast<uint32_t>(dtype_size_x);
        // 选择启动核数：不超过硬件 vector 核数，也不超过“对齐元素个数”能切出的核数
        uint32_t coreNum = (num_cores_aiv > 0) ? static_cast<uint32_t>(num_cores_aiv) : 1u;
        uint32_t maxCoreNum = (length_x + alignElems - 1) / alignElems;
        if (maxCoreNum == 0) {
            maxCoreNum = 1;
        }
        if (coreNum > maxCoreNum) {
            coreNum = maxCoreNum;
        }

        // 每个核处理的元素个数：均分后向上取整到 32B 对齐（最后一个核可能少一些）
        uint32_t blockLength = (length_x + coreNum - 1) / coreNum;
        blockLength = (blockLength + alignElems - 1) / alignElems * alignElems;

        // 单核内 tile 长度：按 UB 容量限制（3 个队列 × BUFFER_NUM 块），并做 32B 对齐
        uint64_t tileBytes = ub_size / (static_cast<uint64_t>(BUFFER_NUM) * 3u * 2u);
        if (tileBytes > MAX_TILE_BYTES) {
            tileBytes = MAX_TILE_BYTES;
        }
        if (tileBytes < ALIGN_BYTES) {
            tileBytes = ALIGN_BYTES;
        }
        tileBytes = tileBytes / ALIGN_BYTES * ALIGN_BYTES;

        uint32_t tileLength = static_cast<uint32_t>(tileBytes / static_cast<uint32_t>(dtype_size_x));
        tileLength = tileLength / alignElems * alignElems;
        if (tileLength == 0) {
            tileLength = alignElems;
        }
        if (tileLength > blockLength) {
            tileLength = blockLength;
        }

        // 填充 tiling 结构体
        tiling->length = length_x;
        tiling->blockLength = blockLength;
        tiling->tileLength = tileLength;

        // 配置启动核数
        context->SetBlockDim(coreNum);
        return ge::GRAPH_SUCCESS;
    }
}  // namespace optiling

namespace ge {
    static graphStatus InferShape(gert::InferShapeContext *context) {
        // z 与 x 形状一致
        const gert::Shape *xShape = context->GetInputShape(0);
        gert::Shape *zShape = context->GetOutputShape(0);
        *zShape = *xShape;
        return ge::GRAPH_SUCCESS;
    }
    static graphStatus InferDataType(gert::InferDataTypeContext *context) {
        // z 与 x 数据类型一致
        ge::DataType dtype = context->GetInputDataType(0);
        context->SetOutputDataType(0, dtype);
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