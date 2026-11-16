// Host侧Tiling实现
//
// 功能：逐元素乘法 z = x * y，支持 float32 / float16，形状 (8, 2048)（共 16384 个元素）
// Tiling 思路：
//   1. 按 Vector Core 个数把 totalLength 均分，perCoreLength 按 32B 对齐；
//   2. 核内再按 tile 切分（最多 TILE_NUM 份、每份至少 1KB），并保证 3 份 buffer * double buffer 能放进 UB；
//   3. 通过 ASCENDC_TPL_SEL_PARAM 生成 TilingKey，使 Kernel 侧按 DT_X 实例化对应实现。
#include <algorithm>
#include <cstdint>

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/mul_tiling.h"
#include "../op_kernel/tiling_key_mul.h"

namespace optiling {
    const uint32_t TILE_NUM = 8;
    constexpr uint32_t BLOCK_BYTES = 32;              // 一次 DMA/Vector 访问的对齐粒度
    constexpr uint32_t BUFFER_NUM = 2;                // Kernel 侧每个队列的 buffer 个数
    constexpr uint32_t RESERVED_UB_BYTES = 8 * 1024;  // 预留的 UB 空间
    constexpr uint32_t MIN_TILE_BYTES = 1024;         // 每个 tile 至少搬运的字节数
    constexpr uint64_t DEFAULT_UB_BYTES = 192 * 1024; // 平台信息获取失败时的兜底值

    static inline uint32_t CeilDiv(uint32_t dividend, uint32_t divisor)
    {
        return (divisor == 0) ? 0 : ((dividend + divisor - 1) / divisor);
    }

    static inline uint32_t AlignUp(uint32_t value, uint32_t align)
    {
        return CeilDiv(value, align) * align;
    }

    static ge::graphStatus TilingFunc(gert::TilingContext *context) {
        // 示例: 获取算子输入数组信息
        const gert::Tensor *tensor_x = context->GetRequiredInputTensor(0);
        const gert::Tensor *tensor_y = context->GetRequiredInputTensor(1);
        if (tensor_x == nullptr || tensor_y == nullptr) {
            return ge::GRAPH_FAILED;
        }
        uint32_t DT_X = static_cast<uint32_t>(tensor_x->GetDataType());
        ASCENDC_TPL_SEL_PARAM(context, DT_X);
        MulTilingData *tiling = context->GetTilingData<MulTilingData>();
        if (tiling == nullptr) {
            return ge::GRAPH_FAILED;
        }

        // 1. 取输入总元素个数（题面形状 (8, 2048) -> 16384）
        const gert::StorageShape *xShapePtr = context->GetInputShape(0);
        const gert::StorageShape *yShapePtr = context->GetInputShape(1);
        if (xShapePtr == nullptr || yShapePtr == nullptr) {
            return ge::GRAPH_FAILED;
        }
        const int64_t xSize = xShapePtr->GetOriginShape().GetShapeSize();
        const int64_t ySize = yShapePtr->GetOriginShape().GetShapeSize();
        // z = x * y 是逐元素运算，输入输出形状一致，不支持广播/动态 shape
        if (xSize < 0 || ySize < 0 || xSize != ySize) {
            return ge::GRAPH_FAILED;
        }
        if (tensor_y->GetDataType() != tensor_x->GetDataType()) {
            return ge::GRAPH_FAILED;
        }
        const uint32_t totalLength = static_cast<uint32_t>(xSize);

        // 2. 32B 对齐粒度：float32 -> 8 个元素，float16 -> 16 个元素
        const uint32_t typeSize = (tensor_x->GetDataType() == ge::DT_FLOAT16) ? 2U : 4U;
        const uint32_t alignElements = BLOCK_BYTES / typeSize;

        // 3. 获取平台信息：可用的 Vector 核个数与 UB 大小
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        uint32_t coreNum = static_cast<uint32_t>(ascendcPlatform.GetCoreNumAiv());
        if (coreNum == 0) {
            coreNum = static_cast<uint32_t>(ascendcPlatform.GetCoreNum());
        }
        if (coreNum == 0) {
            coreNum = 1;
        }
        uint64_t ubSize = DEFAULT_UB_BYTES;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);

        // 4. 空张量：不启动计算，直接给出全 0 的 tiling（Kernel 侧会立刻返回）
        if (totalLength == 0) {
            tiling->totalLength = 0;
            tiling->coreNum = 1;
            tiling->perCoreLength = 0;
            tiling->tileLength = alignElements;
            tiling->tileNum = 0;
            tiling->alignElements = alignElements;
            context->SetBlockDim(1);
            context->GetRawTilingData()->SetDataSize(sizeof(MulTilingData));
            size_t *currentWorkspace = context->GetWorkspaceSizes(1);
            currentWorkspace[0] = 0;
            return ge::GRAPH_SUCCESS;
        }

        // 5. 多核切分：每核至少处理一个 32B 数据块，避免启动空核
        uint32_t maxCoresByData = CeilDiv(totalLength, alignElements);
        if (maxCoresByData == 0) {
            maxCoresByData = 1;
        }
        uint32_t usedCores = std::min(coreNum, maxCoresByData);
        uint32_t perCoreLength = AlignUp(CeilDiv(totalLength, usedCores), alignElements);
        if (perCoreLength == 0) {
            perCoreLength = alignElements;
        }
        usedCores = CeilDiv(totalLength, perCoreLength);  // 重新计算真正会用到的核数
        if (usedCores == 0) {
            usedCores = 1;
        }

        // 6. 核内切 tile：先按 TILE_NUM 均分，再由 UB 容量限制上限
        //    UB 需要同时容纳 x/y/z 三份 buffer，且按 BUFFER_NUM 做 double buffer
        const uint64_t usableUb = (ubSize > RESERVED_UB_BYTES) ? (ubSize - RESERVED_UB_BYTES) : ubSize;
        uint32_t maxTileLength = static_cast<uint32_t>(usableUb / (3ULL * BUFFER_NUM * typeSize));
        maxTileLength = (maxTileLength / alignElements) * alignElements;
        if (maxTileLength < alignElements) {
            maxTileLength = alignElements;
        }
        // 每个 tile 至少搬运 MIN_TILE_BYTES 字节，避免 tile 过碎导致 DMA 启动开销占比过高
        uint32_t minTileLength = AlignUp(MIN_TILE_BYTES / typeSize, alignElements);
        uint32_t tileLength = AlignUp(CeilDiv(perCoreLength, TILE_NUM), alignElements);
        tileLength = std::max(tileLength, minTileLength);
        tileLength = std::min(tileLength, maxTileLength);
        tileLength = std::min(tileLength, perCoreLength);
        tileLength = (tileLength / alignElements) * alignElements;
        if (tileLength < alignElements) {
            tileLength = alignElements;
        }
        const uint32_t tileNum = CeilDiv(perCoreLength, tileLength);

        // 7. 回填 TilingData
        tiling->totalLength = totalLength;
        tiling->coreNum = usedCores;
        tiling->perCoreLength = perCoreLength;
        tiling->tileLength = tileLength;
        tiling->tileNum = tileNum;
        tiling->alignElements = alignElements;

        context->SetBlockDim(usedCores);
        context->GetRawTilingData()->SetDataSize(sizeof(MulTilingData));

        // 配置workspace大小
        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = 0;
        return ge::GRAPH_SUCCESS;
    }
}  // namespace optiling

namespace ge {
    static graphStatus InferShape(gert::InferShapeContext *context) {
        const gert::Shape *xShape = context->GetInputShape(0);
        const gert::Shape *yShape = context->GetInputShape(1);
        gert::Shape *zShape = context->GetOutputShape(0);
        if (xShape == nullptr || yShape == nullptr || zShape == nullptr) {
            return ge::GRAPH_FAILED;
        }
        // z = x * y 为逐元素运算：输出形状与输入形状完全一致
        if (xShape->GetDimNum() != yShape->GetDimNum()) {
            return ge::GRAPH_FAILED;
        }
        for (size_t i = 0; i < xShape->GetDimNum(); ++i) {
            if (xShape->GetDim(i) != yShape->GetDim(i)) {
                return ge::GRAPH_FAILED;
            }
        }
        *zShape = *xShape;
        return GRAPH_SUCCESS;
    }
    static graphStatus InferDataType(gert::InferDataTypeContext *context) {
        // 输出 z 的数据类型与输入 x 保持一致（float32 / float16）
        const ge::DataType xDtype = context->GetInputDataType(0);
        context->SetOutputDataType(0, xDtype);
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
