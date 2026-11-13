// Kernel侧核函数实现
#include "kernel_operator.h"

#include "mul_tiling.h"
#include "tiling_key_mul.h"

using namespace AscendC;

namespace {
    constexpr int32_t BUFFER_NUM = 2;    // 双缓冲：搬运与计算流水重叠
    constexpr int64_t ALIGN_BYTES = 32;  // 与 host 侧保持一致
}

template <class DT_X>
class KernelMul {
public:
    __aicore__ inline KernelMul() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t totalLength, uint32_t blockLength, uint32_t tileLength, uint32_t tileNum) {
        this->tileLength = tileLength;
        this->tileNum = tileNum;
        this->alignNum = static_cast<uint32_t>(ALIGN_BYTES / sizeof(DT_X));

        // 1) 本核在整块数据中的起点与长度
        uint32_t coreIdx = GetBlockIdx();
        uint64_t coreOffset = static_cast<uint64_t>(coreIdx) * blockLength;
        uint32_t coreLength = 0;
        if (coreOffset < totalLength) {
            uint64_t rest = totalLength - coreOffset;
            coreLength = (rest >= blockLength) ? blockLength : static_cast<uint32_t>(rest);
        }
        this->coreLength = coreLength;

        // 2) 绑定 GM 地址：基址 + 偏移，并把长度限制为本核负责的范围
        xGm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(x) + coreOffset, coreLength);
        yGm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(y) + coreOffset, coreLength);
        zGm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(z) + coreOffset, coreLength);

        // 3) 申请 UB：三个队列，每个队列 BUFFER_NUM 块，每块 tileLength 个元素
        pipe.InitBuffer(inQueueX, BUFFER_NUM, tileLength * sizeof(DT_X));
        pipe.InitBuffer(inQueueY, BUFFER_NUM, tileLength * sizeof(DT_X));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, tileLength * sizeof(DT_X));
    }

    __aicore__ inline void Process() {
        for (uint32_t i = 0; i < tileNum; ++i) {
            uint32_t offset = i * tileLength;
            if (offset >= coreLength) {
                break;
            }
            // 本 tile 实际要搬的元素个数（最后一块可能不满）
            uint32_t curLength = tileLength;
            if (offset + curLength > coreLength) {
                curLength = coreLength - offset;
            }
            // 计算长度向上对齐到 32B：多算出来的尾部只会落在 UB 的 padding 区，
            // 由 DataCopyPad 填 0，且不会被拷回 GM，因此安全
            uint32_t calLength = (curLength + alignNum - 1) / alignNum * alignNum;

            CopyIn(offset, curLength);
            Compute(calLength);
            CopyOut(offset, curLength);
        }
    }
private:
    __aicore__ inline void CopyIn(uint32_t offset, uint32_t curLength) {
        LocalTensor<DT_X> xLocal = inQueueX.template AllocTensor<DT_X>();
        LocalTensor<DT_X> yLocal = inQueueY.template AllocTensor<DT_X>();

        // blockCount = 1；blockLen 单位为 Byte；stride 全 0 表示连续搬运
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(curLength * sizeof(DT_X)), 0, 0, 0};
        // 非 32B 对齐的部分在 UB 侧用 0 补齐
        DataCopyPadExtParams<DT_X> padParams{false, 0, 0, static_cast<DT_X>(0)};

        DataCopyPad(xLocal, xGm[offset], copyParams, padParams);
        DataCopyPad(yLocal, yGm[offset], copyParams, padParams);

        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }

    __aicore__ inline void Compute(uint32_t calLength) {
        LocalTensor<DT_X> xLocal = inQueueX.DeQue<DT_X>();
        LocalTensor<DT_X> yLocal = inQueueY.DeQue<DT_X>();
        LocalTensor<DT_X> zLocal = outQueueZ.template AllocTensor<DT_X>();

        AscendC::Mul(zLocal, xLocal, yLocal, calLength);

        outQueueZ.EnQue<DT_X>(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOut(uint32_t offset, uint32_t curLength) {
        LocalTensor<DT_X> zLocal = outQueueZ.DeQue<DT_X>();

        DataCopyExtParams copyParams{1, static_cast<uint32_t>(curLength * sizeof(DT_X)), 0, 0, 0};
        DataCopyPad(zGm[offset], zLocal, copyParams);

        outQueueZ.FreeTensor(zLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueY;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueZ;

    GlobalTensor<DT_X> xGm;
    GlobalTensor<DT_X> yGm;
    GlobalTensor<DT_X> zGm;

    uint32_t tileLength = 0;
    uint32_t tileNum = 0;
    uint32_t coreLength = 0;
    uint32_t alignNum = 0;
};

template <typename DT_X>
 __global__ __aicore__ void mul(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(MulTilingData);
    GET_TILING_DATA_WITH_STRUCT(MulTilingData, tiling_data, tiling);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);   // 纯 Vector 算子，只在 AIV 上跑
    KernelMul<DT_X> op;
    op.Init(x, y, z, tiling_data.totalLength, tiling_data.blockLength, tiling_data.tileLength, tiling_data.tileNum);
    op.Process();
}
