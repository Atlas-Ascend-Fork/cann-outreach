// Kernel侧核函数实现
//
// 功能：逐元素乘法 z = x * y
// 策略：
//   1. Host 侧按核数把 totalLength 均分（32B 对齐），本核只处理自己的那一段；
//   2. 核内再把本段继续切 tile，借助 TQue + double buffer 做 DMA/Vector 流水；
//   3. DMA 统一使用 DataCopyPad，因此任意尾块（非 32B 整数倍）长度都能正确搬运；
//   4. Vector 计算长度向上对齐到 32B：多算出的元素只落在 UB 内，不会被写回 GM。
#include "kernel_operator.h"

#include "mul_tiling.h"
#include "tiling_key_mul.h"

namespace {
constexpr int32_t BUFFER_NUM = 2;     // 每个队列的 buffer 个数（double buffer）
constexpr uint32_t BLOCK_BYTES = 32;  // 一次 DMA/Vector 访问的对齐粒度
}  // namespace

template <class DT_X>
class KernelMul {
public:
    __aicore__ inline KernelMul() {}

    // 参数均由 Host 侧 TilingFunc 计算好，通过 MulTilingData 下发
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t totalLength,
                                uint32_t perCoreLength, uint32_t tileLength, uint32_t alignElements)
    {
        totalLength_ = totalLength;
        perCoreLength_ = perCoreLength;
        tileLength_ = tileLength;
        alignElements_ = alignElements;

        // 兜底：tiling 异常时保证除零/越界不会发生
        if (alignElements_ == 0) {
            alignElements_ = BLOCK_BYTES / static_cast<uint32_t>(sizeof(DT_X));
        }
        if (tileLength_ == 0) {
            tileLength_ = alignElements_;
        }
        if (totalLength_ == 0) {
            coreLength_ = 0;
            return;
        }

        // 当前核负责的全局区间起点：blockIdx * perCoreLength
        const uint64_t begin = static_cast<uint64_t>(AscendC::GetBlockIdx()) *
                               static_cast<uint64_t>(perCoreLength_);
        if (begin >= static_cast<uint64_t>(totalLength_)) {
            // 数据量太小、分到的核没有任务
            coreLength_ = 0;
            return;
        }

        const uint32_t remain = totalLength_ - static_cast<uint32_t>(begin);
        coreLength_ = (remain < perCoreLength_) ? remain : perCoreLength_;
        loopCount_ = (coreLength_ + tileLength_ - 1) / tileLength_;

        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(x) + begin, coreLength_);
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(y) + begin, coreLength_);
        zGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(z) + begin, coreLength_);

        // tileLength_ 已按 32B 对齐，因此每个 buffer 的字节数也是 32B 的整数倍
        const uint32_t tileBytes = tileLength_ * static_cast<uint32_t>(sizeof(DT_X));
        pipe_.InitBuffer(inQueueX_, BUFFER_NUM, tileBytes);
        pipe_.InitBuffer(inQueueY_, BUFFER_NUM, tileBytes);
        pipe_.InitBuffer(outQueueZ_, BUFFER_NUM, tileBytes);
    }

    __aicore__ inline void Process()
    {
        for (uint32_t progress = 0; progress < loopCount_; ++progress) {
            CopyIn(progress);
            Compute(progress);
            CopyOut(progress);
        }
    }

private:
    // 第 progress 个 tile 实际需要处理的元素个数（最后一个 tile 可能更短）
    __aicore__ inline uint32_t TileLengthOf(uint32_t progress) const
    {
        const uint32_t offset = progress * tileLength_;
        return (offset + tileLength_ <= coreLength_) ? tileLength_ : (coreLength_ - offset);
    }

    __aicore__ inline void CopyIn(uint32_t progress)
    {
        AscendC::LocalTensor<DT_X> xLocal = inQueueX_.AllocTensor<DT_X>();
        AscendC::LocalTensor<DT_X> yLocal = inQueueY_.AllocTensor<DT_X>();

        const uint32_t offset = progress * tileLength_;
        const uint32_t curLength = TileLengthOf(progress);
        const uint32_t curBytes = curLength * static_cast<uint32_t>(sizeof(DT_X));

        // blockCount = 1，blockLen 使用字节数：DataCopyPad 支持非 32B 整数倍的长度
        AscendC::DataCopyExtParams copyParams{1, curBytes, 0, 0, 0};
        AscendC::DataCopyPadExtParams<DT_X> padParams;  // isPad = false，不做补 pad

        AscendC::DataCopyPad(xLocal, xGm_[offset], copyParams, padParams);
        AscendC::DataCopyPad(yLocal, yGm_[offset], copyParams, padParams);

        inQueueX_.EnQue(xLocal);
        inQueueY_.EnQue(yLocal);
    }

    __aicore__ inline void Compute(uint32_t progress)
    {
        AscendC::LocalTensor<DT_X> xLocal = inQueueX_.DeQue<DT_X>();
        AscendC::LocalTensor<DT_X> yLocal = inQueueY_.DeQue<DT_X>();
        AscendC::LocalTensor<DT_X> zLocal = outQueueZ_.AllocTensor<DT_X>();

        // Vector 指令的计算长度对齐到 32B：多算的元素只在 UB 内，不会被写回 GM
        uint32_t calCount = (TileLengthOf(progress) + alignElements_ - 1) / alignElements_ * alignElements_;
        if (calCount > tileLength_) {
            calCount = tileLength_;
        }
        // buffer 是按 tileLength_ 分配的，计算长度不会越界
        AscendC::Mul(zLocal, xLocal, yLocal, calCount);

        outQueueZ_.EnQue(zLocal);
        inQueueX_.FreeTensor(xLocal);
        inQueueY_.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOut(uint32_t progress)
    {
        AscendC::LocalTensor<DT_X> zLocal = outQueueZ_.DeQue<DT_X>();

        const uint32_t offset = progress * tileLength_;
        const uint32_t curLength = TileLengthOf(progress);
        const uint32_t curBytes = curLength * static_cast<uint32_t>(sizeof(DT_X));

        AscendC::DataCopyExtParams copyParams{1, curBytes, 0, 0, 0};
        AscendC::DataCopyPad(zGm_[offset], zLocal, copyParams);

        outQueueZ_.FreeTensor(zLocal);
    }

private:
    AscendC::TPipe pipe_;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX_;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueY_;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueZ_;
    AscendC::GlobalTensor<DT_X> xGm_;
    AscendC::GlobalTensor<DT_X> yGm_;
    AscendC::GlobalTensor<DT_X> zGm_;

    uint32_t totalLength_ = 0;    // 总元素个数
    uint32_t perCoreLength_ = 0;  // 每核负责的元素个数
    uint32_t tileLength_ = 0;     // 单 tile 元素个数
    uint32_t alignElements_ = 0;  // 32B 对应元素个数
    uint32_t coreLength_ = 0;     // 本核实际负责的元素个数
    uint32_t loopCount_ = 0;      // 本核 tile 循环次数
};

template <typename DT_X>
__global__ __aicore__ void mul(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    (void)workspace;  // 本算子不需要 workspace
    REGISTER_TILING_DEFAULT(MulTilingData);
    GET_TILING_DATA_WITH_STRUCT(MulTilingData, tiling_data, tiling);
    KernelMul<DT_X> op;
    op.Init(x, y, z, tiling_data.totalLength, tiling_data.perCoreLength,
            tiling_data.tileLength, tiling_data.alignElements);
    op.Process();
}
