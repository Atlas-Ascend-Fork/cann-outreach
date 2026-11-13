/*!
 * \file abs.h
 * \brief Abs 算子 kernel 类定义（专用 shape 8x2048 = 16384，整块 UB 驻留单次搬运）
 */

#ifndef ABS_H
#define ABS_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "abs_tiling_data.h"
#include "abs_tiling_key.h"

namespace NsAbs {

using namespace AscendC;

// 专用 shape：8 x 2048 = 16384 元素
constexpr uint32_t TOTAL_LENGTH = 8U * 2048U;

template <typename DT_X>
class Abs {
public:
    __aicore__ inline Abs() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y)
    {
        // 32B 对齐块；每个核分到的元素数天然是 32B 对齐的整数倍
        constexpr uint32_t ALIGN_NUM = 32U / sizeof(DT_X);
        constexpr uint32_t TOTAL_BLOCKS = TOTAL_LENGTH / ALIGN_NUM;

        uint32_t coreNum = AscendC::GetBlockNum();
        uint32_t blocksPerCore = TOTAL_BLOCKS / coreNum;
        uint32_t tailBlockNum = TOTAL_BLOCKS % coreNum; // 大核个数

        uint32_t blockIdx = AscendC::GetBlockIdx();
        uint32_t isBig = (blockIdx < tailBlockNum) ? 1U : 0U;
        uint32_t smallCores = (blockIdx < tailBlockNum) ? blockIdx : tailBlockNum;

        this->coreLength = (blocksPerCore + isBig) * ALIGN_NUM;
        uint64_t offset = (static_cast<uint64_t>(blockIdx) * blocksPerCore +
                           static_cast<uint64_t>(smallCores)) * ALIGN_NUM;

        inputGMX.SetGlobalBuffer((__gm__ DT_X*)x + offset, this->coreLength);
        outputGMY.SetGlobalBuffer((__gm__ DT_X*)y + offset, this->coreLength);
        pipe.InitBuffer(bufX, this->coreLength * sizeof(DT_X));
    }

    __aicore__ inline void Process()
    {
        // 整块数据一次搬入 UB，原地 Abs，再一次搬出
        LocalTensor<DT_X> xLocal = bufX.Get<DT_X>();
        DataCopy(xLocal, inputGMX, this->coreLength);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);

        AscendC::Abs(xLocal, xLocal, this->coreLength);

        SetFlag<HardEvent::V_MTE3>(EVENT_ID1);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID1);
        DataCopy(outputGMY, xLocal, this->coreLength);
    }

private:
    TPipe pipe;
    TBuf<TPosition::VECCALC> bufX;
    GlobalTensor<DT_X> inputGMX;
    GlobalTensor<DT_X> outputGMY;
    uint32_t coreLength;
};

} // namespace NsAbs
#endif // ABS_H
