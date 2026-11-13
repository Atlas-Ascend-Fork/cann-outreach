/*!
 * \file abs.h
 * \brief Abs 算子 kernel 类定义
 */

#ifndef ABS_H
#define ABS_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "abs_tiling_data.h"
#include "abs_tiling_key.h"

namespace NsAbs {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

template <typename DT_X>
class Abs {
public:
    __aicore__ inline Abs() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const AbsTilingData* tilingData)
    {
        const int64_t offset = static_cast<int64_t>(GetBlockIdx()) * tilingData->blockFactor;
        const int64_t remaining = tilingData->totalNum - offset;
        blockLength = remaining < tilingData->blockFactor ? remaining : tilingData->blockFactor;
        tileLength = tilingData->ubFactor;
        if (blockLength <= 0 || tileLength <= 0) {
            return;
        }
        xGm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X*>(x) + offset, blockLength);
        yGm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X*>(y) + offset, blockLength);
        pipe.InitBuffer(input, BUFFER_NUM, tileLength * sizeof(DT_X));
        pipe.InitBuffer(output, BUFFER_NUM, tileLength * sizeof(DT_X));
    }

    __aicore__ inline void Process()
    {
        if (blockLength <= 0 || tileLength <= 0) {
            return;
        }
        for (int64_t offset = 0; offset < blockLength; offset += tileLength) {
            const int64_t remaining = blockLength - offset;
            const int64_t count = remaining < tileLength ? remaining : tileLength;
            CopyIn(offset, count);
            Compute(count);
            CopyOut(offset, count);
        }
    }

private:
    __aicore__ inline void CopyIn(int64_t offset, int64_t count)
    {
        auto x = input.AllocTensor<DT_X>();
        DataCopyExtParams params{1, static_cast<uint32_t>(count * sizeof(DT_X)), 0, 0, 0};
        DataCopyPadExtParams<DT_X> pad{false, 0, 0, 0};
        DataCopyPad(x, xGm[offset], params, pad);
        input.EnQue(x);
    }

    __aicore__ inline void Compute(int64_t count)
    {
        auto x = input.DeQue<DT_X>();
        auto y = output.AllocTensor<DT_X>();
        AscendC::Abs(y, x, count);
        output.EnQue(y);
        input.FreeTensor(x);
    }

    __aicore__ inline void CopyOut(int64_t offset, int64_t count)
    {
        auto y = output.DeQue<DT_X>();
        DataCopyExtParams params{1, static_cast<uint32_t>(count * sizeof(DT_X)), 0, 0, 0};
        DataCopyPad(yGm[offset], y, params);
        output.FreeTensor(y);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> input;
    TQue<QuePosition::VECOUT, BUFFER_NUM> output;
    GlobalTensor<DT_X> xGm;
    GlobalTensor<DT_X> yGm;
    int64_t blockLength = 0;
    int64_t tileLength = 0;
};

} // namespace NsAbs
#endif // ABS_H
