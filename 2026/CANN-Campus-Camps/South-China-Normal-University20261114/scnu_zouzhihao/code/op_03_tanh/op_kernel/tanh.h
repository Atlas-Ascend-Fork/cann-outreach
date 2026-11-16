/*!
 * \file tanh.h
 * \brief Tanh 算子 kernel 类定义
 */

#ifndef TANH_H
#define TANH_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "tanh_tiling_data.h"
#include "tanh_tiling_key.h"

namespace NsTanh {

using namespace AscendC;

constexpr int32_t EV_MTE2_V = 0;
constexpr int32_t EV_V_MTE3 = 1;

// y = tanh(x) = (e^(2x) - 1) / (e^(2x) + 1)
// 每条 vec 指令只走一遍 UB，整个 kernel 只有 5 遍：
//   Muls(2x) -> Exp -> Adds(+1) -> Adds(-1) -> Div
// 全程 fp32 计算（half 输入用 Cast 转换），保证半精度输出与参考实现同值。

template <typename T>
class Tanh {
public:
    __aicore__ inline Tanh() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const TanhTilingData* tilingData)
    {
        int64_t total = tilingData->totalNum;
        int64_t perCore = tilingData->blockFactor;
        int64_t idx = AscendC::GetBlockIdx();
        int64_t start = idx * perCore;
        int64_t remain = total - start;

        if (remain <= 0) {
            length_ = 0;
            return;
        }
        int64_t len = (remain < perCore) ? remain : perCore;
        length_ = static_cast<uint32_t>(len);

        inputGMX.SetGlobalBuffer((__gm__ T*)x + start, length_);
        outputGMY.SetGlobalBuffer((__gm__ T*)y + start, length_);

        uint32_t bytesF = length_ * sizeof(float);
        pipe.InitBuffer(bufT, bytesF);
        pipe.InitBuffer(bufU, bytesF);
        if constexpr (sizeof(T) != sizeof(float)) {
            pipe.InitBuffer(bufHalf, length_ * sizeof(T));
        }
    }

    __aicore__ inline void Process()
    {
        if (length_ == 0) {
            return;
        }
        LocalTensor<float> t = bufT.Get<float>();
        LocalTensor<float> u = bufU.Get<float>();

        if constexpr (sizeof(T) == sizeof(float)) {
            DataCopy(t, inputGMX, length_);
            SetFlag<HardEvent::MTE2_V>(EV_MTE2_V);
            WaitFlag<HardEvent::MTE2_V>(EV_MTE2_V);

            Muls(t, t, 2.0f, length_);
            ComputeTanh(t, u);

            SetFlag<HardEvent::V_MTE3>(EV_V_MTE3);
            WaitFlag<HardEvent::V_MTE3>(EV_V_MTE3);
            DataCopy(outputGMY, u, length_);
        } else {
            LocalTensor<T> h = bufHalf.Get<T>();
            DataCopy(h, inputGMX, length_);
            SetFlag<HardEvent::MTE2_V>(EV_MTE2_V);
            WaitFlag<HardEvent::MTE2_V>(EV_MTE2_V);

            Cast(t, h, RoundMode::CAST_NONE, length_);
            Muls(t, t, 2.0f, length_);
            ComputeTanh(t, u);
            Cast(h, u, RoundMode::CAST_RINT, length_);

            SetFlag<HardEvent::V_MTE3>(EV_V_MTE3);
            WaitFlag<HardEvent::V_MTE3>(EV_V_MTE3);
            DataCopy(outputGMY, h, length_);
        }
    }

private:
    // 入口：t = 2x，出口：u = tanh(x)
    __aicore__ inline void ComputeTanh(const LocalTensor<float>& t, const LocalTensor<float>& u)
    {
        Exp(u, t, length_);
        Adds(t, u, 1.0f, length_);
        Adds(u, u, -1.0f, length_);
        Div(u, u, t, length_);
    }

private:
    TPipe pipe;
    TBuf<TPosition::VECCALC> bufT;
    TBuf<TPosition::VECCALC> bufU;
    TBuf<TPosition::VECCALC> bufHalf;

    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;

    uint32_t length_ = 0;
};

} // namespace NsTanh
#endif // TANH_H