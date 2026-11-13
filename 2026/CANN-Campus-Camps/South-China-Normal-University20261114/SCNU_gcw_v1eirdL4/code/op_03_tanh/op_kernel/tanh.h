#ifndef TANH_H
#define TANH_H

#include "kernel_operator.h"

namespace tanh_op {
using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;     // 输入/输出队列双缓冲深度
constexpr uint32_t ALIGN_BYTES = 32;  // UB 搬运 32B 对齐
constexpr float CLAMP_VALUE = 8.0f;   // exp 溢出保护阈值

// 编译期分发：fp16 需要进/出各一次 Cast，fp32 直接算（C++11 也能编）
template <typename T>
struct TanhCast {
    __aicore__ static inline void ToFloat(LocalTensor<float>& src, const LocalTensor<T>& xLocal,
                                          const LocalTensor<float>& xf, uint32_t count)
    {
        Cast(xf, xLocal, RoundMode::CAST_NONE, count);
        src = xf;
    }
    __aicore__ static inline void ToDst(LocalTensor<T>& yLocal, LocalTensor<float>& num,
                                        LocalTensor<float>& den, uint32_t count)
    {
        Div(num, num, den, count);
        Cast(yLocal, num, RoundMode::CAST_NONE, count);
    }
};

template <>
struct TanhCast<float> {
    __aicore__ static inline void ToFloat(LocalTensor<float>& src, const LocalTensor<float>& xLocal,
                                          const LocalTensor<float>& xf, uint32_t count)
    {
        (void)xf;
        (void)count;
        src = xLocal;
    }
    __aicore__ static inline void ToDst(LocalTensor<float>& yLocal, LocalTensor<float>& num,
                                        LocalTensor<float>& den, uint32_t count)
    {
        Div(yLocal, num, den, count);
    }
};

template <typename T>
class KernelTanh {
public:
    __aicore__ inline KernelTanh() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLen,
                                uint32_t coreNum, uint32_t tileLen, uint32_t alignCnt)
    {
        blockIdx = static_cast<uint32_t>(GetBlockIdx());
        totalLength = totalLen;
        coreDataNum = coreNum;
        tileLength = tileLen;
        alignCount = (alignCnt == 0U) ? 1U : alignCnt;

        coreOffset = blockIdx * coreDataNum;
        coreLength = 0;
        if (coreOffset < totalLength) {
            uint32_t left = totalLength - coreOffset;
            coreLength = (left < coreDataNum) ? left : coreDataNum;
        }

        xGm.SetGlobalBuffer((__gm__ T*)x + coreOffset, coreLength);
        yGm.SetGlobalBuffer((__gm__ T*)y + coreOffset, coreLength);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, tileLength * sizeof(T));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, tileLength * sizeof(T));
        pipe.InitBuffer(bufXf, tileLength * sizeof(float));
        pipe.InitBuffer(bufE1, tileLength * sizeof(float));
        pipe.InitBuffer(bufE2, tileLength * sizeof(float));
        pipe.InitBuffer(bufNum, tileLength * sizeof(float));
        pipe.InitBuffer(bufDen, tileLength * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        uint32_t offset = 0;
        while (offset < coreLength) {
            uint32_t left = coreLength - offset;
            uint32_t count = (left < tileLength) ? left : tileLength;
            CopyIn(offset, count);
            Compute(count);
            CopyOut(offset, count);
            offset += count;
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t offset, uint32_t count)
    {
        LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
        if (count % alignCount == 0) {
            DataCopy(xLocal, xGm[offset], count);
        } else {
            DataCopyExtParams copyParams{1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
            DataCopyPadExtParams<T> padParams{false, 0, 0, static_cast<T>(0.0f)};
            DataCopyPad(xLocal, xGm[offset], copyParams, padParams);
        }
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void CopyOut(uint32_t offset, uint32_t count)
    {
        LocalTensor<T> yLocal = outQueueY.DeQue<T>();
        if (count % alignCount == 0) {
            DataCopy(yGm[offset], yLocal, count);
        } else {
            DataCopyExtParams copyParams{1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
            DataCopyPad(yGm[offset], yLocal, copyParams);
        }
        outQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void Compute(uint32_t count)
    {
        LocalTensor<T> xLocal = inQueueX.DeQue<T>();
        LocalTensor<T> yLocal = outQueueY.AllocTensor<T>();

        LocalTensor<float> e1 = bufE1.Get<float>();
        LocalTensor<float> e2 = bufE2.Get<float>();
        LocalTensor<float> num = bufNum.Get<float>();
        LocalTensor<float> den = bufDen.Get<float>();
        LocalTensor<float> xf = bufXf.Get<float>();
        LocalTensor<float> src;

        // 1) 统一到 fp32
        TanhCast<T>::ToFloat(src, xLocal, xf, count);

        // 2) clamp 到 [-8, 8]（src 恒为 fp32 张量，标量必须是 float）
        Maxs(src, src, -CLAMP_VALUE, count);
        Mins(src, src, CLAMP_VALUE, count);

        // 3) e1 = exp(x), e2 = exp(-x)
        Exp(e1, src, count);
        Muls(num, src, -1.0f, count);
        Exp(e2, num, count);

        // 4) (e1 - e2) / (e1 + e2)
        Sub(num, e1, e2, count);
        Add(den, e1, e2, count);

        // 5) 写回（fp16 需要 Cast 回半精度）
        TanhCast<T>::ToDst(yLocal, num, den, count);

        inQueueX.FreeTensor(xLocal);
        outQueueY.EnQue(yLocal);
    }

    TPipe pipe;
    TQue<TPosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<TPosition::VECOUT, BUFFER_NUM> outQueueY;
    TBuf<TPosition::VECCALC> bufXf;
    TBuf<TPosition::VECCALC> bufE1;
    TBuf<TPosition::VECCALC> bufE2;
    TBuf<TPosition::VECCALC> bufNum;
    TBuf<TPosition::VECCALC> bufDen;

    GlobalTensor<T> xGm;
    GlobalTensor<T> yGm;

    uint32_t blockIdx = 0;
    uint32_t totalLength = 0;
    uint32_t coreDataNum = 0;
    uint32_t coreOffset = 0;
    uint32_t coreLength = 0;
    uint32_t tileLength = 0;
    uint32_t alignCount = 1;
};

}  // namespace tanh_op

#endif  // TANH_H