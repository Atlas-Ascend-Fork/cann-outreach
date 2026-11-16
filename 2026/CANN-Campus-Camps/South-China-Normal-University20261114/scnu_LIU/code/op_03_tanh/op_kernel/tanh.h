#ifndef TANH_H
#define TANH_H
#include "kernel_operator.h"
#include "tanh_tiling_data.h"
#include "tanh_tiling_key.h"
namespace NsTanh {
using namespace AscendC;
// Candidate optimized for the problem's fixed 8*2048 contiguous input.
// All intermediate arithmetic remains float32. No reciprocal approximation.
template<class T> class Tanh {
    GlobalTensor<T> xG, yG;
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const TanhTilingData*) {
        constexpr uint32_t n=2048;
        xG.SetGlobalBuffer((__gm__ T*)x+GetBlockIdx()*n,n);
        yG.SetGlobalBuffer((__gm__ T*)y+GetBlockIdx()*n,n);
    }
    __aicore__ inline void Process() {
        constexpr uint32_t n=2048;

        // UB byte offsets are disjoint and aligned to 32 bytes.
        LocalTensor<T> x(TPosition::VECCALC,0,n);
        LocalTensor<float> a(TPosition::VECCALC,(sizeof(T)==2 ? 8448 : 8224),n);
        LocalTensor<float> b(TPosition::VECCALC,(sizeof(T)==2 ? 16896 : 16448),n);
        LocalTensor<T> y(TPosition::VECCALC,(sizeof(T)==2 ? 25344 : 24672),n);
        DataCopy(x,xG,n);
        SetFlag<HardEvent::MTE2_V>(0);
        WaitFlag<HardEvent::MTE2_V>(0);
        if constexpr(sizeof(T)==2) {
            Cast(a,x,RoundMode::CAST_NONE,n);
            PipeBarrier<PIPE_V>();
            Muls(a,a,-2.0f,n);
        } else {
            Muls(a,x,-2.0f,n);
        }
        PipeBarrier<PIPE_V>(); Exp(a,a,n);
        PipeBarrier<PIPE_V>(); Adds(a,a,1.0f,n);
        Duplicate(b,2.0f,n);
        PipeBarrier<PIPE_V>(); Div(b,b,a,n);
        PipeBarrier<PIPE_V>(); Adds(b,b,-1.0f,n);
        PipeBarrier<PIPE_V>();
        if constexpr(sizeof(T)==2) {
            Cast(y,b,RoundMode::CAST_RINT,n);
            SetFlag<HardEvent::V_MTE3>(0);
            WaitFlag<HardEvent::V_MTE3>(0);
            DataCopy(yG,y,n);
        } else {
            SetFlag<HardEvent::V_MTE3>(0);
            WaitFlag<HardEvent::V_MTE3>(0);
            DataCopy(yG,b,n);
        }
    }
};
}
#endif
