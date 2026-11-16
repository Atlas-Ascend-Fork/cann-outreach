#ifndef RELU_H
#define RELU_H
#include "kernel_operator.h"
#include "relu_tiling_data.h"
#include "relu_tiling_key.h"
namespace NsRelu {
using namespace AscendC;
template<class T> class Relu {
    GlobalTensor<T> xG,yG;
    uint32_t length=0;
    bool small=false;
    static constexpr uint32_t TILE=32768/sizeof(T);
    static constexpr uint32_t STRIDE=33024;
public:
    __aicore__ inline void Init(GM_ADDR x,GM_ADDR y,const ReluTilingData* d) {
        uint64_t start=uint64_t(GetBlockIdx())*d->blockFactor;
        if(start>=uint64_t(d->totalNum)) return;
        uint64_t remaining=d->totalNum-start;
        length=remaining<uint64_t(d->blockFactor)?remaining:d->blockFactor;
        small=d->totalNum==16384 && d->blockFactor==2048;
        xG.SetGlobalBuffer((__gm__ T*)x+start,length);
        yG.SetGlobalBuffer((__gm__ T*)y+start,length);
    }
    __aicore__ inline void Process() {
        if(length==0) return;
        if(small) {
            LocalTensor<T> x(TPosition::VECCALC,0,2048);
            LocalTensor<T> y(TPosition::VECCALC,8448,2048);
            DataCopy(x,xG,2048);
            SetFlag<HardEvent::MTE2_V>(0); WaitFlag<HardEvent::MTE2_V>(0);
            AscendC::Relu(y,x,2048);
            SetFlag<HardEvent::V_MTE3>(0); WaitFlag<HardEvent::V_MTE3>(0);
            DataCopy(yG,y,2048);
            SetFlag<HardEvent::MTE3_S>(0); WaitFlag<HardEvent::MTE3_S>(0);
            return;
        }
        LocalTensor<T> x[2]={LocalTensor<T>(TPosition::VECCALC,0,TILE),
                            LocalTensor<T>(TPosition::VECCALC,STRIDE,TILE)};
        LocalTensor<T> y[2]={LocalTensor<T>(TPosition::VECCALC,2*STRIDE,TILE),
                            LocalTensor<T>(TPosition::VECCALC,3*STRIDE,TILE)};
        uint32_t loops=(length+TILE-1)/TILE;
        // Prime both input slots before the steady-state pipeline.
        for(uint32_t i=0;i<2 && i<loops;++i) {
            uint32_t n=length-i*TILE<TILE?length-i*TILE:TILE;
            DataCopy(x[i],xG[i*TILE],n);
            SetFlag<HardEvent::MTE2_V>(i);
        }
        for(uint32_t i=0;i<loops;++i) {
            uint32_t s=i&1;
            uint32_t n=length-i*TILE<TILE?length-i*TILE:TILE;
            WaitFlag<HardEvent::MTE2_V>(s);
            if(i>=2) WaitFlag<HardEvent::MTE3_V>(s);
            AscendC::Relu(y[s],x[s],n);
            SetFlag<HardEvent::V_MTE2>(s);
            SetFlag<HardEvent::V_MTE3>(s); WaitFlag<HardEvent::V_MTE3>(s);
            DataCopy(yG[i*TILE],y[s],n);
            SetFlag<HardEvent::MTE3_V>(s);
            WaitFlag<HardEvent::V_MTE2>(s);
            if(i+2<loops) {
                uint32_t next=(i+2)*TILE;
                uint32_t count=length-next<TILE?length-next:TILE;
                DataCopy(x[s],xG[next],count);
                SetFlag<HardEvent::MTE2_V>(s);
            }
        }
        // Drain outstanding stores before leaving the kernel.
        WaitFlag<HardEvent::MTE3_V>(0);
        if(loops>1) WaitFlag<HardEvent::MTE3_V>(1);
    }
};
}
#endif
