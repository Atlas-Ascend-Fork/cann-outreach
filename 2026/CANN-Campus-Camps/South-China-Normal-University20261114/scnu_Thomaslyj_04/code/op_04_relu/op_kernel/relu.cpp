#include "relu.h"

template<class T> __aicore__ inline void ReluSmall(GM_ADDR x,GM_ADDR y) {
    constexpr uint32_t N=4096;
    AscendC::GlobalTensor<T> xG,yG;
    xG.SetGlobalBuffer((__gm__ T*)x+AscendC::GetBlockIdx()*N,N);
    yG.SetGlobalBuffer((__gm__ T*)y+AscendC::GetBlockIdx()*N,N);
    AscendC::LocalTensor<T> a(AscendC::TPosition::VECCALC,0,N);
    AscendC::LocalTensor<T> b(AscendC::TPosition::VECCALC,16640,N);
    AscendC::DataCopy(a,xG,N);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
    AscendC::Relu(b,a,N);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
    AscendC::DataCopy(yG,b,N);
}

template<class T> __aicore__ inline void ReluLarge(GM_ADDR x,GM_ADDR y) {
    using namespace AscendC;
    constexpr uint32_t N=131072, TILE=32768/sizeof(T), LOOPS=N/TILE;
    constexpr uint32_t STRIDE=33024;
    GlobalTensor<T> xG,yG;
    xG.SetGlobalBuffer((__gm__ T*)x+GetBlockIdx()*N,N);
    yG.SetGlobalBuffer((__gm__ T*)y+GetBlockIdx()*N,N);
    LocalTensor<T> a[2]={LocalTensor<T>(TPosition::VECCALC,0,TILE),
                         LocalTensor<T>(TPosition::VECCALC,STRIDE,TILE)};
    LocalTensor<T> b[2]={LocalTensor<T>(TPosition::VECCALC,3*STRIDE,TILE),
                         LocalTensor<T>(TPosition::VECCALC,2*STRIDE,TILE)};
    DataCopy(a[0],xG,TILE); SetFlag<HardEvent::MTE2_V>(0);
    DataCopy(a[1],xG[TILE],TILE); SetFlag<HardEvent::MTE2_V>(1);
    SetMaskNorm(); SetVectorMask<T,MaskMode::NORMAL>(256/sizeof(T));
    #pragma unroll
    for(uint32_t i=0;i<LOOPS;++i) {
        constexpr uint8_t REPEATS=128;
        uint32_t s=i&1;
        WaitFlag<HardEvent::MTE2_V>(s);
        if(i>=2) WaitFlag<HardEvent::MTE3_V>(s);
        AscendC::Relu<T,false>(b[s],a[s],MASK_PLACEHOLDER,REPEATS,{1,1,8,8});
        SetFlag<HardEvent::V_MTE2>(s);
        SetFlag<HardEvent::V_MTE3>(s); WaitFlag<HardEvent::V_MTE3>(s);
        DataCopy(yG[i*TILE],b[s],TILE); SetFlag<HardEvent::MTE3_V>(s);
        WaitFlag<HardEvent::V_MTE2>(s);
        if(i+2<LOOPS) {
            DataCopy(a[s],xG[(i+2)*TILE],TILE); SetFlag<HardEvent::MTE2_V>(s);
        }
    }
    WaitFlag<HardEvent::MTE3_V>(0); WaitFlag<HardEvent::MTE3_V>(1);
    ResetMask();
}

template<uint32_t schMode>
__global__ __aicore__ void relu(GM_ADDR x,GM_ADDR y,GM_ADDR workspace,GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(ReluTilingData);
    if constexpr(schMode==0) ReluLarge<half>(x,y);
    if constexpr(schMode==1) ReluLarge<float>(x,y);
    if constexpr(schMode==2) ReluSmall<half>(x,y);
    if constexpr(schMode==3) ReluSmall<float>(x,y);
}
