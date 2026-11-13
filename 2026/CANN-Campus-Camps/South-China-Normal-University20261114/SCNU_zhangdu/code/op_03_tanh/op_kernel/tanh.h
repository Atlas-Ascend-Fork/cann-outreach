#include "kernel_operator.h"
#include "tanh_tiling_data.h"
#include "tanh_tiling_key.h"

constexpr int32_t BUFFER_NUM = 2; // 双缓冲提升速度

namespace NsTanh {

template <class DT_X>
class Tanh {
public:
    __aicore__ inline Tanh() {}
    
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const TanhTilingData* tilingData) {
        this->blockLength = tilingData->totalLength / AscendC::GetBlockNum();
        this->tileNum = tilingData->tileNum;
        this->tileLength = this->blockLength / this->tileNum / BUFFER_NUM;
        
        xGm.SetGlobalBuffer((__gm__ DT_X*)x + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        yGm.SetGlobalBuffer((__gm__ DT_X*)y + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(DT_X));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileLength * sizeof(DT_X));
    }
    
    __aicore__ inline void Process() {
        int32_t loopCount = this->tileNum * BUFFER_NUM;
        for (int32_t i = 0; i < loopCount; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        AscendC::LocalTensor<DT_X> xLocal = inQueueX.AllocTensor<DT_X>();
        AscendC::DataCopy(xLocal, xGm[progress * this->tileLength], this->tileLength);
        inQueueX.EnQue(xLocal);
    }
    
    __aicore__ inline void Compute(int32_t progress)
    {
        AscendC::LocalTensor<DT_X> xLocal = inQueueX.DeQue<DT_X>();
        AscendC::LocalTensor<DT_X> yLocal = outQueueY.AllocTensor<DT_X>();
        
        // 【绝对最稳最快】直接调用硬件原生的 Tanh 指令
        AscendC::Tanh(yLocal, xLocal, this->tileLength);

        inQueueX.FreeTensor(xLocal);
        outQueueY.EnQue<DT_X>(yLocal);
    }
    
    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<DT_X> yLocal = outQueueY.DeQue<DT_X>();
        AscendC::DataCopy(yGm[progress * this->tileLength], yLocal, this->tileLength);
        outQueueY.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    
    AscendC::GlobalTensor<DT_X> xGm;
    AscendC::GlobalTensor<DT_X> yGm;
    
    uint32_t blockLength; 
    uint32_t tileNum;  
    uint32_t tileLength;  
};

} // namespace NsTanh