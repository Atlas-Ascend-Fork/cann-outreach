#include "kernel_operator.h"

#include "mul_tiling.h"
#include "tiling_key_mul.h"

template <class DT_X>
class KernelMul {
public:
    __aicore__ inline KernelMul() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t totalLength)
    {
        uint32_t blockNum = AscendC::GetBlockNum();
        uint32_t blockIdx = AscendC::GetBlockIdx();
        this->thisLength = totalLength / blockNum;
        uint32_t offset = blockIdx * this->thisLength;

        xGm.SetGlobalBuffer((__gm__ DT_X *)x + offset, this->thisLength);
        yGm.SetGlobalBuffer((__gm__ DT_X *)y + offset, this->thisLength);
        zGm.SetGlobalBuffer((__gm__ DT_X *)z + offset, this->thisLength);
    }

    __aicore__ inline void Process()
    {
        uint32_t xAddr = 0;
        uint32_t yAddr = this->thisLength * sizeof(DT_X);
        uint32_t zAddr = this->thisLength * sizeof(DT_X) * 2;
        AscendC::LocalTensor<DT_X> xLocal(AscendC::TPosition::VECIN, xAddr, this->thisLength);
        AscendC::LocalTensor<DT_X> yLocal(AscendC::TPosition::VECIN, yAddr, this->thisLength);
        AscendC::LocalTensor<DT_X> zLocal(AscendC::TPosition::VECOUT, zAddr, this->thisLength);

        AscendC::DataCopy(xLocal, xGm, this->thisLength);
        AscendC::DataCopy(yLocal, yGm, this->thisLength);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);

        AscendC::Mul(zLocal, xLocal, yLocal, this->thisLength);

        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
        AscendC::DataCopy(zGm, zLocal, this->thisLength);
    }

private:
    AscendC::GlobalTensor<DT_X> xGm;
    AscendC::GlobalTensor<DT_X> yGm;
    AscendC::GlobalTensor<DT_X> zGm;
    uint32_t thisLength;
};

template <typename DT_X>
__global__ __aicore__ void mul(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling)
{
    AscendC::InitSocState();
    REGISTER_TILING_DEFAULT(MulTilingData);
    GET_TILING_DATA_WITH_STRUCT(MulTilingData, tiling_data, tiling);
    KernelMul<DT_X> op;
    op.Init(x, y, z, tiling_data.length);
    op.Process();
}
