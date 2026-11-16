// Kernel侧核函数实现
#include "kernel_operator.h"

#include "mul_tiling.h"
#include "tiling_key_mul.h"

constexpr int32_t EV_MTE2_V = 0;
constexpr int32_t EV_V_MTE3 = 1;

template <class DT_X>
class KernelMul {
public:
    __aicore__ inline KernelMul() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, const MulTilingData& tiling)
    {
        constexpr uint32_t alignElements = 32 / sizeof(DT_X);
        uint32_t idx = static_cast<uint32_t>(AscendC::GetBlockIdx());
        uint32_t extra = tiling.extra;
        uint32_t low = (idx < extra) ? idx : extra;
        uint32_t startBlock = tiling.blocksPerCore * idx + low;

        this->length = (tiling.blocksPerCore + ((idx < extra) ? 1u : 0u)) * alignElements;
        if (this->length == 0) {
            return;
        }
        uint32_t offset = startBlock * alignElements;
        xGm.SetGlobalBuffer((__gm__ DT_X *)x + offset, this->length);
        yGm.SetGlobalBuffer((__gm__ DT_X *)y + offset, this->length);
        zGm.SetGlobalBuffer((__gm__ DT_X *)z + offset, this->length);

        uint32_t bytes = this->length * sizeof(DT_X);
        pipe.InitBuffer(xBuf, bytes);
        pipe.InitBuffer(yBuf, bytes);
        pipe.InitBuffer(zBuf, bytes);
    }

    __aicore__ inline void Process()
    {
        if (this->length == 0) {
            return;
        }
        AscendC::LocalTensor<DT_X> xLocal = xBuf.Get<DT_X>();
        AscendC::LocalTensor<DT_X> yLocal = yBuf.Get<DT_X>();
        AscendC::LocalTensor<DT_X> zLocal = zBuf.Get<DT_X>();

        AscendC::DataCopy(xLocal, xGm, this->length);
        AscendC::DataCopy(yLocal, yGm, this->length);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EV_MTE2_V);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EV_MTE2_V);

        AscendC::Mul(zLocal, xLocal, yLocal, this->length);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EV_V_MTE3);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EV_V_MTE3);

        AscendC::DataCopy(zGm, zLocal, this->length);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> xBuf, yBuf, zBuf;
    AscendC::GlobalTensor<DT_X> xGm, yGm, zGm;
    uint32_t length;
};

template <typename DT_X>
__global__ __aicore__ void mul(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(MulTilingData);
    GET_TILING_DATA_WITH_STRUCT(MulTilingData, tiling_data, tiling);
    KernelMul<DT_X> op;
    op.Init(x, y, z, tiling_data);
    op.Process();
}