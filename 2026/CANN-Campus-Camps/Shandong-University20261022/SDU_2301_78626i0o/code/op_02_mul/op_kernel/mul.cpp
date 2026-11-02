// Kernel侧核函数实现
#include "kernel_operator.h"

#include "mul_tiling.h"
#include "tiling_key_mul.h"

constexpr int32_t BUFFER_NUM = 1;

template <class DT_X>
class KernelMul {
public:
    __aicore__ inline KernelMul() {}

    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR y,
        GM_ADDR z,
        uint32_t totalLength,
        uint32_t blockLength,
        uint32_t tileLength)
    {
        this->tileLength = tileLength;
        this->coreLength = 0;
        this->currentLength = 0;

        // 使用64位计算，避免空闲核的偏移乘法溢出。
        uint64_t offset =
            static_cast<uint64_t>(AscendC::GetBlockIdx())
            * blockLength;

        // 空输入或没有分配到有效数据的核直接返回。
        if (offset >= totalLength) {
            return;
        }

        uint32_t remaining =
            totalLength - static_cast<uint32_t>(offset);

        this->coreLength =
            remaining < blockLength ? remaining : blockLength;

        // 当前核负责的全局内存范围。
        xGm.SetGlobalBuffer(
            (__gm__ DT_X *)x + offset, this->coreLength);

        yGm.SetGlobalBuffer(
            (__gm__ DT_X *)y + offset, this->coreLength);

        zGm.SetGlobalBuffer(
            (__gm__ DT_X *)z + offset, this->coreLength);

        // 固定分配一个tile的空间，最后一块可少于tileLength。
        pipe.InitBuffer(
            inQueueX,
            BUFFER_NUM,
            this->tileLength * sizeof(DT_X));

        pipe.InitBuffer(
            inQueueY,
            BUFFER_NUM,
            this->tileLength * sizeof(DT_X));

        pipe.InitBuffer(
            outQueueZ,
            BUFFER_NUM,
            this->tileLength * sizeof(DT_X));
    }

    __aicore__ inline void Process()
    {
        if (this->coreLength == 0) {
            return;
        }

        uint32_t loopCount =
            this->coreLength / this->tileLength;

        if (this->coreLength % this->tileLength != 0) {
            ++loopCount;
        }

        for (uint32_t i = 0; i < loopCount; ++i) {
            uint32_t remaining =
                this->coreLength - i * this->tileLength;

            this->currentLength =
                remaining < this->tileLength
                    ? remaining
                    : this->tileLength;

            CopyIn(static_cast<int32_t>(i));
            Compute(static_cast<int32_t>(i));
            CopyOut(static_cast<int32_t>(i));
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        AscendC::LocalTensor<DT_X> xLocal =
            inQueueX.AllocTensor<DT_X>();

        AscendC::LocalTensor<DT_X> yLocal =
            inQueueY.AllocTensor<DT_X>();

        uint32_t offset =
            static_cast<uint32_t>(progress) * this->tileLength;

        // blockLen单位为字节，只搬运当前块的有效数据。
        AscendC::DataCopyExtParams copyParams{
            1,
            static_cast<uint32_t>(
                this->currentLength * sizeof(DT_X)),
            0,
            0,
            0
        };

        AscendC::DataCopyPadExtParams<DT_X> padParams{
            false, 0, 0, 0
        };

        AscendC::DataCopyPad(
            xLocal, xGm[offset], copyParams, padParams);

        AscendC::DataCopyPad(
            yLocal, yGm[offset], copyParams, padParams);

        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }

    __aicore__ inline void Compute(int32_t progress)
    {
        AscendC::LocalTensor<DT_X> xLocal =
            inQueueX.DeQue<DT_X>();

        AscendC::LocalTensor<DT_X> yLocal =
            inQueueY.DeQue<DT_X>();

        AscendC::LocalTensor<DT_X> zLocal =
            outQueueZ.AllocTensor<DT_X>();

        // 矢量乘法：仅计算有效元素。
        AscendC::Mul(
            zLocal, xLocal, yLocal, this->currentLength);

        outQueueZ.EnQue(zLocal);

        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<DT_X> zLocal =
            outQueueZ.DeQue<DT_X>();

        uint32_t offset =
            static_cast<uint32_t>(progress) * this->tileLength;

        AscendC::DataCopyExtParams copyParams{
            1,
            static_cast<uint32_t>(
                this->currentLength * sizeof(DT_X)),
            0,
            0,
            0
        };

        // 只写回有效元素，避免尾块越界。
        AscendC::DataCopyPad(
            zGm[offset], zLocal, copyParams);

        outQueueZ.FreeTensor(zLocal);
    }

private:
    AscendC::TPipe pipe;

    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM>
        inQueueX;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM>
        inQueueY;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM>
        outQueueZ;

    AscendC::GlobalTensor<DT_X> xGm;
    AscendC::GlobalTensor<DT_X> yGm;
    AscendC::GlobalTensor<DT_X> zGm;

    uint32_t coreLength;
    uint32_t tileLength;
    uint32_t currentLength;
};

template <typename DT_X>
__global__ __aicore__ void mul(
    GM_ADDR x,
    GM_ADDR y,
    GM_ADDR z,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(MulTilingData);
    GET_TILING_DATA_WITH_STRUCT(
        MulTilingData, tiling_data, tiling);

    KernelMul<DT_X> op;

    op.Init(
        x,
        y,
        z,
        tiling_data.totalLength,
        tiling_data.blockLength,
        tiling_data.tileLength);

    op.Process();
}