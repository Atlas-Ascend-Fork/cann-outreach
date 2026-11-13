// Kernel侧核函数实现
// Lesson 2 mul 自定义算子工程（平台「下载空工程」骨架版）
// 唯一需要实现的文件：KernelMul 的 Init / Process / CopyIn / Compute / CopyOut
#include "kernel_operator.h"

#include "mul_tiling.h"
#include "tiling_key_mul.h"

constexpr int32_t BUFFER_NUM = 2;              // Double Buffer
constexpr uint32_t TILE_BYTES = 8192;          // 每块每路 LocalTensor 字节数

template <class DT_X>
class KernelMul {
public:
    __aicore__ inline KernelMul() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t length) {
        uint32_t blockNum = AscendC::GetBlockNum();
        uint32_t blockIdx = AscendC::GetBlockIdx();
        if (blockNum == 0) {
            blockNum = 1;
        }

        // 32 字节对齐粒度（元素数）：fp32 = 8，fp16 = 16
        constexpr uint32_t ALIGN_ELEM = 32 / sizeof(DT_X);

        // host 侧 SetBlockDim(num_cores_aiv) 未必整除 length，
        // 每核长度向上对齐到 32B，保证各核 GM 起始地址对齐、且不丢元素
        uint32_t perCore = (length + blockNum - 1) / blockNum;
        perCore = (perCore + ALIGN_ELEM - 1) / ALIGN_ELEM * ALIGN_ELEM;
        uint32_t start = perCore * blockIdx;
        if (start >= length) {
            this->blockLength = 0;              // 超出数据范围的核空转
            return;
        }
        uint32_t rest = length - start;
        this->blockLength = (rest < perCore) ? rest : perCore;

        xGm.SetGlobalBuffer((__gm__ DT_X*)x + start, this->blockLength);
        yGm.SetGlobalBuffer((__gm__ DT_X*)y + start, this->blockLength);
        zGm.SetGlobalBuffer((__gm__ DT_X*)z + start, this->blockLength);

        // 每块元素数按 32B 对齐；三路队列各 BUFFER_NUM 块，总 UB 约 48 KB
        this->tileLength = TILE_BYTES / sizeof(DT_X);
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(DT_X));
        pipe.InitBuffer(inQueueY, BUFFER_NUM, this->tileLength * sizeof(DT_X));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileLength * sizeof(DT_X));
    }

    __aicore__ inline void Process() {
        if (this->blockLength == 0) {
            return;
        }
        uint32_t loopCount = (this->blockLength + this->tileLength - 1) / this->tileLength;
        for (uint32_t i = 0; i < loopCount; i++) {
            uint32_t offset = i * this->tileLength;
            // 最后一块为实际剩余元素数，避免尾块越界
            uint32_t currentNum = (i == loopCount - 1)
                                      ? (this->blockLength - offset)
                                      : this->tileLength;
            CopyIn(offset, currentNum);
            Compute(currentNum);
            CopyOut(offset, currentNum);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t offset, uint32_t currentNum) {
        AscendC::LocalTensor<DT_X> xLocal = inQueueX.AllocTensor<DT_X>();
        AscendC::LocalTensor<DT_X> yLocal = inQueueY.AllocTensor<DT_X>();
        AscendC::DataCopyParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = currentNum * sizeof(DT_X);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        // DataCopyPad 支持任意字节长度，尾块不会越界
        AscendC::DataCopyPad(xLocal, xGm[offset], copyParams, {false, 0, 0, 0});
        AscendC::DataCopyPad(yLocal, yGm[offset], copyParams, {false, 0, 0, 0});
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }

    __aicore__ inline void Compute(uint32_t currentNum) {
        AscendC::LocalTensor<DT_X> xLocal = inQueueX.DeQue<DT_X>();
        AscendC::LocalTensor<DT_X> yLocal = inQueueY.DeQue<DT_X>();
        AscendC::LocalTensor<DT_X> zLocal = outQueueZ.AllocTensor<DT_X>();

        // 矢量指令计算长度需 32B 对齐，向上取整；
        // 多算的部分仍在 buffer 内，CopyOut 只回拷 currentNum 个有效元素
        constexpr uint32_t ALIGN_ELEM = 32 / sizeof(DT_X);
        uint32_t calcNum = (currentNum + ALIGN_ELEM - 1) / ALIGN_ELEM * ALIGN_ELEM;
        AscendC::Mul(zLocal, xLocal, yLocal, calcNum);

        outQueueZ.EnQue<DT_X>(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOut(uint32_t offset, uint32_t currentNum) {
        AscendC::LocalTensor<DT_X> zLocal = outQueueZ.DeQue<DT_X>();
        AscendC::DataCopyParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = currentNum * sizeof(DT_X);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        AscendC::DataCopyPad(zGm[offset], zLocal, copyParams);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueY;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueZ;
    AscendC::GlobalTensor<DT_X> xGm;
    AscendC::GlobalTensor<DT_X> yGm;
    AscendC::GlobalTensor<DT_X> zGm;
    uint32_t blockLength = 0;   // 本核负责的元素数
    uint32_t tileLength = 0;    // 每块元素数
};

template <typename DT_X>
 __global__ __aicore__ void mul(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(MulTilingData);
    GET_TILING_DATA_WITH_STRUCT(MulTilingData, tiling_data, tiling);
    KernelMul<DT_X> op;
    op.Init(x, y, z, tiling_data.length);
    op.Process();
}
