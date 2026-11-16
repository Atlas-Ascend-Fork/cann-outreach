// Kernel侧核函数实现
#include "kernel_operator.h"

#include "mul_tiling.h"
#include "tiling_key_mul.h"

constexpr int32_t BUFFER_NUM = 2;               // 每个队列的 buffer 个数（双缓冲）
constexpr uint32_t ALIGN_BYTES = 32;            // DataCopy 最小搬运单位：32B（一个 block）
constexpr uint32_t MAX_TILE_BYTES = 16 * 1024;  // 单块上限 16KB，控制 UB 占用

template <class DT_X>
class KernelMul {
public:
    __aicore__ inline KernelMul() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t length) {
        this->length = length;
        // 32B 所对应的元素个数：float32 -> 8 个，float16 -> 16 个
        this->alignElems = ALIGN_BYTES / sizeof(DT_X);

        uint32_t blockNum = AscendC::GetBlockNum();  // host 侧 SetBlockDim 启动的核数
        uint32_t blockIdx = AscendC::GetBlockIdx();  // 当前核编号

        // ---- 1. 多核切分：按 32B 对齐切成"块"后均匀分配，前 rem 个核各多拿一块 ----
        // 注意：DataCopy 以 32B 为一个 block 搬运，故要求张量元素数 >= 8(float32) / 16(float16)；
        // 本赛题 shape (8, 2048) = 16384 个元素远大于该下限，满足约束
        uint32_t alignedLen = (length / this->alignElems) * this->alignElems;
        uint32_t totalBlocks = alignedLen / this->alignElems;
        uint32_t base = totalBlocks / blockNum;
        uint32_t rem = totalBlocks % blockNum;
        uint32_t myBlocks = base + (blockIdx < rem ? 1 : 0);
        uint32_t startBlock = blockIdx * base + (blockIdx < rem ? blockIdx : rem);
        this->coreStart = startBlock * this->alignElems;
        this->coreLength = myBlocks * this->alignElems;

        // ---- 2. 单核内块大小：一次尽量搬完，减少 DMA 指令条数（本赛题每核只需 1 块）----
        uint32_t maxTileElems = (MAX_TILE_BYTES / sizeof(DT_X) / this->alignElems) * this->alignElems;
        uint32_t perTile = this->coreLength;
        if (perTile > maxTileElems) {
            perTile = maxTileElems;
        }
        if (perTile < this->alignElems) {
            perTile = this->alignElems;  // 数据很少时按一个 32B 块处理
        }
        this->tileLength = perTile;

        // ---- 3. 全局末尾不足 32B 的余数：由"最后一个有数据的核"重算末尾一个对齐块 ----
        this->tailStart = 0;
        this->tailLength = 0;
        uint32_t rest = length - alignedLen;
        if (rest > 0 && totalBlocks > 0 && this->coreLength > 0 &&
            (this->coreStart + this->coreLength == alignedLen)) {
            this->tailStart = length - this->alignElems;
            this->tailLength = this->alignElems;
        }

        // ---- 4. 绑定输入输出 GM 地址 ----
        xGm.SetGlobalBuffer((__gm__ DT_X *)x, length);
        yGm.SetGlobalBuffer((__gm__ DT_X *)y, length);
        zGm.SetGlobalBuffer((__gm__ DT_X *)z, length);

        // ---- 5. 分配 UB：每个队列 BUFFER_NUM 块，每块 tileLength 个元素 ----
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(DT_X));
        pipe.InitBuffer(inQueueY, BUFFER_NUM, this->tileLength * sizeof(DT_X));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileLength * sizeof(DT_X));
    }

    __aicore__ inline void Process() {
        // 单核内按块循环：搬入 -> 计算 -> 搬出（本赛题每核 1 块，只发 3 条搬运指令）
        if (this->coreLength > 0) {
            uint32_t tileCount = (this->coreLength + this->tileLength - 1) / this->tileLength;
            for (uint32_t i = 0; i < tileCount; i++) {
                uint32_t offset = this->coreStart + i * this->tileLength;
                uint32_t len = this->tileLength;
                if (i == tileCount - 1) {
                    len = this->coreLength - i * this->tileLength;  // 尾块（仍为 32B 对齐）
                }
                CopyIn(offset, len);
                Compute(len);
                CopyOut(offset, len);
            }
        }
        // 全局末尾不足 32B 的余数：重算末尾一个对齐块并把结果写回，保证数据完整
        if (this->tailLength > 0) {
            CopyIn(this->tailStart, this->tailLength);
            Compute(this->tailLength);
            CopyOut(this->tailStart, this->tailLength);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t offset, uint32_t len) {
        AscendC::LocalTensor<DT_X> xLocal = inQueueX.AllocTensor<DT_X>();
        AscendC::LocalTensor<DT_X> yLocal = inQueueY.AllocTensor<DT_X>();
        AscendC::DataCopy(xLocal, xGm[offset], len);  // GM -> UB
        AscendC::DataCopy(yLocal, yGm[offset], len);
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }

    __aicore__ inline void Compute(uint32_t len) {
        AscendC::LocalTensor<DT_X> xLocal = inQueueX.DeQue<DT_X>();
        AscendC::LocalTensor<DT_X> yLocal = inQueueY.DeQue<DT_X>();
        AscendC::LocalTensor<DT_X> zLocal = outQueueZ.AllocTensor<DT_X>();
        AscendC::Mul(zLocal, xLocal, yLocal, len);  // z = x * y
        outQueueZ.EnQue<DT_X>(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOut(uint32_t offset, uint32_t len) {
        AscendC::LocalTensor<DT_X> zLocal = outQueueZ.DeQue<DT_X>();
        AscendC::DataCopy(zGm[offset], zLocal, len);  // UB -> GM
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
    uint32_t length;      // 张量总元素数
    uint32_t alignElems;  // 32B 对应的元素个数
    uint32_t coreStart;   // 本核负责的数据起始下标
    uint32_t coreLength;  // 本核负责的数据元素个数
    uint32_t tileLength;  // 单核内每块处理的元素个数
    uint32_t tailStart;   // 全局末尾余数的处理起点
    uint32_t tailLength;  // 全局末尾余数的处理长度
};

template <typename DT_X>
 __global__ __aicore__ void mul(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(MulTilingData);
    GET_TILING_DATA_WITH_STRUCT(MulTilingData, tiling_data, tiling);
    KernelMul<DT_X> op;
    op.Init(x, y, z, tiling_data.length);
    op.Process();
}
