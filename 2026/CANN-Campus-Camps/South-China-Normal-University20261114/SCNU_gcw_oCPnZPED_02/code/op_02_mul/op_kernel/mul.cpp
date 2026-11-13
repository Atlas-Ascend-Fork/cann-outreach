// // Kernel侧核函数实现
// #include "kernel_operator.h"

// #include "mul_tiling.h"
// #include "tiling_key_mul.h"

// template <class DT_X>
// class KernelMul {
// public:
//     __aicore__ inline KernelMul() {}
//     __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t length) {

//     }
//     __aicore__ inline void Process() {

//     }
// private:

// };

// template <typename DT_X>
//  __global__ __aicore__ void mul(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
//     REGISTER_TILING_DEFAULT(MulTilingData);
//     GET_TILING_DATA_WITH_STRUCT(MulTilingData, tiling_data, tiling);
//     KernelMul<DT_X> op;
//     op.Init(x, y, z, tiling_data.length);
//     op.Process();
// }

// // Kernel侧核函数实现
// #include "kernel_operator.h"
// #include "mul_tiling.h"
// #include "tiling_key_mul.h"

// using namespace AscendC;

// constexpr uint32_t TILE_MAX_NUM = 8192;  // 每个tile最多处理的元素数，防止UB装不下

// template <class DT_X>
// class KernelMul {
// public:
//     using T = DT_X;   // DT_X 就是真正的数据类型：float 或 half

//     __aicore__ inline KernelMul() {}

//     __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t length) {
//         uint32_t alignNum = 32 / sizeof(T);                              // 32字节对齐对应的元素个数
//         uint32_t alignLen = (length + alignNum - 1) / alignNum * alignNum;
//         uint32_t coreNum  = GetBlockNum();                               // 核数，和host侧SetBlockDim一致
//         uint32_t perCore  = (alignLen + coreNum - 1) / coreNum;
//         perCore = (perCore + alignNum - 1) / alignNum * alignNum;        // 每核算多少（32字节对齐）
//         if (perCore == 0) perCore = alignNum;

//         uint32_t start = GetBlockIdx() * perCore;                        // 本核从哪个元素开始
//         if (start >= alignLen) { length_ = 0; return; }
//         length_ = (start + perCore > alignLen) ? (alignLen - start) : perCore;

//         xGm_.SetGlobalBuffer((__gm__ T*)x + start, length_);
//         yGm_.SetGlobalBuffer((__gm__ T*)y + start, length_);
//         zGm_.SetGlobalBuffer((__gm__ T*)z + start, length_);

//         uint32_t bufLen = length_ < TILE_MAX_NUM ? length_ : TILE_MAX_NUM;
//         pipe_.InitBuffer(inQueueX_, 1, bufLen * sizeof(T));
//         pipe_.InitBuffer(inQueueY_, 1, bufLen * sizeof(T));
//         pipe_.InitBuffer(outQueueZ_, 1, bufLen * sizeof(T));
//     }

//     __aicore__ inline void Process() {
//         for (uint32_t offset = 0; offset < length_; offset += TILE_MAX_NUM) {
//             uint32_t n = (length_ - offset) < TILE_MAX_NUM ? (length_ - offset) : TILE_MAX_NUM;
//             CopyIn(offset, n);
//             Compute(n);
//             CopyOut(offset, n);
//         }
//     }

// private:
//     __aicore__ inline void CopyIn(uint32_t offset, uint32_t n) {
//         LocalTensor<T> xL = inQueueX_.AllocTensor<T>();
//         LocalTensor<T> yL = inQueueY_.AllocTensor<T>();
//         DataCopy(xL, xGm_[offset], n);
//         DataCopy(yL, yGm_[offset], n);
//         inQueueX_.EnQue(xL);
//         inQueueY_.EnQue(yL);
//     }
//     __aicore__ inline void Compute(uint32_t n) {
//         LocalTensor<T> xL = inQueueX_.DeQue<T>();
//         LocalTensor<T> yL = inQueueY_.DeQue<T>();
//         LocalTensor<T> zL = outQueueZ_.AllocTensor<T>();
//         Mul(zL, xL, yL, n);            // ← 题目要的 z = x * y 就是这一行
//         inQueueX_.FreeTensor(xL);
//         inQueueY_.FreeTensor(yL);
//         outQueueZ_.EnQue(zL);
//     }
//     __aicore__ inline void CopyOut(uint32_t offset, uint32_t n) {
//         LocalTensor<T> zL = outQueueZ_.DeQue<T>();
//         DataCopy(zGm_[offset], zL, n);
//         outQueueZ_.FreeTensor(zL);
//     }

//     TPipe pipe_;
//     TQue<QuePosition::VECIN, 1>  inQueueX_, inQueueY_;
//     TQue<QuePosition::VECOUT, 1> outQueueZ_;
//     GlobalTensor<T> xGm_, yGm_, zGm_;
//     uint32_t length_ = 0;
// };

// template <typename DT_X>
// __global__ __aicore__ void mul(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
//     REGISTER_TILING_DEFAULT(MulTilingData);
//     GET_TILING_DATA_WITH_STRUCT(MulTilingData, tiling_data, tiling);
//     KernelMul<DT_X> op;
//     op.Init(x, y, z, tiling_data.length);
//     op.Process();
// }


// Kernel侧核函数实现
#include "kernel_operator.h"

#include "mul_tiling.h"
#include "tiling_key_mul.h"

using namespace AscendC;

template <class DT_X>
class KernelMul {
    // 单次搬运上限（元素个数）。本算子每核只处理 2048 个元素，一轮就搬完
    static constexpr uint32_t TILE_LEN = 4096;
    // 数据量极小，单缓冲即可：少一次 AllocTensor/EnQue，就少一次流水同步
    static constexpr uint32_t BUFFER_NUM = 1;

public:
    __aicore__ inline KernelMul() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t length) {
        uint32_t blockIdx = GetBlockIdx();
        uint32_t blockNum = GetBlockNum();

        // 每核分到的元素个数（向上取整，并对齐到32字节，保证 DataCopy 地址对齐）
        constexpr uint32_t ALIGN_ELE = 32 / sizeof(DT_X);
        uint32_t perCore = (length + blockNum - 1) / blockNum;
        perCore = (perCore + ALIGN_ELE - 1) / ALIGN_ELE * ALIGN_ELE;

        startIdx = blockIdx * perCore;
        coreLength = (startIdx < length) ? ((length - startIdx) < perCore ? (length - startIdx) : perCore) : 0;

        xGm.SetGlobalBuffer((__gm__ DT_X *)x + startIdx, coreLength);
        yGm.SetGlobalBuffer((__gm__ DT_X *)y + startIdx, coreLength);
        zGm.SetGlobalBuffer((__gm__ DT_X *)z + startIdx, coreLength);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, TILE_LEN * sizeof(DT_X));
        pipe.InitBuffer(inQueueY, BUFFER_NUM, TILE_LEN * sizeof(DT_X));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, TILE_LEN * sizeof(DT_X));
    }

    __aicore__ inline void Process() {
        if (coreLength == 0) {
            return;
        }
        for (uint32_t offset = 0; offset < coreLength; offset += TILE_LEN) {
            uint32_t n = (coreLength - offset < TILE_LEN) ? (coreLength - offset) : TILE_LEN;
            CopyIn(offset, n);
            Compute(n);
            CopyOut(offset, n);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t offset, uint32_t n) {
        LocalTensor<DT_X> xLocal = inQueueX.AllocTensor<DT_X>();
        LocalTensor<DT_X> yLocal = inQueueY.AllocTensor<DT_X>();
        DataCopy(xLocal, xGm[offset], n);
        DataCopy(yLocal, yGm[offset], n);
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }

    __aicore__ inline void Compute(uint32_t n) {
        LocalTensor<DT_X> xLocal = inQueueX.DeQue<DT_X>();
        LocalTensor<DT_X> yLocal = inQueueY.DeQue<DT_X>();
        LocalTensor<DT_X> zLocal = outQueueZ.AllocTensor<DT_X>();
        // 一条向量指令完成整块逐元素乘法
        Mul(zLocal, xLocal, yLocal, n);
        outQueueZ.EnQue(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOut(uint32_t offset, uint32_t n) {
        LocalTensor<DT_X> zLocal = outQueueZ.DeQue<DT_X>();
        DataCopy(zGm[offset], zLocal, n);
        outQueueZ.FreeTensor(zLocal);
    }

    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueY;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueZ;
    GlobalTensor<DT_X> xGm;
    GlobalTensor<DT_X> yGm;
    GlobalTensor<DT_X> zGm;
    uint32_t startIdx = 0;
    uint32_t coreLength = 0;
};

template <typename DT_X>
 __global__ __aicore__ void mul(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(MulTilingData);
    GET_TILING_DATA_WITH_STRUCT(MulTilingData, tiling_data, tiling);
    KernelMul<DT_X> op;
    op.Init(x, y, z, tiling_data.length);
    op.Process();
}
