// Kernel侧核函数实现
#include "kernel_operator.h"

#include "mul_tiling.h"
#include "tiling_key_mul.h"

constexpr uint32_t BUFFER_NUM = 2;   // 每个队列 buffer 块数
constexpr uint32_t QUEUE_DEPTH = 1;  // 队列深度（串行搬运，无乒乓流水）

// ---------------------------------------------------------------------------
// 模板参数（整型）→ 元素类型的映射
// 0 对应 ge::DT_FLOAT，1 对应 ge::DT_FLOAT16，需与 tiling_key_mul.h 的取值保持一致
// ---------------------------------------------------------------------------
template <int T>
struct MulDtypeMap;

template <>
struct MulDtypeMap<MUL_TPL_FP32> { using Type = float; };

template <>
struct MulDtypeMap<MUL_TPL_FP16> { using Type = half; };

template <typename DT_X>
class KernelMul {
public:
    __aicore__ inline KernelMul() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t totalLength, uint32_t tileNum)
    {
        // 每个核负责的数据量（多核切分）
        this->blockLength = totalLength / AscendC::GetBlockNum();
        this->tileNum = tileNum;
        // 每个核内每块的长度（已按 BUFFER_NUM 折半）
        this->tileLength = this->blockLength / tileNum / BUFFER_NUM;

        // 各核 GM 起始地址：第 i 个核往后偏移 i*blockLength
        uint64_t gmOffset = static_cast<uint64_t>(this->blockLength) * AscendC::GetBlockIdx();
        xGm.SetGlobalBuffer((__gm__ DT_X *)x + gmOffset, this->blockLength);
        yGm.SetGlobalBuffer((__gm__ DT_X *)y + gmOffset, this->blockLength);
        zGm.SetGlobalBuffer((__gm__ DT_X *)z + gmOffset, this->blockLength);

        // 队列 UB 内存分配
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(DT_X));
        pipe.InitBuffer(inQueueY, BUFFER_NUM, this->tileLength * sizeof(DT_X));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileLength * sizeof(DT_X));
    }

    __aicore__ inline void Process()
    {
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
        AscendC::LocalTensor<DT_X> yLocal = inQueueY.AllocTensor<DT_X>();
        AscendC::DataCopy(xLocal, xGm[progress * this->tileLength], this->tileLength);
        AscendC::DataCopy(yLocal, yGm[progress * this->tileLength], this->tileLength);
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }

    __aicore__ inline void Compute(int32_t progress)
    {
        AscendC::LocalTensor<DT_X> xLocal = inQueueX.DeQue<DT_X>();
        AscendC::LocalTensor<DT_X> yLocal = inQueueY.DeQue<DT_X>();
        AscendC::LocalTensor<DT_X> zLocal = outQueueZ.AllocTensor<DT_X>();

        // z = x * y
        AscendC::Mul(zLocal, xLocal, yLocal, this->tileLength);

        outQueueZ.EnQue<DT_X>(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<DT_X> zLocal = outQueueZ.DeQue<DT_X>();
        AscendC::DataCopy(zGm[progress * this->tileLength], zLocal, this->tileLength);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, QUEUE_DEPTH> inQueueX, inQueueY;
    AscendC::TQue<AscendC::TPosition::VECOUT, QUEUE_DEPTH> outQueueZ;
    AscendC::GlobalTensor<DT_X> xGm, yGm, zGm;

    uint32_t blockLength;  // 每核处理的元素数
    uint32_t tileNum;      // 单核内分块数
    uint32_t tileLength;   // 单核内每块元素数
};

// ---------------------------------------------------------------------------
// 注意：模板参数类型必须是 int（框架按 TilingKey 传 0/1 这样的整型常量进来）
// ---------------------------------------------------------------------------
template <int DT_X>
__global__ __aicore__ void mul(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(MulTilingData);
    GET_TILING_DATA_WITH_STRUCT(MulTilingData, tiling_data, tiling);
    (void)workspace;  // 本算子不使用 workspace

    using DType = typename MulDtypeMap<DT_X>::Type;  // 整型模板参数 → float / half

    KernelMul<DType> op;
    op.Init(x, y, z, tiling_data.totalLength, tiling_data.tileNum);
    op.Process();
}
