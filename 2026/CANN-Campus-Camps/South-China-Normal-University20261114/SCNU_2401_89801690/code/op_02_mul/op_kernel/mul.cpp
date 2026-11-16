// Kernel侧核函数实现： z = x * y
#include "kernel_operator.h"

#include "mul_tiling.h"
#include "tiling_key_mul.h"

// 每个队列申请的 buffer 块数：单缓冲即可（每核一个 tile，省掉一块 UB 与多余的队列同步）
constexpr int32_t BUFFER_NUM = 1;
// UB 与 GM 之间 DataCopy 的对齐粒度（字节）
constexpr uint32_t ALIGN_BYTES = 32;

template <class DT_X>
class KernelMul {
public:
    __aicore__ inline KernelMul() {}

    /**
     * @brief 初始化：做多核切分、设置 GM 上的数据视图、为队列分配 UB 内存
     * @param x 第一个乘数在 Global Memory 上的首地址
     * @param y 第二个乘数在 Global Memory 上的首地址
     * @param z 结果在 Global Memory 上的首地址
     * @param tiling tiling 参数（总长度 / 每核长度 / 单 tile 长度）
     */
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, const MulTilingData &tiling)
    {
        totalLength = tiling.length;
        blockLength = tiling.blockLength;
        tileLength = tiling.tileLength;

        // 多核切分：第 blockIdx 个核处理 [blockIdx * blockLength, blockIdx * blockLength + blockLength)
        const uint32_t blockIdx = static_cast<uint32_t>(AscendC::GetBlockIdx());
        const uint64_t offset = static_cast<uint64_t>(blockIdx) * static_cast<uint64_t>(blockLength);
        if (blockLength == 0 || offset >= static_cast<uint64_t>(totalLength)) {
            blockLength = 0;  // 本核没有数据，Process 中直接返回
            return;
        }
        // 最后一个核可能分不到完整的 blockLength 个元素
        const uint32_t remain = totalLength - static_cast<uint32_t>(offset);
        if (blockLength > remain) {
            blockLength = remain;
        }
        // tile 长度不能超过单核的数据量
        if (tileLength == 0 || tileLength > blockLength) {
            tileLength = blockLength;
        }

        // 分别设置三个张量在本核上的 GM 视图（起点随 blockIdx 偏移）
        xGm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(x) + offset, blockLength);
        yGm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(y) + offset, blockLength);
        zGm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(z) + offset, blockLength);

        // 为队列分配 UB 内存：每个队列 BUFFER_NUM 块，每块 tileLength 个元素（按 32B 向上取整）
        const uint32_t tileBytes = static_cast<uint32_t>(
            (static_cast<uint64_t>(tileLength) * sizeof(DT_X) + ALIGN_BYTES - 1) / ALIGN_BYTES * ALIGN_BYTES);
        pipe.InitBuffer(inQueueX, BUFFER_NUM, tileBytes);
        pipe.InitBuffer(inQueueY, BUFFER_NUM, tileBytes);
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, tileBytes);
    }

    __aicore__ inline void Process()
    {
        if (blockLength == 0) {
            return;
        }
        // DataCopy 的长度必须是 32B 的整数倍，这里按元素个数换算
        constexpr uint32_t alignElems = ALIGN_BYTES / sizeof(DT_X);
        const uint32_t alignedLength = (blockLength / alignElems) * alignElems;

        // 单核内按 tile 依次处理：搬运 -> 计算 -> 搬出
        for (uint32_t offset = 0; offset < alignedLength; offset += tileLength) {
            uint32_t len = alignedLength - offset;
            if (len > tileLength) {
                len = tileLength;
            }
            CopyIn(offset, len);
            Compute(len);
            CopyOut(offset, len);
        }

        // 兜底：处理不足 32B 的尾部元素（本题 8x2048 的总长度 32B 对齐，不会进入此处）
        for (uint32_t i = alignedLength; i < blockLength; ++i) {
            const float xValue = static_cast<float>(xGm.GetValue(i));
            const float yValue = static_cast<float>(yGm.GetValue(i));
            zGm.SetValue(i, static_cast<DT_X>(xValue * yValue));
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t offset, uint32_t len)
    {
        // 从队列申请 LocalTensor（UB 上的 buffer）
        AscendC::LocalTensor<DT_X> xLocal = inQueueX.AllocTensor<DT_X>();
        AscendC::LocalTensor<DT_X> yLocal = inQueueY.AllocTensor<DT_X>();
        // GM -> UB
        AscendC::DataCopy(xLocal, xGm[offset], len);
        AscendC::DataCopy(yLocal, yGm[offset], len);
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }

    __aicore__ inline void Compute(uint32_t len)
    {
        AscendC::LocalTensor<DT_X> xLocal = inQueueX.DeQue<DT_X>();
        AscendC::LocalTensor<DT_X> yLocal = inQueueY.DeQue<DT_X>();
        AscendC::LocalTensor<DT_X> zLocal = outQueueZ.AllocTensor<DT_X>();
        // 向量乘法：z = x * y（支持 float / half）
        AscendC::Mul(zLocal, xLocal, yLocal, len);
        outQueueZ.EnQue<DT_X>(zLocal);
        // 计算完成后及时释放输入 buffer
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOut(uint32_t offset, uint32_t len)
    {
        AscendC::LocalTensor<DT_X> zLocal = outQueueZ.DeQue<DT_X>();
        // UB -> GM
        AscendC::DataCopy(zGm[offset], zLocal, len);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    AscendC::TPipe pipe;  // UB 内存管理对象
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueY;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueZ;
    AscendC::GlobalTensor<DT_X> xGm;
    AscendC::GlobalTensor<DT_X> yGm;
    AscendC::GlobalTensor<DT_X> zGm;

    uint32_t totalLength = 0;   // 元素总个数
    uint32_t blockLength = 0;   // 本核负责的元素个数
    uint32_t tileLength = 0;    // 单核内每个 tile 的元素个数
};

template <typename DT_X>
 __global__ __aicore__ void mul(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    (void)workspace;  // 本算子不使用 workspace
    REGISTER_TILING_DEFAULT(MulTilingData);
    GET_TILING_DATA_WITH_STRUCT(MulTilingData, tiling_data, tiling);
    KernelMul<DT_X> op;
    op.Init(x, y, z, tiling_data);
    op.Process();
}