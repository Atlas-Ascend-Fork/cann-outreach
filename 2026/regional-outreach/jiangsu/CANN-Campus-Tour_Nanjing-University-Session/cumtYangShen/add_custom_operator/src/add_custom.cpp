/*
 * \file add_custom.cpp
 * \brief add_custom 算子：对两个输入张量执行逐元素加法
 *
 * 算子规格：
 *   - 名称：add_custom
 *   - 功能：z = x + y（逐元素加法）
 *   - 形状：2 维 [N2, N1]
 *   - 数据类型：float16 (half)
 *   - 数据格式：ND（2 维下即行主序连续排布，元素总数 = N2 * N1）
 *
 * 编程范式：矢量编程范式（TQue 队列 + TPipe 统一 UB 内存管理）
 *   Stage1 CopyIn  : GM -> UB（DataCopy）
 *   Stage2 Compute : UB 内向量加法（Add）
 *   Stage3 CopyOut : UB -> GM（DataCopy）
 *
 * 数据切分：
 *   1. 核间：按 GetBlockNum() 均分全部元素（向上取整，末核负责余数）；
 *   2. 核内：按 TILE_LEN（32 字节的整数倍）分块（tile）处理，
 *      末尾不足一个 tile 的部分按实际长度处理，保证任意 [N2, N1] 结果正确。
 */
#include "kernel_operator.h"

namespace {
// 每个 tile 处理的元素个数。half 为 2 字节，256 个元素 = 512 字节，
// 是 32 字节对齐要求的整数倍，满足 MTE 搬运与向量单元的高效执行。
constexpr int32_t TILE_LEN = 256;
}  // namespace

class KernelAddCustom {
public:
    __aicore__ inline KernelAddCustom() {}

    // 初始化：按核切分 GM 缓冲区，申请 UB 队列缓冲
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, int64_t n2, int64_t n1)
    {
        int64_t totalLen = n2 * n1;  // 元素总数（ND 2 维按行主序连续排布）
        int32_t coreNum = AscendC::GetBlockNum();
        int32_t coreId = AscendC::GetBlockIdx();

        perCoreLen = (totalLen + coreNum - 1) / coreNum;  // 每核均分长度（向上取整）
        coreStart = static_cast<int64_t>(coreId) * perCoreLen;
        if (coreStart >= totalLen) {
            coreLen = 0;  // 数据不够分到本核，直接退出（Process 的循环不会执行）
            return;
        }
        coreLen = perCoreLen;
        if (coreStart + coreLen > totalLen) {
            coreLen = totalLen - coreStart;  // 最后一个核可能不足 perCoreLen
        }

        // 各核只操作自己负责的分片
        xGm.SetGlobalBuffer((__gm__ half *)x + coreStart, coreLen);
        yGm.SetGlobalBuffer((__gm__ half *)y + coreStart, coreLen);
        zGm.SetGlobalBuffer((__gm__ half *)z + coreStart, coreLen);

        // 申请 UB 队列缓冲：1 个缓冲，每个 TILE_LEN 个 half
        pipe.InitBuffer(inQueueX, 1, TILE_LEN * sizeof(half));
        pipe.InitBuffer(inQueueY, 1, TILE_LEN * sizeof(half));
        pipe.InitBuffer(outQueueZ, 1, TILE_LEN * sizeof(half));
    }

    // 核心处理：核内按 tile 顺序执行 CopyIn -> Compute -> CopyOut
    __aicore__ inline void Process()
    {
        int64_t pos = 0;
        while (pos < coreLen) {
            int64_t tileLen = (coreLen - pos > TILE_LEN) ? static_cast<int64_t>(TILE_LEN) : (coreLen - pos);
            CopyIn(pos, tileLen);
            Compute(tileLen);
            CopyOut(pos, tileLen);
            pos += tileLen;
        }
    }

private:
    // Stage1：将本 tile 的输入从 GM 搬入 UB，并放入 VECIN 队列
    __aicore__ inline void CopyIn(int64_t offset, int64_t len)
    {
        AscendC::LocalTensor<half> xLocal = inQueueX.AllocTensor<half>();
        AscendC::LocalTensor<half> yLocal = inQueueY.AllocTensor<half>();
        AscendC::DataCopy(xLocal, xGm[offset], len);
        AscendC::DataCopy(yLocal, yGm[offset], len);
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }

    // Stage2：从 VECIN 队列取出输入，在 UB 内执行逐元素加法 z = x + y
    __aicore__ inline void Compute(int64_t len)
    {
        AscendC::LocalTensor<half> xLocal = inQueueX.DeQue<half>();
        AscendC::LocalTensor<half> yLocal = inQueueY.DeQue<half>();
        AscendC::LocalTensor<half> zLocal = outQueueZ.AllocTensor<half>();
        AscendC::Add(zLocal, xLocal, yLocal, len);
        outQueueZ.EnQue<half>(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }

    // Stage3：从 VECOUT 队列取出结果，写回 GM
    __aicore__ inline void CopyOut(int64_t offset, int64_t len)
    {
        AscendC::LocalTensor<half> zLocal = outQueueZ.DeQue<half>();
        AscendC::DataCopy(zGm[offset], zLocal, len);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> inQueueX;   // x 输入队列
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> inQueueY;   // y 输入队列
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> outQueueZ; // z 输出队列
    AscendC::GlobalTensor<half> xGm;
    AscendC::GlobalTensor<half> yGm;
    AscendC::GlobalTensor<half> zGm;

    int64_t coreStart = 0;  // 本核在全局中的起始下标
    int64_t coreLen = 0;    // 本核负责的元素个数
    int64_t perCoreLen = 0; // 每核均分长度
};

// 算子入口：add_custom
// x, y, z：GM 地址（half，行主序连续）；n2, n1：张量形状 [N2, N1]
extern "C" __global__ __aicore__ void add_custom(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t n2, uint32_t n1)
{
    KernelAddCustom op;
    op.Init(x, y, z, static_cast<int64_t>(n2), static_cast<int64_t>(n1));
    op.Process();
}
