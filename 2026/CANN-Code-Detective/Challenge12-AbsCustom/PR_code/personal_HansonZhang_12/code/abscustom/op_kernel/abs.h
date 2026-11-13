/*!
 * \file abs.h
 * \brief Abs 算子 kernel 类定义
 *
 * Abs 为逐元素一元算子： y = |x|
 *   - 直接调用 Ascend C 向量 API AscendC::Abs，float32 / float16 走同一条路径。
 *   - ★ 不需要 fp32 中间缓冲：官方 asc-devkit 源码
 *     impl/basic_api/dav_c220/kernel_operator_vec_unary_impl.h:71-74 对 Abs 做了
 *     static_assert(SupportType<T, half, float>())（Ascend910B / arch 2201 即本题目标芯片），
 *     官方文档「Atlas A2：T 支持 half、float」，官方样例
 *     ascend/samples operator/ascendc/0_introduction/16_unaligned_abs_kernellaunch
 *     的 abs_pad.cpp 也是直接对 LocalTensor<half> 调用 AscendC::Abs。
 *     （对比：第十期 Exp 必须 Cast 到 fp32，是因为 Exp 按 fp32 精度计算；Abs 只清符号位，
 *      vabs 硬件指令对 fp16 原生支持，且 |x| 在 fp16 下精确无损。）
 *
 * 模板的 Process() 骨架保持不变：循环次数 = tileNum × BUFFER_NUM，
 * 因此每个 tile 的元素个数 tileLength 必须满足
 *     tileNum × BUFFER_NUM × tileLength >= blockLength
 * 本实现中 BUFFER_NUM = 1，tileNum 由 blockLength / tileLength 推出。
 *
 * ★ 对齐说明：
 *   host 侧把 blockFactor / ubFactor 都按 32 字节 DMA 粒度取整；
 *   但末尾核的 blockLength = totalNum - offset 是**截断值**，
 *   当 totalNum 不是 32 字节对齐单位的整数倍时（例如 totalNum=1001 的 float32
 *   尾核只有 9 个元素），该截断值不再对齐 —— 这种尾块走 DataCopyPad
 *   （见 CopyIn / CopyOut），其 DMA 长度按字节给出，不再有 32 字节整数倍约束；
 *   UB 容量则一律按对齐后的 ubFactor 申请（见 Init 的 processNum）。
 */

#ifndef ABS_H
#define ABS_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "abs_tiling_data.h"
#include "abs_tiling_key.h"

namespace NsAbs {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 1; // 每队列的 buffer 块数（沿用模板取值；模板 Process() 骨架的循环次数 = tileNum × BUFFER_NUM）

template <typename DT_X>
class Abs {
public:
    __aicore__ inline Abs() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const AbsTilingData* tilingData)
    {
        // 本核负责的数据区间起点（用 int64_t 承接，避免 uint32 溢出）
        int64_t offset = static_cast<int64_t>(AscendC::GetBlockIdx()) * tilingData->blockFactor;

        blockLength_ = tilingData->blockFactor;
        // 防止最后一个核越界：blockFactor 超过剩余数据时截断
        if (offset + blockLength_ > tilingData->totalNum) {
            blockLength_ = tilingData->totalNum - offset;
        }
        if (blockLength_ <= 0) {
            // 本核无数据（例如总元素数少于核数时的空核）
            blockLength_ = 0;
            tileLength_ = 0;
            tileNum = 0;
            processNum = 0;
            return;
        }

        // 单个 tile 的元素个数（host 侧已按 32 字节 DMA 粒度取整）
        tileLength_ = tilingData->ubFactor;
        if (tileLength_ > blockLength_) {
            tileLength_ = blockLength_;
        }
        // 最后一个 tile 可能不满，向上取整得到 tile 数量
        tileNum = (tileLength_ > 0) ? (blockLength_ + tileLength_ - 1) / tileLength_ : 0;

        // ★ UB 缓冲容量用 host 给的 ubFactor（已按 32 字节对齐），
        //   而不是可能被截断的 tileLength_：末尾核的 blockLength 可能是非对齐值
        //   （如 1001 个 float32 的尾核只有 9 个元素），
        //   用它申请会让 InitBuffer 收到非 32 字节整数倍的字节数。
        processNum = tilingData->ubFactor;
        if (processNum < tileLength_) {
            processNum = tileLength_;
        }

        inputGMX.SetGlobalBuffer((__gm__ DT_X*)x + offset, blockLength_);
        outputGMY.SetGlobalBuffer((__gm__ DT_X*)y + offset, blockLength_);

        // ★ UB 申请量必须与 host 侧 abs_tiling.cpp 的 ubFactor 预算严格一致：
        //     float32: in(T) + out(T) = 2 * 4 字节/元素
        //     float16: in(T) + out(T) = 2 * 2 字节/元素
        //   即 bytesPerElement = 2 * sizeof(DT_X)（不需要 fp32 中间缓冲）。
        pipe.InitBuffer(inputQueueX, BUFFER_NUM, processNum * sizeof(DT_X));
        pipe.InitBuffer(outputQueueY, BUFFER_NUM, processNum * sizeof(DT_X));
    }

    __aicore__ inline void Process()
    {
        // 循环次数 = tileNum × BUFFER_NUM（tileLength 已按 BUFFER_NUM 折半）
        int32_t loopCount = this->tileNum * BUFFER_NUM;
        for (int32_t i = 0; i < loopCount; i++) {
            CopyIn(i);
            Compute();
            CopyOut(i);
        }
    }

private:
    // 当前 tile 实际需要处理的元素个数（最后一个 tile 可能不满）
    __aicore__ inline int64_t TileCount(int32_t progress)
    {
        int64_t count = blockLength_ - static_cast<int64_t>(progress) * tileLength_;
        if (count > tileLength_) {
            count = tileLength_;
        }
        if (count < 0) {
            count = 0;
        }
        return count;
    }

    __aicore__ inline void CopyIn(int32_t progress)
    {
        int64_t offset = static_cast<int64_t>(progress) * tileLength_;
        curCount_ = TileCount(progress);

        LocalTensor<DT_X> xLocal = inputQueueX.AllocTensor<DT_X>();
        // DataCopy 要求 32 字节整数倍的 DMA 粒度；末尾 tile 可能不满一个对齐单元，
        // 此时改用 DataCopyPad（DMA 长度按字节给出，不受 32 字节整数倍约束）。
        constexpr int64_t alignNum = 32 / sizeof(DT_X);
        if (curCount_ % alignNum == 0) {
            DataCopy(xLocal, inputGMX[offset], static_cast<uint32_t>(curCount_));
        } else {
            // ★ 按官方《DataCopyPad（GMToUB非对齐数据搬运）》文档的用法：
            //   - blockLen = 有效数据字节数，必须是 sizeof(T) 的整数倍（此处满足）；
            //   - isPad = true 时用 padParams.paddingValue 填充到 32 字节对齐；
            //   - rightPadding = 补齐的元素个数（文档示例 rightPadding = dstCols - srcCols）。
            //   补齐区不参与计算（只算 curCount_ 个元素），填 0 即可（|0| = 0，数值安全）。
            const DataCopyExtParams copyParams{
                1, static_cast<uint32_t>(curCount_ * sizeof(DT_X)), 0, 0, 0};
            const uint8_t padElems =
                static_cast<uint8_t>(alignNum - curCount_ % alignNum);
            const DataCopyPadExtParams<DT_X> padParams{
                true, 0, padElems, static_cast<DT_X>(0)};
            DataCopyPad(xLocal, inputGMX[offset], copyParams, padParams);
        }
        inputQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute()
    {
        // curCount_ 由紧邻其前的 CopyIn 设置
        int32_t count = static_cast<int32_t>(curCount_);

        LocalTensor<DT_X> xLocal = inputQueueX.DeQue<DT_X>();
        LocalTensor<DT_X> yLocal = outputQueueY.AllocTensor<DT_X>();

        // ★ float32 / float16 走完全相同的一条路径：直接对输入 dtype 调 AscendC::Abs。
        //   依据（均为官方一手证据）：
        //   1) asc-devkit 源码 impl/basic_api/dav_c220/kernel_operator_vec_unary_impl.h:71-74
        //      static_assert(SupportType<T, half, float>()) —— arch 2201 即本题目标芯片 Ascend910B；
        //   2) Ascend C API 文档「Abs」：Atlas A2 系列 T 支持 half、float；
        //   3) 官方样例 ascend/samples .../16_unaligned_abs_kernellaunch/AbsPadKernelInvocation/
        //      abs_pad.cpp:62  AscendC::Abs(outputLocal, outputLocal, ...) —— 直接对
        //      LocalTensor<half> 调用，且官方样例本身就是原地调用（dst 与 src 同一块缓冲）。
        //   |x| 只清符号位，fp16 下精确无损，不需要 fp32 中间态。
        //
        // count 参数按官方原型 const int32_t& 传（asc-devkit
        // basic_api/kernel_operator_vec_unary_intf.h:215-218）。
        AscendC::Abs(yLocal, xLocal, count);
        outputQueueY.EnQue(yLocal);

        inputQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress)
    {
        int64_t offset = static_cast<int64_t>(progress) * tileLength_;
        int64_t count = TileCount(progress);

        LocalTensor<DT_X> yLocal = outputQueueY.DeQue<DT_X>();
        // 与 CopyIn 对称：非对齐尾块走 DataCopyPad，避免 DMA 粒度违规。
        constexpr int64_t alignNum = 32 / sizeof(DT_X);
        if (count % alignNum == 0) {
            DataCopy(outputGMY[offset], yLocal, static_cast<uint32_t>(count));
        } else {
            const DataCopyExtParams copyParams{
                1, static_cast<uint32_t>(count * sizeof(DT_X)), 0, 0, 0};
            DataCopyPad(outputGMY[offset], yLocal, copyParams);
        }
        outputQueueY.FreeTensor(yLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueY;
    // ★ 不需要 fp32 中间缓冲：float32 / float16 均直接对输入 dtype 调 AscendC::Abs。

    GlobalTensor<DT_X> inputGMX;
    GlobalTensor<DT_X> outputGMY;

    int64_t blockLength_ = 0; // 本核处理的元素总数
    int64_t tileLength_ = 0;  // 单个 tile 的元素个数
    int64_t tileNum = 0;      // tile 数量（模板 Process() 使用）
    int64_t processNum = 0;   // UB 缓冲容量（元素个数）
    int64_t curCount_ = 0;    // 当前 tile 的实际元素个数
};

} // namespace NsAbs
#endif // ABS_H
