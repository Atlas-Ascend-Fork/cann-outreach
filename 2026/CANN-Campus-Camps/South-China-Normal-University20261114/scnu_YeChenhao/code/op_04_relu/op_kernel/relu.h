/*!
 * \file relu.h
 * \brief Relu 算子 kernel 类定义（三级流水 ping-pong 版）
 *
 *   1. 一次 InitBuffer 拿 4 块 UB：前 2 块搬入 ping-pong，后 2 块搬出 ping-pong。
 *      输入输出分开，MTE2 / Vector / MTE3 三条流水可以真正同时在跑。
 *   2. 自己管 SetFlag/WaitFlag，不用 TQue，省掉每块的队列注册与事件管理开销。
 *   3. 整块 32B 对齐走 DataCopy 快路径，只有尾部残块用 DataCopyPad。
 *   4. Vector 单条指令 repeat 上限 255，超过就分段。
 */

#ifndef RELU_H
#define RELU_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "relu_tiling_data.h"
#include "relu_tiling_key.h"

namespace NsRelu {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;              // 进/出各 ping-pong 两块
constexpr int64_t GM_ALIGN_BYTES = 32;         // GM<->UB 搬运对齐粒度
constexpr int32_t BYTES_PER_REPEAT = 256;      // 单次 repeat 处理的字节数
constexpr int32_t MAX_REPEAT_TIMES = 255;      // repeatTimes 是 uint8_t

// 事件 id 按 4 类 × 2 块分配，共 8 个
constexpr int32_t EV_IN_MTE2_V = 0;    // 0/1 搬入完成 -> 可计算
constexpr int32_t EV_IN_V_MTE2 = 2;    // 2/3 计算读完   -> 可覆盖
constexpr int32_t EV_OUT_V_MTE3 = 4;   // 4/5 计算写完   -> 可搬出
constexpr int32_t EV_OUT_MTE3_V = 6;   // 6/7 搬出完成   -> 可覆盖

template <typename T>
class Relu {
public:
    __aicore__ inline explicit Relu(TPipe* pipe) : pipe_(pipe) {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const ReluTilingData* tilingData)
    {
        blockLength_ = tilingData->blockFactor;
        ubLength_ = tilingData->ubFactor;

        const int64_t blockIdx = static_cast<int64_t>(AscendC::GetBlockIdx());
        const int64_t gmOffset = blockIdx * blockLength_;

        // 尾核数量可能不足 blockFactor，按实际剩余裁剪；越界核直接退出
        const int64_t remainNum = tilingData->totalNum - gmOffset;
        if (remainNum < blockLength_) {
            blockLength_ = remainNum;
        }
        if (blockLength_ <= 0) {
            blockLength_ = 0;
            return;
        }
        if (ubLength_ <= 0 || ubLength_ > blockLength_) {
            ubLength_ = blockLength_;
        }
        tileNum_ = (blockLength_ + ubLength_ - 1) / ubLength_;

        inputGMX.SetGlobalBuffer((__gm__ T*)x + gmOffset, static_cast<uint32_t>(blockLength_));
        outputGMY.SetGlobalBuffer((__gm__ T*)y + gmOffset, static_cast<uint32_t>(blockLength_));

        pipe_->InitBuffer(ubBuf_, static_cast<uint32_t>(SlotElems() * sizeof(T) * BUFFER_NUM * 2));
        ubLocal_ = ubBuf_.Get<T>();
    }

    __aicore__ inline void Process()
    {
        if (blockLength_ <= 0) {
            return;
        }
        // 预取第 0 块，之后每轮"预取 i+1 / 算 i / 写回 i"，三级流水重叠
        CopyIn(0, TileLength(0));
        for (int64_t i = 0; i < tileNum_; ++i) {
            if (i + 1 < tileNum_) {
                CopyIn(i + 1, TileLength(i + 1));
            }
            Compute(i, TileLength(i));
            CopyOut(i, TileLength(i));
        }
    }

private:
    // 每个 slot 的元素数按 32B 上取整，保证 slot 起始地址对齐
    __aicore__ inline int64_t SlotElems() const
    {
        const int64_t bytes = ubLength_ * static_cast<int64_t>(sizeof(T));
        return ((bytes + GM_ALIGN_BYTES - 1) / GM_ALIGN_BYTES) * GM_ALIGN_BYTES /
               static_cast<int64_t>(sizeof(T));
    }

    __aicore__ inline int64_t TileLength(int64_t progress) const
    {
        const int64_t restNum = blockLength_ - progress * ubLength_;
        return (restNum < ubLength_) ? restNum : ubLength_;
    }

    __aicore__ inline bool IsAligned(int64_t num) const
    {
        return ((num * static_cast<int64_t>(sizeof(T))) % GM_ALIGN_BYTES) == 0;
    }

    __aicore__ inline LocalTensor<T> InTensor(int64_t slot) const
    {
        return ubLocal_[static_cast<int32_t>(slot * SlotElems())];
    }

    __aicore__ inline LocalTensor<T> OutTensor(int64_t slot) const
    {
        return ubLocal_[static_cast<int32_t>((BUFFER_NUM + slot) * SlotElems())];
    }

    __aicore__ inline void CopyIn(int64_t progress, int64_t currentNum)
    {
        const int32_t slot = static_cast<int32_t>(progress % BUFFER_NUM);
        // 同一块输入 buffer 上一轮的计算必须读完才能覆盖
        if (progress >= BUFFER_NUM) {
            WaitFlag<HardEvent::V_MTE2>(static_cast<event_t>(EV_IN_V_MTE2 + slot));
        }
        LocalTensor<T> xLocal = InTensor(slot);
        const int64_t offset = progress * ubLength_;
        if (IsAligned(currentNum)) {
            DataCopy(xLocal, inputGMX[offset], static_cast<uint32_t>(currentNum));
        } else {
            DataCopyExtParams copyParams{1, static_cast<uint32_t>(currentNum * sizeof(T)), 0, 0, 0};
            DataCopyPadExtParams<T> padParams{false, 0, 0, static_cast<T>(0)};
            DataCopyPad(xLocal, inputGMX[offset], copyParams, padParams);
        }
        SetFlag<HardEvent::MTE2_V>(static_cast<event_t>(EV_IN_MTE2_V + slot));
    }

    __aicore__ inline void Compute(int64_t progress, int64_t currentNum)
    {
        const int32_t slot = static_cast<int32_t>(progress % BUFFER_NUM);
        WaitFlag<HardEvent::MTE2_V>(static_cast<event_t>(EV_IN_MTE2_V + slot));
        // 同一块输出 buffer 上一轮的写回必须完成才能覆盖
        if (progress >= BUFFER_NUM) {
            WaitFlag<HardEvent::MTE3_V>(static_cast<event_t>(EV_OUT_MTE3_V + slot));
        }
        LocalTensor<T> xLocal = InTensor(slot);
        LocalTensor<T> yLocal = OutTensor(slot);

        // y = max(x, 0)，单条 Vector 指令的 repeat 上限 255，超了就分段
        constexpr int32_t maxNumPerCall = MAX_REPEAT_TIMES * (BYTES_PER_REPEAT / static_cast<int32_t>(sizeof(T)));
        int64_t doneNum = 0;
        while (doneNum < currentNum) {
            const int64_t restNum = currentNum - doneNum;
            const int32_t callNum =
                (restNum < static_cast<int64_t>(maxNumPerCall)) ? static_cast<int32_t>(restNum) : maxNumPerCall;
            Maxs(yLocal[static_cast<int32_t>(doneNum)], xLocal[static_cast<int32_t>(doneNum)],
                 static_cast<T>(0), callNum);
            doneNum += callNum;
        }

        SetFlag<HardEvent::V_MTE3>(static_cast<event_t>(EV_OUT_V_MTE3 + slot));
        // 后面还会往这块输入 buffer 搬入时才需要放行
        if (progress + BUFFER_NUM < tileNum_) {
            SetFlag<HardEvent::V_MTE2>(static_cast<event_t>(EV_IN_V_MTE2 + slot));
        }
    }

    __aicore__ inline void CopyOut(int64_t progress, int64_t currentNum)
    {
        const int32_t slot = static_cast<int32_t>(progress % BUFFER_NUM);
        WaitFlag<HardEvent::V_MTE3>(static_cast<event_t>(EV_OUT_V_MTE3 + slot));
        LocalTensor<T> yLocal = OutTensor(slot);
        const int64_t offset = progress * ubLength_;
        if (IsAligned(currentNum)) {
            DataCopy(outputGMY[offset], yLocal, static_cast<uint32_t>(currentNum));
        } else {
            DataCopyExtParams copyParams{1, static_cast<uint32_t>(currentNum * sizeof(T)), 0, 0, 0};
            DataCopyPad(outputGMY[offset], yLocal, copyParams);
        }
        // 后面还会往这块输出 buffer 写时才需要放行
        if (progress + BUFFER_NUM < tileNum_) {
            SetFlag<HardEvent::MTE3_V>(static_cast<event_t>(EV_OUT_MTE3_V + slot));
        }
    }

private:
    TPipe* pipe_ = nullptr;
    TBuf<TPosition::VECIN> ubBuf_;
    LocalTensor<T> ubLocal_;

    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;

    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
    int64_t tileNum_ = 0;
};

} // namespace NsRelu
#endif // RELU_H