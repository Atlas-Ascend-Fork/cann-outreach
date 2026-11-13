/*!
 * \file relu.h
 * \brief Relu 算子 kernel 类定义
 *
 * 实现 y = max(0, x)：
 *   1. 多核并行：按 blockFactor 把展平后的一维数据均分到各 AIV 核；
 *   2. UB 切分：每核再按 ubFactor 分块，块内用单条 Vector 指令完成计算；
 *   3. 手写 ping-pong 流水：一次 InitBuffer 拿到 2*BUFFER_NUM 块 UB（前一半做输入、
 *      后一半做输出），用显式 SetFlag/WaitFlag 组织"预取下一块 + 计算当前块 + 写回当前块"
 *      的三级流水，使 MTE2 / Vector / MTE3 并行。相比 TQue 省掉了每块的 buffer 注册与
 *      队列事件管理（该固定开销 × 块数是大 shape 的主要差距来源）；
 *   4. 对齐快路径：整块 32B 对齐时用 DataCopy，只有尾部残块才用 DataCopyPad。
 */
#ifndef RELU_H
#define RELU_H
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "relu_tiling_data.h"
#include "relu_tiling_key.h"
namespace NsRelu {
using namespace AscendC;
// ping-pong 缓冲块数（进/出各 BUFFER_NUM 块）
constexpr int32_t BUFFER_NUM = 2;
// GM <-> UB 搬运的 32B 对齐粒度
constexpr int64_t GM_ALIGN_BYTES = 32;
// Vector 单次 repeat 处理的字节数
constexpr int32_t BYTES_PER_REPEAT = 256;
// 单条 Vector 指令最大 repeat 次数（repeatTimes 为 uint8_t）
constexpr int32_t MAX_REPEAT_TIMES = 255;
// 显式同步事件（不用 TQue，同步全部自己管）：
//   输入块 b：MTE2_V[b] 搬入完成 -> 可计算；V_MTE2[b] 计算读完 -> 可覆盖
//   输出块 b：V_MTE3[b] 计算写完 -> 可写回；MTE3_V[b] 写回完成 -> 可覆盖
constexpr event_t EV_IN_MTE2_V[BUFFER_NUM] = {EVENT_ID0, EVENT_ID2};
constexpr event_t EV_IN_V_MTE2[BUFFER_NUM] = {EVENT_ID1, EVENT_ID3};
constexpr event_t EV_OUT_V_MTE3[BUFFER_NUM] = {EVENT_ID4, EVENT_ID6};
constexpr event_t EV_OUT_MTE3_V[BUFFER_NUM] = {EVENT_ID5, EVENT_ID7};
template <typename T>
class Relu {
public:
    __aicore__ inline explicit Relu(TPipe* pipe) : pipe_(pipe){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const ReluTilingData* tilingData);
    __aicore__ inline void Process();
private:
    __aicore__ inline int64_t SlotElems() const;
    __aicore__ inline int64_t TileLength(int64_t progress) const;
    __aicore__ inline bool IsAligned(int64_t currentNum) const;
    __aicore__ inline LocalTensor<T> InTensor(int64_t slot) const;
    __aicore__ inline LocalTensor<T> OutTensor(int64_t slot) const;
    __aicore__ inline void CopyIn(int64_t progress, int64_t currentNum);
    __aicore__ inline void Compute(int64_t progress, int64_t currentNum);
    __aicore__ inline void CopyOut(int64_t progress, int64_t currentNum);
private:
    // TPipe 放在 kernel 入口（类外）构造，类内只持指针：利于编译器的标量化优化
    TPipe* pipe_ = nullptr;
    // 只做一次 InitBuffer：2*BUFFER_NUM 块连成一片（0/1 输入 ping-pong，2/3 输出 ping-pong）
    TBuf<TPosition::VECIN> ubBuf_;
    LocalTensor<T> ubLocal_;
    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;
    int64_t blockLength_ = 0;   // 本核负责的元素数
    int64_t ubLength_ = 0;      // 单次 UB 循环处理的元素数
    int64_t tileNum_ = 0;       // 本核的块数（判断是否还需要 set flag）
};
template <typename T>
__aicore__ inline int64_t Relu<T>::SlotElems() const
{
    // 每块按 32B 向上取整，保证每个 slot 起始地址 32B 对齐
    return ((ubLength_ * static_cast<int64_t>(sizeof(T)) + GM_ALIGN_BYTES - 1) / GM_ALIGN_BYTES) *
           GM_ALIGN_BYTES / static_cast<int64_t>(sizeof(T));
}
template <typename T>
__aicore__ inline void Relu<T>::Init(GM_ADDR x, GM_ADDR y, const ReluTilingData* tilingData)
{
    blockLength_ = tilingData->blockFactor;
    ubLength_ = tilingData->ubFactor;
    // 每个核在 Global Memory 上的起始元素偏移
    const int64_t blockIdx = static_cast<int64_t>(GetBlockIdx());
    const int64_t gmOffset = blockIdx * blockLength_;
    // 尾核数据量可能不足 blockFactor，按实际剩余元素数裁剪；越界核直接退出
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
    inputGMX.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x) + gmOffset,
                             static_cast<uint32_t>(blockLength_));
    outputGMY.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y) + gmOffset,
                              static_cast<uint32_t>(blockLength_));
    pipe_->InitBuffer(ubBuf_, static_cast<uint32_t>(SlotElems() * sizeof(T) * BUFFER_NUM * 2));
    ubLocal_ = ubBuf_.Get<T>();
}
template <typename T>
__aicore__ inline LocalTensor<T> Relu<T>::InTensor(int64_t slot) const
{
    return ubLocal_[static_cast<int32_t>(slot * SlotElems())];
}
template <typename T>
__aicore__ inline LocalTensor<T> Relu<T>::OutTensor(int64_t slot) const
{
    return ubLocal_[static_cast<int32_t>((BUFFER_NUM + slot) * SlotElems())];
}
template <typename T>
__aicore__ inline int64_t Relu<T>::TileLength(int64_t progress) const
{
    const int64_t restNum = blockLength_ - progress * ubLength_;
    return (restNum < ubLength_) ? restNum : ubLength_;
}
template <typename T>
__aicore__ inline bool Relu<T>::IsAligned(int64_t currentNum) const
{
    return ((currentNum * static_cast<int64_t>(sizeof(T))) % GM_ALIGN_BYTES) == 0;
}
template <typename T>
__aicore__ inline void Relu<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    const int32_t slot = static_cast<int32_t>(progress % BUFFER_NUM);
    // 同一块 buffer 上，上一轮（progress - BUFFER_NUM）的计算必须读完才能覆盖
    if (progress >= BUFFER_NUM) {
        WaitFlag<HardEvent::V_MTE2>(EV_IN_V_MTE2[slot]);
    }
    LocalTensor<T> xLocal = InTensor(slot);
    const int64_t offset = progress * ubLength_;
    if (IsAligned(currentNum)) {
        // 整块 32B 对齐：走 DataCopy 快路径，省掉 pad 的额外开销
        DataCopy(xLocal, inputGMX[offset], static_cast<uint32_t>(currentNum));
    } else {
        // 尾部残块（元素数非 32B 对齐）：DataCopyPad 自动处理对齐
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(currentNum * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padParams{false, 0, 0, static_cast<T>(0)};
        DataCopyPad(xLocal, inputGMX[offset], copyParams, padParams);
    }
    SetFlag<HardEvent::MTE2_V>(EV_IN_MTE2_V[slot]);
}
template <typename T>
__aicore__ inline void Relu<T>::Compute(int64_t progress, int64_t currentNum)
{
    const int32_t slot = static_cast<int32_t>(progress % BUFFER_NUM);
    WaitFlag<HardEvent::MTE2_V>(EV_IN_MTE2_V[slot]);
    // 同一块输出 buffer 上，上上轮（progress - BUFFER_NUM）的写回必须完成才能覆盖
    if (progress >= BUFFER_NUM) {
        WaitFlag<HardEvent::MTE3_V>(EV_OUT_MTE3_V[slot]);
    }
    LocalTensor<T> xLocal = InTensor(slot);
    LocalTensor<T> yLocal = OutTensor(slot);
    // y = max(x, 0)：单条 Vector 指令最多覆盖 MAX_REPEAT_TIMES 个 repeat，按此上限分段
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
    SetFlag<HardEvent::V_MTE3>(EV_OUT_V_MTE3[slot]);
    // 只有后续还有搬入会用到这块输入 buffer 时才需要通知，省掉收尾的额外 wait
    if (progress + BUFFER_NUM < tileNum_) {
        SetFlag<HardEvent::V_MTE2>(EV_IN_V_MTE2[slot]);
    }
}
template <typename T>
__aicore__ inline void Relu<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    const int32_t slot = static_cast<int32_t>(progress % BUFFER_NUM);
    WaitFlag<HardEvent::V_MTE3>(EV_OUT_V_MTE3[slot]);
    LocalTensor<T> yLocal = OutTensor(slot);
    const int64_t offset = progress * ubLength_;
    if (IsAligned(currentNum)) {
        DataCopy(outputGMY[offset], yLocal, static_cast<uint32_t>(currentNum));
    } else {
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(currentNum * sizeof(T)), 0, 0, 0};
        DataCopyPad(outputGMY[offset], yLocal, copyParams);
    }
    // 只有后续还会写这块输出 buffer 时才需要通知
    if (progress + BUFFER_NUM < tileNum_) {
        SetFlag<HardEvent::MTE3_V>(EV_OUT_MTE3_V[slot]);
    }
}
template <typename T>
__aicore__ inline void Relu<T>::Process()
{
    if (blockLength_ <= 0) {
        return;
    }
    // 预取第 0 块，随后逐块"预取 i+1 / 计算 i / 写回 i"，三级流水重叠
    CopyIn(0, TileLength(0));
    for (int64_t i = 0; i < tileNum_; ++i) {
        if (i + 1 < tileNum_) {
            CopyIn(i + 1, TileLength(i + 1));
        }
        Compute(i, TileLength(i));
        CopyOut(i, TileLength(i));
    }
}
} // namespace NsRelu
#endif // RELU_H