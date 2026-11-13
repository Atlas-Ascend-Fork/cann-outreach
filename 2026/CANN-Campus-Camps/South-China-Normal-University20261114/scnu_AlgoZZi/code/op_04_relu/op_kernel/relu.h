/*!
 * \file relu.h
  * \brief Relu 算子 kernel 类定义
   */

   #ifndef RELU_H
   #define RELU_H

   #include "kernel_operator.h"
   #include "kernel_tiling/kernel_tiling.h"
   #include "relu_tiling_data.h"
   #include "relu_tiling_key.h"

   namespace NsRelu {

   using namespace AscendC;

   template <typename T>
   class Relu {
   public:
       __aicore__ inline Relu() {}
           __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const ReluTilingData* tilingData);
               __aicore__ inline void Process();

               private:
                   __aicore__ inline void CopyIn(LocalTensor<T> local, int64_t offset, uint32_t count);
                       __aicore__ inline void CopyOut(LocalTensor<T> local, int64_t offset, uint32_t count);

                           TPipe pipe;
                               TBuf<TPosition::VECCALC> buffer;
                                   GlobalTensor<T> inputGMX;
                                       GlobalTensor<T> outputGMY;
                                           int64_t blockLength_ = 0;
                                               int64_t ubLength_ = 0;
                                               };

                                               template <typename T>
                                               __aicore__ inline void Relu<T>::Init(GM_ADDR x, GM_ADDR y, const ReluTilingData* tilingData)
                                               {
                                                   const int64_t offset = GetBlockIdx() * tilingData->blockFactor;
                                                       const int64_t remaining = tilingData->totalNum - offset;
                                                           blockLength_ = remaining < tilingData->blockFactor ? remaining : tilingData->blockFactor;
                                                               if (blockLength_ < 0) {
                                                                       blockLength_ = 0;
                                                                           }
                                                                               ubLength_ = tilingData->ubFactor;
                                                                                   inputGMX.SetGlobalBuffer((__gm__ T*)x + offset, blockLength_);
                                                                                       outputGMY.SetGlobalBuffer((__gm__ T*)y + offset, blockLength_);
                                                                                           const int32_t bufferCount = blockLength_ > ubLength_ ? 2 : 1;
                                                                                               pipe.InitBuffer(buffer, bufferCount * ubLength_ * sizeof(T));
                                                                                               }

                                                                                               template <typename T>
                                                                                               __aicore__ inline void Relu<T>::CopyIn(LocalTensor<T> local, int64_t offset, uint32_t count)
                                                                                               {
                                                                                                   if (count * sizeof(T) % 32 == 0) {
                                                                                                           DataCopy(local, inputGMX[offset], count);
                                                                                                               } else {
                                                                                                                       DataCopyExtParams params = {1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
                                                                                                                               DataCopyPadExtParams<T> padding = {false, 0, 0, 0};
                                                                                                                                       DataCopyPad(local, inputGMX[offset], params, padding);
                                                                                                                                           }
                                                                                                                                           }

                                                                                                                                           template <typename T>
                                                                                                                                           __aicore__ inline void Relu<T>::CopyOut(LocalTensor<T> local, int64_t offset, uint32_t count)
                                                                                                                                           {
                                                                                                                                               if (count * sizeof(T) % 32 == 0) {
                                                                                                                                                       DataCopy(outputGMY[offset], local, count);
                                                                                                                                                           } else {
                                                                                                                                                                   DataCopyExtParams params = {1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
                                                                                                                                                                           DataCopyPad(outputGMY[offset], local, params);
                                                                                                                                                                               }
                                                                                                                                                                               }

                                                                                                                                                                               template <typename T>
                                                                                                                                                                               __aicore__ inline void Relu<T>::Process()
                                                                                                                                                                               {
                                                                                                                                                                                   if (blockLength_ <= 0) {
                                                                                                                                                                                           return;
                                                                                                                                                                                               }
                                                                                                                                                                                                   LocalTensor<T> local = buffer.Get<T>();
                                                                                                                                                                                                       if (blockLength_ <= ubLength_) {
                                                                                                                                                                                                               const uint32_t count = static_cast<uint32_t>(blockLength_);
                                                                                                                                                                                                                       CopyIn(local, 0, count);
                                                                                                                                                                                                                               SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
                                                                                                                                                                                                                                       WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
                                                                                                                                                                                                                                               AscendC::Relu(local, local, count);
                                                                                                                                                                                                                                                       SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
                                                                                                                                                                                                                                                               WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
                                                                                                                                                                                                                                                                       CopyOut(local, 0, count);
                                                                                                                                                                                                                                                                               return;
                                                                                                                                                                                                                                                                                   }

                                                                                                                                                                                                                                                                                       const int64_t loopCount = (blockLength_ + ubLength_ - 1) / ubLength_;
                                                                                                                                                                                                                                                                                           for (int64_t i = 0; i < loopCount; ++i) {
                                                                                                                                                                                                                                                                                                   const auto eventId = static_cast<event_t>(i & 1);
                                                                                                                                                                                                                                                                                                           if (i >= 2) {
                                                                                                                                                                                                                                                                                                                       WaitFlag<HardEvent::MTE3_MTE2>(eventId);
                                                                                                                                                                                                                                                                                                                               }
                                                                                                                                                                                                                                                                                                                                       const int64_t offset = i * ubLength_;
                                                                                                                                                                                                                                                                                                                                               const int64_t remaining = blockLength_ - offset;
                                                                                                                                                                                                                                                                                                                                                       const uint32_t count = static_cast<uint32_t>(remaining < ubLength_ ? remaining : ubLength_);
                                                                                                                                                                                                                                                                                                                                                               LocalTensor<T> tile = local[(i & 1) * ubLength_];
                                                                                                                                                                                                                                                                                                                                                                       CopyIn(tile, offset, count);
                                                                                                                                                                                                                                                                                                                                                                               SetFlag<HardEvent::MTE2_V>(eventId);
                                                                                                                                                                                                                                                                                                                                                                                       WaitFlag<HardEvent::MTE2_V>(eventId);
                                                                                                                                                                                                                                                                                                                                                                                               AscendC::Relu(tile, tile, count);
                                                                                                                                                                                                                                                                                                                                                                                                       SetFlag<HardEvent::V_MTE3>(eventId);
                                                                                                                                                                                                                                                                                                                                                                                                               WaitFlag<HardEvent::V_MTE3>(eventId);
                                                                                                                                                                                                                                                                                                                                                                                                                       CopyOut(tile, offset, count);
                                                                                                                                                                                                                                                                                                                                                                                                                               // 写回完成后才能复用同一块 UB，另一块可继续搬入和计算。
                                                                                                                                                                                                                                                                                                                                                                                                                                       SetFlag<HardEvent::MTE3_MTE2>(eventId);
                                                                                                                                                                                                                                                                                                                                                                                                                                           }
                                                                                                                                                                                                                                                                                                                                                                                                                                               WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
                                                                                                                                                                                                                                                                                                                                                                                                                                                   WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
                                                                                                                                                                                                                                                                                                                                                                                                                                                   }

                                                                                                                                                                                                                                                                                                                                                                                                                                                   } // namespace NsRelu
                                                                                                                                                                                                                                                                                                                                                                                                                                                   #endif // RELU_H
                                                                                                                                                                                                                                                                                                                                                                                                                                                   