/*!
 * \file tanh.h
  * \brief Tanh 算子 kernel 类定义
   */

   #ifndef TANH_H
   #define TANH_H

   #include "kernel_operator.h"
   #include "kernel_tiling/kernel_tiling.h"
   #include "tanh_tiling_data.h"
   #include "tanh_tiling_key.h"

   namespace NsTanh {

   using namespace AscendC;

   constexpr int32_t BUFFER_NUM = 2;

   template <typename T>
   class Tanh {
   public:
       __aicore__ inline Tanh() {}

           __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const TanhTilingData* tilingData)
               {
                       blockLength = tilingData->blockFactor;
                               ubLength = tilingData->ubFactor;
                                       const int64_t blockOffset = blockLength * GetBlockIdx();
                                               inputGm.SetGlobalBuffer((__gm__ T*)x + blockOffset, blockLength);
                                                       outputGm.SetGlobalBuffer((__gm__ T*)y + blockOffset, blockLength);
                                                               pipe.InitBuffer(inputQueue, BUFFER_NUM, ubLength * sizeof(T));
                                                                       pipe.InitBuffer(outputQueue, BUFFER_NUM, ubLength * sizeof(T));
                                                                           }

                                                                               __aicore__ inline void Process()
                                                                                   {
                                                                                           const int64_t loopCount = (blockLength + ubLength - 1) / ubLength;
                                                                                                   for (int64_t i = 0; i < loopCount; ++i) {
                                                                                                               const int64_t remaining = blockLength - i * ubLength;
                                                                                                                           const int64_t currentNum = remaining < ubLength ? remaining : ubLength;
                                                                                                                                       CopyIn(i, currentNum);
                                                                                                                                                   Compute(currentNum);
                                                                                                                                                               CopyOut(i, currentNum);
                                                                                                                                                                       }
                                                                                                                                                                           }

                                                                                                                                                                           private:
                                                                                                                                                                               __aicore__ inline void CopyIn(int64_t progress, int64_t currentNum)
                                                                                                                                                                                   {
                                                                                                                                                                                           LocalTensor<T> xLocal = inputQueue.AllocTensor<T>();
                                                                                                                                                                                                   DataCopy(xLocal, inputGm[progress * ubLength], currentNum);
                                                                                                                                                                                                           inputQueue.EnQue(xLocal);
                                                                                                                                                                                                               }

                                                                                                                                                                                                                   __aicore__ inline void Compute(int64_t currentNum)
                                                                                                                                                                                                                       {
                                                                                                                                                                                                                               LocalTensor<T> xLocal = inputQueue.DeQue<T>();
                                                                                                                                                                                                                                       LocalTensor<T> yLocal = outputQueue.AllocTensor<T>();

                                                                                                                                                                                                                                               Muls(yLocal, xLocal, static_cast<T>(2.0f), currentNum);
                                                                                                                                                                                                                                                       Exp(yLocal, yLocal, currentNum);
                                                                                                                                                                                                                                                               Adds(yLocal, yLocal, static_cast<T>(1.0f), currentNum);
                                                                                                                                                                                                                                                                       Duplicate(xLocal, static_cast<T>(2.0f), currentNum);
                                                                                                                                                                                                                                                                               Div(xLocal, xLocal, yLocal, currentNum);
                                                                                                                                                                                                                                                                                       Muls(xLocal, xLocal, static_cast<T>(-1.0f), currentNum);
                                                                                                                                                                                                                                                                                               Adds(yLocal, xLocal, static_cast<T>(1.0f), currentNum);

                                                                                                                                                                                                                                                                                                       outputQueue.EnQue(yLocal);
                                                                                                                                                                                                                                                                                                               inputQueue.FreeTensor(xLocal);
                                                                                                                                                                                                                                                                                                                   }

                                                                                                                                                                                                                                                                                                                       __aicore__ inline void CopyOut(int64_t progress, int64_t currentNum)
                                                                                                                                                                                                                                                                                                                           {
                                                                                                                                                                                                                                                                                                                                   LocalTensor<T> yLocal = outputQueue.DeQue<T>();
                                                                                                                                                                                                                                                                                                                                           DataCopy(outputGm[progress * ubLength], yLocal, currentNum);
                                                                                                                                                                                                                                                                                                                                                   outputQueue.FreeTensor(yLocal);
                                                                                                                                                                                                                                                                                                                                                       }

                                                                                                                                                                                                                                                                                                                                                       private:
                                                                                                                                                                                                                                                                                                                                                           TPipe pipe;
                                                                                                                                                                                                                                                                                                                                                               TQue<QuePosition::VECIN, BUFFER_NUM> inputQueue;
                                                                                                                                                                                                                                                                                                                                                                   TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueue;
                                                                                                                                                                                                                                                                                                                                                                       GlobalTensor<T> inputGm;
                                                                                                                                                                                                                                                                                                                                                                           GlobalTensor<T> outputGm;
                                                                                                                                                                                                                                                                                                                                                                               int64_t blockLength = 0;
                                                                                                                                                                                                                                                                                                                                                                                   int64_t ubLength = 0;
                                                                                                                                                                                                                                                                                                                                                                                   };

                                                                                                                                                                                                                                                                                                                                                                                   } // namespace NsTanh
                                                                                                                                                                                                                                                                                                                                                                                   #endif // TANH_H

                                                                                                                                                                                                                                                                                                                                                                                   