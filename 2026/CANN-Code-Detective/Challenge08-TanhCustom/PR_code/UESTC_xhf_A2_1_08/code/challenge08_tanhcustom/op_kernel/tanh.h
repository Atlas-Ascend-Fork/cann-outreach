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
       __aicore__ inline Tanh() {};
           __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const TanhTilingData* tilingData);
               __aicore__ inline void Process();
               private:
                   __aicore__ inline void CopyIn(int64_t progress, int64_t currentNum);
                       __aicore__ inline void CopyOut(int64_t progress, int64_t currentNum);
                           __aicore__ inline void Compute(int64_t currentNum);
                               TPipe pipe;
                                   TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;
                                       TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueY;
                                           GlobalTensor<T> inputGMX;
                                               GlobalTensor<T> outputGMY;
                                                   int64_t blockLength_ = 0;
                                                       int64_t ubLength_ = 0;
                                                       };

                                                       template <typename T>
                                                       __aicore__ inline void Tanh<T>::Init(GM_ADDR x, GM_ADDR y, const TanhTilingData* tilingData)
                                                       {
                                                           blockLength_ = tilingData->blockFactor;
                                                               ubLength_ = tilingData->ubFactor;
                                                                   const int64_t blockOffset = blockLength_ * AscendC::GetBlockIdx();
                                                                       inputGMX.SetGlobalBuffer((__gm__ T*)x + blockOffset, blockLength_);
                                                                           outputGMY.SetGlobalBuffer((__gm__ T*)y + blockOffset, blockLength_);
                                                                               pipe.InitBuffer(inputQueueX, BUFFER_NUM, ubLength_ * sizeof(T));
                                                                                   pipe.InitBuffer(outputQueueY, BUFFER_NUM, ubLength_ * sizeof(T));
                                                                                   }

                                                                                   template <typename T>
                                                                                   __aicore__ inline void Tanh<T>::CopyIn(int64_t progress, int64_t currentNum)
                                                                                   {
                                                                                       LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();
                                                                                           DataCopy(xLocal, inputGMX[progress * ubLength_], currentNum);
                                                                                               inputQueueX.EnQue(xLocal);
                                                                                               }

                                                                                               template <typename T>
                                                                                               __aicore__ inline void Tanh<T>::Compute(int64_t currentNum)
                                                                                               {
                                                                                                   LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
                                                                                                       LocalTensor<T> yLocal = outputQueueY.AllocTensor<T>();
                                                                                                           AscendC::Muls(yLocal, xLocal, static_cast<T>(2.0f), currentNum);
                                                                                                               AscendC::Exp(yLocal, yLocal, currentNum);
                                                                                                                   AscendC::Adds(yLocal, yLocal, static_cast<T>(1.0f), currentNum);
                                                                                                                       AscendC::Duplicate(xLocal, static_cast<T>(2.0f), currentNum);
                                                                                                                           AscendC::Div(xLocal, xLocal, yLocal, currentNum);
                                                                                                                               AscendC::Muls(xLocal, xLocal, static_cast<T>(-1.0f), currentNum);
                                                                                                                                   AscendC::Adds(yLocal, xLocal, static_cast<T>(1.0f), currentNum);
                                                                                                                                       outputQueueY.EnQue(yLocal);
                                                                                                                                           inputQueueX.FreeTensor(xLocal);
                                                                                                                                           }

                                                                                                                                           template <typename T>
                                                                                                                                           __aicore__ inline void Tanh<T>::CopyOut(int64_t progress, int64_t currentNum)
                                                                                                                                           {
                                                                                                                                               LocalTensor<T> yLocal = outputQueueY.DeQue<T>();
                                                                                                                                                   DataCopy(outputGMY[progress * ubLength_], yLocal, currentNum);
                                                                                                                                                       outputQueueY.FreeTensor(yLocal);
                                                                                                                                                       }

                                                                                                                                                       template <typename T>
                                                                                                                                                       __aicore__ inline void Tanh<T>::Process()
                                                                                                                                                       {
                                                                                                                                                           const int64_t loopCount = (blockLength_ + ubLength_ - 1) / ubLength_;
                                                                                                                                                               for (int64_t i = 0; i < loopCount; ++i) {
                                                                                                                                                                       const int64_t remaining = blockLength_ - i * ubLength_;
                                                                                                                                                                               const int64_t currentNum = remaining < ubLength_ ? remaining : ubLength_;
                                                                                                                                                                                       CopyIn(i, currentNum);
                                                                                                                                                                                               Compute(currentNum);
                                                                                                                                                                                                       CopyOut(i, currentNum);
                                                                                                                                                                                                           }
                                                                                                                                                                                                           }
                                                                                                                                                                                                           } // namespace NsTanh
                                                                                                                                                                                                           #endif // TANH_H
                                                                                                                                                                                                           