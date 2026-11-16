 #ifndef RELU_H
  #define RELU_H

  #include "kernel_operator.h"
  #include "kernel_tiling/kernel_tiling.h"
  #include "relu_tiling_data.h"
  #include "relu_tiling_key.h"

  namespace NsRelu {

  using namespace AscendC;

  constexpr int32_t BUFFER_NUM = 2;

  template <typename DT_X>
  class Relu {
  public:
      __aicore__ inline Relu() {}

      __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const ReluTilingData* tilingData)
      {
          uint32_t totalLength = tilingData->totalLength;
          uint32_t blockNum = AscendC::GetBlockNum();
          uint32_t blockIdx = AscendC::GetBlockIdx();
          uint32_t alignElem = 32 / sizeof(DT_X);

          uint32_t blockLength = (totalLength / blockNum / alignElem) * alignElem;
          if (blockLength == 0) {
              if (blockIdx != 0) {
                  this->curLength = 0;
                  return;
              }
              blockLength = totalLength;
          }

          uint32_t start = blockIdx * blockLength;
          if (start >= totalLength) {
              this->curLength = 0;
              return;
          }

          uint32_t end = start + blockLength;
          if (blockIdx == blockNum - 1 || end > totalLength) {
              end = totalLength;
          }
          this->curLength = end - start;

          this->tileLength = tilingData->tileLength < this->curLength
                                 ? tilingData->tileLength
                                 : this->curLength;
          if (this->tileLength < alignElem) {
              this->tileLength = alignElem;
          }
          this->allocLength =
              ((this->tileLength + alignElem - 1) / alignElem) * alignElem;

          inputGMX.SetGlobalBuffer((__gm__ DT_X*)x + start, this->curLength);
          outputGMY.SetGlobalBuffer((__gm__ DT_X*)y + start, this->curLength);

          pipe.InitBuffer(inputQueueX, BUFFER_NUM, this->allocLength * sizeof(DT_X));
          pipe.InitBuffer(outputQueueY, BUFFER_NUM, this->allocLength * sizeof(DT_X));
      }

      __aicore__ inline void Process()
      {
          if (this->curLength == 0) {
              return;
          }

          uint32_t loopCount =
              (this->curLength + this->tileLength - 1) / this->tileLength;
          uint32_t pre = BUFFER_NUM < loopCount ? BUFFER_NUM : loopCount;

          for (uint32_t i = 0; i < pre; ++i) {
              CopyIn(i);
          }

          for (uint32_t i = 0; i < loopCount; ++i) {
              Compute(i);

              uint32_t next = i + BUFFER_NUM;
              if (next < loopCount) {
                  CopyIn(next);
              }

              CopyOut(i);
          }
      }

  private:
      __aicore__ inline uint32_t GetCopyLen(uint32_t progress)
      {
          uint32_t offset = progress * this->tileLength;
          uint32_t remain = this->curLength - offset;
          return this->tileLength < remain ? this->tileLength : remain;
      }

      __aicore__ inline void CopyIn(uint32_t progress)
      {
          uint32_t copyLen = GetCopyLen(progress);
          LocalTensor<DT_X> xLocal = inputQueueX.AllocTensor<DT_X>();
          DataCopy(xLocal, inputGMX[progress * this->tileLength], copyLen);
          inputQueueX.EnQue(xLocal);
      }

      __aicore__ inline void Compute(uint32_t progress)
      {
          uint32_t copyLen = GetCopyLen(progress);
          LocalTensor<DT_X> xLocal = inputQueueX.DeQue<DT_X>();
          LocalTensor<DT_X> yLocal = outputQueueY.AllocTensor<DT_X>();
          AscendC::Maxs(yLocal, xLocal, (DT_X)0.0f, copyLen);
          outputQueueY.EnQue<DT_X>(yLocal);
          inputQueueX.FreeTensor(xLocal);
      }

      __aicore__ inline void CopyOut(uint32_t progress)
      {
          uint32_t copyLen = GetCopyLen(progress);
          LocalTensor<DT_X> yLocal = outputQueueY.DeQue<DT_X>();
          DataCopy(outputGMY[progress * this->tileLength], yLocal, copyLen);
          outputQueueY.FreeTensor(yLocal);
      }

  private:
      TPipe pipe;
      TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;
      TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueY;

      GlobalTensor<DT_X> inputGMX;
      GlobalTensor<DT_X> outputGMY;

      uint32_t curLength;
      uint32_t tileLength;
      uint32_t allocLength;
  };

  } // namespace NsRelu
  #endif // RELU_H