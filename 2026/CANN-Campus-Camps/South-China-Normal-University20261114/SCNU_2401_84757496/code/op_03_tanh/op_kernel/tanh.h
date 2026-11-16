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

  template <typename DT_X>
  class Tanh {
  public:
      __aicore__ inline Tanh() {}

      __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const TanhTilingData* tilingData)
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
          pipe.InitBuffer(tmpBuf0, this->allocLength * sizeof(DT_X));
          pipe.InitBuffer(tmpBuf1, this->allocLength * sizeof(DT_X));
          pipe.InitBuffer(tmpBuf2, this->allocLength * sizeof(DT_X));
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
          LocalTensor<DT_X> tmp0 = tmpBuf0.Get<DT_X>();
          LocalTensor<DT_X> tmp1 = tmpBuf1.Get<DT_X>();
          LocalTensor<DT_X> tmp2 = tmpBuf2.Get<DT_X>();

          // 仅 float16 需要 clamp，防止 exp(±10) 以上溢出
          if (sizeof(DT_X) == 2) {
              AscendC::Mins(xLocal, xLocal, (DT_X)(10.0f), copyLen);
              AscendC::Maxs(xLocal, xLocal, (DT_X)(-10.0f), copyLen);
          }

          AscendC::Exp(tmp0, xLocal, copyLen);                    // a = exp(x)
          AscendC::Muls(xLocal, xLocal, (DT_X)(-1.0f), copyLen);  // x = -x
          AscendC::Exp(tmp1, xLocal, copyLen);                    // b = exp(-x)
          AscendC::Add(xLocal, tmp0, tmp1, copyLen);              // d = a + b
          AscendC::Sub(tmp2, tmp0, tmp1, copyLen);                // n = a - b
          AscendC::Div(yLocal, tmp2, xLocal, copyLen);            // y = n / d

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

      TBuf<TPosition::VECCALC> tmpBuf0;
      TBuf<TPosition::VECCALC> tmpBuf1;
      TBuf<TPosition::VECCALC> tmpBuf2;

      GlobalTensor<DT_X> inputGMX;
      GlobalTensor<DT_X> outputGMY;

      uint32_t curLength;
      uint32_t tileLength;
      uint32_t allocLength;
  };

  } // namespace NsTanh
  #endif // TANH_H