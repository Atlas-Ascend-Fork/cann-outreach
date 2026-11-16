 // Kernel侧核函数实现
  #include "kernel_operator.h"

  #include "mul_tiling.h"
  #include "tiling_key_mul.h"
  constexpr int32_t BUFFER_NUM = 2; // tensor num for each queue

  template <class DT_X>
  class KernelMul {
  public:
      __aicore__ inline KernelMul() {}
      __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z,
                                  uint32_t totalLength, uint32_t tileLength) {
          uint32_t blockNum = AscendC::GetBlockNum();
          uint32_t blockIdx = AscendC::GetBlockIdx();
          uint32_t alignElem = 32 / sizeof(DT_X);

          // 每核处理长度按 32B 对齐，最后一块补余数
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

          this->tileLength = tileLength < this->curLength ? tileLength : this->curLength;
          if (this->tileLength < alignElem) {
              this->tileLength = alignElem;
          }
          this->allocLength =
              ((this->tileLength + alignElem - 1) / alignElem) * alignElem;

          xGm.SetGlobalBuffer((__gm__ DT_X*)x + start, this->curLength);
          yGm.SetGlobalBuffer((__gm__ DT_X*)y + start, this->curLength);
          zGm.SetGlobalBuffer((__gm__ DT_X*)z + start, this->curLength);
          pipe.InitBuffer(inQueueX, BUFFER_NUM, this->allocLength * sizeof(DT_X));
          pipe.InitBuffer(inQueueY, BUFFER_NUM, this->allocLength * sizeof(DT_X));
          pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->allocLength * sizeof(DT_X));
      }

      __aicore__ inline void Process() {
          if (this->curLength == 0) {
              return;
          }

          uint32_t loopCount =
              (this->curLength + this->tileLength - 1) / this->tileLength;
          uint32_t pre = BUFFER_NUM < loopCount ? BUFFER_NUM : loopCount;

          // 预取前两块，之后 Compute 与下一块搬运重叠
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
      __aicore__ inline uint32_t GetCopyLen(uint32_t progress) {
          uint32_t offset = progress * this->tileLength;
          uint32_t remain = this->curLength - offset;
          return this->tileLength < remain ? this->tileLength : remain;
      }

      __aicore__ inline void CopyIn(uint32_t progress) {
          uint32_t copyLen = GetCopyLen(progress);
          AscendC::LocalTensor<DT_X> xLocal = inQueueX.AllocTensor<DT_X>();
          AscendC::LocalTensor<DT_X> yLocal = inQueueY.AllocTensor<DT_X>();
          AscendC::DataCopy(xLocal, xGm[progress * this->tileLength], copyLen);
          AscendC::DataCopy(yLocal, yGm[progress * this->tileLength], copyLen);
          inQueueX.EnQue(xLocal);
          inQueueY.EnQue(yLocal);
      }

      __aicore__ inline void Compute(uint32_t progress) {
          uint32_t copyLen = GetCopyLen(progress);
          AscendC::LocalTensor<DT_X> xLocal = inQueueX.DeQue<DT_X>();
          AscendC::LocalTensor<DT_X> yLocal = inQueueY.DeQue<DT_X>();
          AscendC::LocalTensor<DT_X> zLocal = outQueueZ.AllocTensor<DT_X>();

          // 核心计算：矢量乘法 z = x * y
          AscendC::Mul(zLocal, xLocal, yLocal, copyLen);

          outQueueZ.EnQue<DT_X>(zLocal);
          inQueueX.FreeTensor(xLocal);
          inQueueY.FreeTensor(yLocal);
      }

      __aicore__ inline void CopyOut(uint32_t progress) {
          uint32_t copyLen = GetCopyLen(progress);
          AscendC::LocalTensor<DT_X> zLocal = outQueueZ.DeQue<DT_X>();
          AscendC::DataCopy(zGm[progress * this->tileLength], zLocal, copyLen);
          outQueueZ.FreeTensor(zLocal);
      }

  private:
      AscendC::TPipe pipe;
      AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX;
      AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueY;
      AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueZ;
      AscendC::GlobalTensor<DT_X> xGm;
      AscendC::GlobalTensor<DT_X> yGm;
      AscendC::GlobalTensor<DT_X> zGm;
      uint32_t curLength;
      uint32_t tileLength;
      uint32_t allocLength;
  };

  template <typename DT_X>
  __global__ __aicore__ void mul(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
      REGISTER_TILING_DEFAULT(MulTilingData);
      GET_TILING_DATA_WITH_STRUCT(MulTilingData, tiling_data, tiling);
      KernelMul<DT_X> op;
      op.Init(x, y, z, tiling_data.totalLength, tiling_data.tileLength);
      op.Process();
  }