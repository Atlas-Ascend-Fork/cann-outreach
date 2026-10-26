/*!
 * \file exp.h
  * \brief Exp 算子 kernel 类定义
   */

   #ifndef EXP_H
   #define EXP_H

   #include "kernel_operator.h"
   #include "kernel_tiling/kernel_tiling.h"
   #include "exp_tiling_data.h"
   #include "exp_tiling_key.h"

   namespace NsExp {

   using namespace AscendC;

   constexpr int32_t BUFFER_NUM = 2;

   template <typename T>
   class Exp {
   public:
   __aicore__ inline Exp(){};

   __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const ExpTilingData* tilingData);
   __aicore__ inline void Process();

   private:
   __aicore__ inline void CopyIn(int64_t progress, int64_t currentNum);
   __aicore__ inline void CopyOut(int64_t progress, int64_t currentNum);
   __aicore__ inline void Compute(int64_t currentNum);

   private:
   TPipe pipe;
   TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;
   TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueY;

   GlobalTensor<T> inputGMX;
   GlobalTensor<T> outputGMY;

   int64_t totalNum_ = 0;
   int64_t blockLength_ = 0;
   int64_t ubLength_ = 0;

   int64_t coreOffset_ = 0;
   int64_t coreLength_ = 0;
   };

   // kernel 逻辑实现：多核切分 + UB 分块双缓冲流水（CopyIn -> Compute -> CopyOut）
   template <typename T>
   __aicore__ inline void Exp<T>::Init(GM_ADDR x, GM_ADDR y, const ExpTilingData* tilingData)
   {
   this->totalNum_ = tilingData->totalNum;
   this->blockLength_ = tilingData->blockFactor;
   this->ubLength_ = tilingData->ubFactor;
   // 本核负责的数据区间 [coreOffset, coreOffset + coreLength)
   this->coreOffset_ = GetBlockIdx() * this->blockLength_;
   this->coreLength_ = this->totalNum_ - this->coreOffset_;
   if (this->coreLength_ > this->blockLength_) {
   this->coreLength_ = this->blockLength_;
   }
   // 设置 Global Memory 地址
   inputGMX.SetGlobalBuffer((__gm__ T*)x);
   outputGMY.SetGlobalBuffer((__gm__ T*)y);
   // 为输入/输出队列分配 UB 空间（双缓冲）
   pipe.InitBuffer(inputQueueX, BUFFER_NUM, this->ubLength_ * sizeof(T));
   pipe.InitBuffer(outputQueueY, BUFFER_NUM, this->ubLength_ * sizeof(T));
   }

   template <typename T>
   __aicore__ inline void Exp<T>::CopyIn(int64_t progress, int64_t currentNum)
   {
   // 本核数据起始偏移 + 已处理块偏移
   int64_t offset = this->coreOffset_ + progress * this->ubLength_;
   // GM -> UB，数据搬入并入输入队列
   LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();
   DataCopy(xLocal, inputGMX[offset], currentNum);
   inputQueueX.EnQue(xLocal);
   }

   template <typename T>
   __aicore__ inline void Exp<T>::Compute(int64_t currentNum)
   {
   // 出队，执行逐元素自然指数计算 y = e^x，结果入输出队列
   LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
   LocalTensor<T> yLocal = outputQueueY.AllocTensor<T>();
   AscendC::Exp(yLocal, xLocal, currentNum);
   outputQueueY.EnQue<T>(yLocal);
   inputQueueX.FreeTensor(xLocal);
   }

   template <typename T>
   __aicore__ inline void Exp<T>::CopyOut(int64_t progress, int64_t currentNum)
   {
   // 本核数据起始偏移 + 已处理块偏移
   int64_t offset = this->coreOffset_ + progress * this->ubLength_;
   // 出队，UB -> GM，结果搬回全局内存
   LocalTensor<T> yLocal = outputQueueY.DeQue<T>();
   DataCopy(outputGMY[offset], yLocal, currentNum);
   outputQueueY.FreeTensor(yLocal);
   }

   template <typename T>
   __aicore__ inline void Exp<T>::Process()
   {
   if (this->coreLength_ <= 0) {
   return; // 该核无数据（多算的核直接返回）
   }
   // 按 ubLength_ 分块循环处理，最后一个块处理剩余元素（尾块）
   int64_t loopCount = (this->coreLength_ + this->ubLength_ - 1) / this->ubLength_;
   for (int64_t i = 0; i < loopCount; i++) {
   int64_t currentNum = this->coreLength_ - i * this->ubLength_;
   if (currentNum > this->ubLength_) {
   currentNum = this->ubLength_;
   }
   CopyIn(i, currentNum);
   Compute(currentNum);
   CopyOut(i, currentNum);
   }
   }

   } // namespace NsExp
   #endif // EXP_H