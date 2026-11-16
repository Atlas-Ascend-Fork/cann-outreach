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
 	     __aicore__ inline Tanh(){};
 	 
 	     __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const TanhTilingData* tilingData);
 	     __aicore__ inline void Process();
 	 
 	 private:
 	     __aicore__ inline void CopyIn(int64_t progress, int64_t currentNum);
 	     __aicore__ inline void CopyOut(int64_t progress, int64_t currentNum);
 	     __aicore__ inline void Compute(int64_t currentNum);
 	 
 	 private:
 	     TPipe pipe;
 	     TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;
 	     TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueY;
 	 
 	     // 计算过程的临时缓冲：A 放 e^{2x} 链路，B 放除法结果，Two 存常量 2
 	     TBuf<QuePosition::VECCALC> tmpBufA;
 	     TBuf<QuePosition::VECCALC> tmpBufB;
 	     TBuf<QuePosition::VECCALC> bufTwo;
 	 
 	     GlobalTensor<T> inputGMX;
 	     GlobalTensor<T> outputGMY;
 	 
 	     int64_t totalNum_ = 0;
 	     int64_t blockLength_ = 0;
 	     int64_t ubLength_ = 0;
 	 };
 	 
 	 template <typename T>
 	 __aicore__ inline void Tanh<T>::Init(GM_ADDR x, GM_ADDR y, const TanhTilingData* tilingData)
 	 {
 	     totalNum_ = tilingData->totalNum;
 	     blockLength_ = tilingData->blockFactor;
 	     ubLength_ = tilingData->ubFactor;
 	     if (ubLength_ <= 0) {
 	         ubLength_ = blockLength_ > 0 ? blockLength_ : 1;
 	     }
 	 
 	     // 每个核只负责 blockLength_ 这段，起始地址按核号平移
 	     inputGMX.SetGlobalBuffer(
 	         reinterpret_cast<__gm__ T*>(x) + blockLength_ * static_cast<int64_t>(AscendC::GetBlockIdx()),
 	         static_cast<uint32_t>(blockLength_));
 	     outputGMY.SetGlobalBuffer(
 	         reinterpret_cast<__gm__ T*>(y) + blockLength_ * static_cast<int64_t>(AscendC::GetBlockIdx()),
 	         static_cast<uint32_t>(blockLength_));
 	 
 	     pipe.InitBuffer(inputQueueX, BUFFER_NUM, static_cast<uint32_t>(ubLength_ * static_cast<int64_t>(sizeof(T))));
 	     pipe.InitBuffer(outputQueueY, BUFFER_NUM, static_cast<uint32_t>(ubLength_ * static_cast<int64_t>(sizeof(T))));
 	     pipe.InitBuffer(tmpBufA, static_cast<uint32_t>(ubLength_ * static_cast<int64_t>(sizeof(float))));
 	     pipe.InitBuffer(tmpBufB, static_cast<uint32_t>(ubLength_ * static_cast<int64_t>(sizeof(float))));
 	     pipe.InitBuffer(bufTwo, static_cast<uint32_t>(ubLength_ * static_cast<int64_t>(sizeof(float))));
 	 
 	     // 常量 2 只写一次，后面每次除法直接拿来用
 	     LocalTensor<float> twoLocal = bufTwo.Get<float>();
 	     Duplicate<float>(twoLocal, 2.0f, static_cast<uint32_t>(ubLength_));
 	 }
 	 
 	 template <typename T>
 	 __aicore__ inline void Tanh<T>::CopyIn(int64_t progress, int64_t currentNum)
 	 {
 	     LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();
 	     DataCopy(xLocal, inputGMX[progress * ubLength_], static_cast<uint32_t>(currentNum));
 	     inputQueueX.EnQue(xLocal);
 	 }
 	 
 	 template <typename T>
 	 __aicore__ inline void Tanh<T>::Compute(int64_t currentNum)
 	 {
 	     LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
 	     LocalTensor<T> yLocal = outputQueueY.AllocTensor<T>();
 	     LocalTensor<float> aLocal = tmpBufA.Get<float>();
 	     LocalTensor<float> twoLocal = bufTwo.Get<float>();
 	     uint32_t count = static_cast<uint32_t>(currentNum);
 	 
 	     // tanh(x) = 1 - 2 / (1 + e^{2x})，这条式子在大 |x| 时也不会出现 inf/inf
 	     if constexpr (sizeof(T) == sizeof(float)) {
 	         Muls(aLocal, xLocal, 2.0f, count);      // a = 2x
 	         Exp(aLocal, aLocal, count);             // a = e^{2x}
 	         Adds(aLocal, aLocal, 1.0f, count);      // a = 1 + e^{2x}
 	         Div(yLocal, twoLocal, aLocal, count);   // y = 2 / (1 + e^{2x})
 	         Muls(yLocal, yLocal, -1.0f, count);     // y = -2 / (1 + e^{2x})
 	         Adds(yLocal, yLocal, 1.0f, count);      // y = 1 - 2 / (1 + e^{2x})
 	     } else {
 	         // half 的 e^{2x} 在 |x| 稍大时就顶到 65504，转到 float32 上算再转回来
 	         LocalTensor<float> bLocal = tmpBufB.Get<float>();
 	         Cast(aLocal, xLocal, RoundMode::CAST_NONE, count);
 	         Muls(aLocal, aLocal, 2.0f, count);
 	         Exp(aLocal, aLocal, count);
 	         Adds(aLocal, aLocal, 1.0f, count);
 	         Div(bLocal, twoLocal, aLocal, count);
 	         Muls(bLocal, bLocal, -1.0f, count);
 	         Adds(bLocal, bLocal, 1.0f, count);
 	         Cast(yLocal, bLocal, RoundMode::CAST_RINT, count);
 	     }
 	 
 	     outputQueueY.EnQue<T>(yLocal);
 	     inputQueueX.FreeTensor(xLocal);
 	 }
 	 
 	 template <typename T>
 	 __aicore__ inline void Tanh<T>::CopyOut(int64_t progress, int64_t currentNum)
 	 {
 	     LocalTensor<T> yLocal = outputQueueY.DeQue<T>();
 	     DataCopy(outputGMY[progress * ubLength_], yLocal, static_cast<uint32_t>(currentNum));
 	     outputQueueY.FreeTensor(yLocal);
 	 }
 	 
 	 template <typename T>
 	 __aicore__ inline void Tanh<T>::Process()
 	 {
 	     // 尾核分不到 blockLength_ 时按剩余长度截断
 	     int64_t selfLength = totalNum_ - blockLength_ * static_cast<int64_t>(AscendC::GetBlockIdx());
 	     if (selfLength <= 0) {
 	         return;
 	     }
 	     if (selfLength > blockLength_) {
 	         selfLength = blockLength_;
 	     }
 	 
 	     int64_t loopCount = (selfLength + ubLength_ - 1) / ubLength_;
 	     for (int64_t i = 0; i < loopCount; i++) {
 	         int64_t currentNum = (i == loopCount - 1) ? (selfLength - i * ubLength_) : ubLength_;
 	         CopyIn(i, currentNum);
 	         Compute(currentNum);
 	         CopyOut(i, currentNum);
 	     }
 	 }
 	 
 	 } // namespace NsTanh
 	 #endif // TANH_H