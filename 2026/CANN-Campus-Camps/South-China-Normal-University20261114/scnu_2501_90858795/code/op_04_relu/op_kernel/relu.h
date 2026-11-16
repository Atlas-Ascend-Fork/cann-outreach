/*!
 * \file relu.h
 * \brief Relu 算子 kernel 类定义

#ifndef RELU_H
#define RELU_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "relu_tiling_data.h"
#include "relu_tiling_key.h"

namespace NsRelu {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

template <typename T>
class Relu {
public:
    __aicore__ inline Relu(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const ReluTilingData* tilingData);
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

    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
};

// TODO: 实现具体的 kernel 逻辑
template <typename T>
__aicore__ inline void Relu<T>::Init(GM_ADDR x, GM_ADDR y, const ReluTilingData* tilingData)
{
    // TODO: 实现 Init 逻辑
}

template <typename T>
__aicore__ inline void Relu<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    // TODO: 实现 CopyIn 逻辑
}

template <typename T>
__aicore__ inline void Relu<T>::Compute(int64_t currentNum)
{
    // TODO: 实现 Compute 逻辑
}

template <typename T>
__aicore__ inline void Relu<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    // TODO: 实现 CopyOut 逻辑
}

template <typename T>
__aicore__ inline void Relu<T>::Process()
{
    // TODO: 实现 Process 逻辑
}

} // namespace NsRelu
#endif // RELU_H
*/
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
                    blockLength = static_cast<uint32_t>(tilingData->blockFactor);
                            tileLength = static_cast<uint32_t>(tilingData->ubFactor);

                                    // 每个核从自己的分段起点读写。
                                            uint32_t offset = blockLength * AscendC::GetBlockIdx();
                                                    inputGMX.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X*>(x) + offset, blockLength);
                                                            outputGMY.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X*>(y) + offset, blockLength);

                                                                    pipe.InitBuffer(inputQueueX, BUFFER_NUM, tileLength * sizeof(DT_X));
                                                                            pipe.InitBuffer(outputQueueY, BUFFER_NUM, tileLength * sizeof(DT_X));
                                                                                }

                                                                                    __aicore__ inline void Process()
                                                                                        {
                                                                                                uint32_t loopCount = blockLength / tileLength;
                                                                                                        for (uint32_t i = 0; i < loopCount; ++i) {
                                                                                                                    CopyIn(i);
                                                                                                                                Compute();
                                                                                                                                            CopyOut(i);
                                                                                                                                                    }
                                                                                                                                                        }

                                                                                                                                                        private:
                                                                                                                                                            __aicore__ inline void CopyIn(uint32_t progress)
                                                                                                                                                                {
                                                                                                                                                                        LocalTensor<DT_X> xLocal = inputQueueX.AllocTensor<DT_X>();
                                                                                                                                                                                DataCopy(xLocal, inputGMX[progress * tileLength], tileLength);
                                                                                                                                                                                        inputQueueX.EnQue(xLocal);
                                                                                                                                                                                            }

                                                                                                                                                                                                __aicore__ inline void Compute()
                                                                                                                                                                                                    {
                                                                                                                                                                                                            LocalTensor<DT_X> xLocal = inputQueueX.DeQue<DT_X>();
                                                                                                                                                                                                                    LocalTensor<DT_X> yLocal = outputQueueY.AllocTensor<DT_X>();

                                                                                                                                                                                                                            AscendC::Relu(yLocal, xLocal, tileLength);

                                                                                                                                                                                                                                    outputQueueY.EnQue(yLocal);
                                                                                                                                                                                                                                            inputQueueX.FreeTensor(xLocal);
                                                                                                                                                                                                                                                }

                                                                                                                                                                                                                                                    __aicore__ inline void CopyOut(uint32_t progress)
                                                                                                                                                                                                                                                        {
                                                                                                                                                                                                                                                                LocalTensor<DT_X> yLocal = outputQueueY.DeQue<DT_X>();
                                                                                                                                                                                                                                                                        DataCopy(outputGMY[progress * tileLength], yLocal, tileLength);
                                                                                                                                                                                                                                                                                outputQueueY.FreeTensor(yLocal);
                                                                                                                                                                                                                                                                                    }

                                                                                                                                                                                                                                                                                    private:
                                                                                                                                                                                                                                                                                        TPipe pipe;
                                                                                                                                                                                                                                                                                            TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;
                                                                                                                                                                                                                                                                                                TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueY;

                                                                                                                                                                                                                                                                                                    GlobalTensor<DT_X> inputGMX;
                                                                                                                                                                                                                                                                                                        GlobalTensor<DT_X> outputGMY;
                                                                                                                                                                                                                                                                                                            uint32_t blockLength;
                                                                                                                                                                                                                                                                                                                uint32_t tileLength;
                                                                                                                                                                                                                                                                                                                };

                                                                                                                                                                                                                                                                                                                } // namespace NsRelu

                                                                                                                                                                                                                                                                                                                #endif // RELU_H