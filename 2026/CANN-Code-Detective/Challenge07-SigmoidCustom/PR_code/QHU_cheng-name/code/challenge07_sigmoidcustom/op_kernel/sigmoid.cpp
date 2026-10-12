/*!
 * \file sigmoid.cpp
  * \brief Sigmoid 算子 kernel 入口
   */

   #include "sigmoid.h"

   template <uint32_t tilingKey>
   __global__ __aicore__ void sigmoid(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
   {
       REGISTER_TILING_DEFAULT(SigmoidTilingData);
           GET_TILING_DATA_WITH_STRUCT(SigmoidTilingData, tilingData, tiling);
               if constexpr (tilingKey == 0) {
                       // float32 实例（框架实例序号 0 = C_DT_FLOAT）
                               NsSigmoid::Sigmoid<float> op;
                                       op.Init(x, y, &tilingData);
                                               op.Process();
                                                   } else {
                                                           // float16 实例（框架实例序号 1 = C_DT_FLOAT16）
                                                                   NsSigmoid::Sigmoid<half> op;
                                                                           op.Init(x, y, &tilingData);
                                                                                   op.Process();
                                                                                       }
                                                                                       }
                                                                                       