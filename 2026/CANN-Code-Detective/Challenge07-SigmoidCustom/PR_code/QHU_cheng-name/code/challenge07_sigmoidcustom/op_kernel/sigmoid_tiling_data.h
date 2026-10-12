/*!
 * \file sigmoid_tiling_data.h
  * \brief tiling data struct
   */

   #ifndef _SIGMOID_TILING_DATA_H_
   #define _SIGMOID_TILING_DATA_H_

   struct SigmoidTilingData {
       int64_t totalNum;    // 总元素数量
           int64_t blockFactor; // 每核处理元素数（多核切分后）
               int64_t ubFactor;    // UB 单块元素数
               };
               #endif
               