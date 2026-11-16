/*!
 * \file relu_tiling_data.h
 * \brief tiling data struct
 */

#ifndef _RELU_TILING_DATA_H_
  #define _RELU_TILING_DATA_H_

  #include <cstdint>

  struct ReluTilingData {
      uint32_t totalLength; // 总元素数量
      uint32_t tileNum;     // 保留字段，未使用
      uint32_t tileLength;  // 单块元素数（host 按 dtype 下发）
  };
  #endif
