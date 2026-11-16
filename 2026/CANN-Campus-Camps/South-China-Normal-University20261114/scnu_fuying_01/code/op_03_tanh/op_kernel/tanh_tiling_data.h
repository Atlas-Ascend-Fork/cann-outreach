/*!
 * \file tanh_tiling_data.h
 * \brief tiling data struct
 */

 #ifndef _TANH_TILING_DATA_H_
  #define _TANH_TILING_DATA_H_

  #include <cstdint>

  struct TanhTilingData {
      uint32_t totalLength;
      uint32_t tileNum;
      uint32_t tileLength;
  };
  #endif