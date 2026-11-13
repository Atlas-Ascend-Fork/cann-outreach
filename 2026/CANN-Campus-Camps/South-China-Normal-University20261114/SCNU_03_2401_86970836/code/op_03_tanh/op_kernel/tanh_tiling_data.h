/*!
 * \file tanh_tiling_data.h
 * \brief tiling data struct
 */

#ifndef _TANH_TILING_DATA_H_
#define _TANH_TILING_DATA_H_

struct TanhTilingData {
    uint32_t blockLength; // 每核处理的元素数（host 按最终核数预计算好，核内免除法）
    uint32_t tileNum;     // 单核内循环块数（与缓冲数解耦）
    uint32_t tileLength;  // 每块元素数
};
#endif