/*!
 * \file exp_tiling_data.h
 * \brief tiling data struct
 *
 * 字段集与 kernel UT（tests/ut/op_kernel/test_exp.cpp）引用的
 * totalNum / blockFactor / ubFactor 保持一致，避免结构体契约单侧修改。
 */

#ifndef _EXP_TILING_DATA_H_
#define _EXP_TILING_DATA_H_

struct ExpTilingData {
    int64_t totalNum = 0;     // 总元素数量
    int64_t blockFactor = 0;  // 多核切分时每核标准处理的元素数（前 N-1 核满块）
    int64_t ubFactor = 0;     // 核内单次 UB 循环处理的元素数
};
#endif
