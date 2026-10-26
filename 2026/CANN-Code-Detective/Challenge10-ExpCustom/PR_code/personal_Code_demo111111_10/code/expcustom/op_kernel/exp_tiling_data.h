#ifndef _EXP_TILING_DATA_H_
#define _EXP_TILING_DATA_H_

struct ExpTilingData {
    int64_t totalNum = 0;     // 总元素数量
    int64_t blockLength = 0;  // 每个核处理的元素数量
    int64_t tileLength = 0;   // 每次循环搬运的元素数量
    int64_t tileNum = 0;      // 每个核的循环次数
};
#endif