/*!
 * \file relu_tiling_data.h
 * \brief tiling data struct
 */

#ifndef _RELU_TILING_DATA_H_
#define _RELU_TILING_DATA_H_

struct ReluTilingData {
    uint32_t totalLength;    // 总元素数量
    uint32_t blockLength;    // 大 shape 每核处理元素数（整除模式）
    uint32_t tileNum;        // 大 shape 循环次数
    uint32_t tileLength;     // 大 shape 每循环元素数
    uint32_t smallCoreDataNum; // 小核每个核处理的元素数
    uint32_t bigCoreDataNum;   // 大核每个核处理的元素数
    uint32_t finalBigTileNum;  // 大核的循环次数
    uint32_t finalSmallTileNum; // 小核的循环次数
    uint32_t tileDataNum;      // 每次循环处理元素数
    uint32_t smallTailDataNum; // 小核最后一块元素数
    uint32_t bigTailDataNum;   // 大核最后一块元素数
    uint32_t tailBlockNum;     // 大核的数量
};
#endif
