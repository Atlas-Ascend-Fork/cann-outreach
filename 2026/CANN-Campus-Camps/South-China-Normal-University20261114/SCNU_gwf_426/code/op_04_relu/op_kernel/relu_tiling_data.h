#ifndef _RELU_TILING_DATA_H_
#define _RELU_TILING_DATA_H_

struct ReluTilingData {
    uint32_t totalLength;
    uint32_t blockLength;
    uint32_t lastBlockLength;
    uint32_t tileNum;
    uint32_t tileLength;
    uint32_t lastTileLength;
};
#endif