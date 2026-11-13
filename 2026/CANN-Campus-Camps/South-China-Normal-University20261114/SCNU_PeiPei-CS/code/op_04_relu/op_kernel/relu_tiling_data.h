/*!
 * \file relu_tiling_data.h
 * \brief tiling data struct
 */

#ifndef _RELU_TILING_DATA_H_
#define _RELU_TILING_DATA_H_

// 与评测平台不可编辑的 op_kernel/relu_tiling_data.h 保持一致（本地模板原先是 totalLength/tileNum，
// 提交到平台会因为字段不存在而编译失败，故此处同步为平台的三字段结构）
struct ReluTilingData {
    uint32_t totalNum;    // 总元素数量
    uint32_t blockFactor; // 每核处理的元素数量
    uint32_t ubFactor;    // 单次 UB 计算的元素数量
};
#endif
