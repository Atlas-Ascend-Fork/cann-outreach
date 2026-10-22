// Host 侧调用示例 (acl 接口, 供参考)
// 前提: self 已按 4 维 ND 布局 [N4, N3, N2, N1] 连续存放, dtype=float16
//
// #include "acl/acl.h"
//
// struct TilingInplaceRsqrt { uint32_t totalLength; uint32_t tileNum; };
//
// void run_inplace_rsqrt(void *selfDev, uint64_t numel, aclrtStream stream)
// {
//     // 1. 准备 tiling (host 侧)
//     TilingInplaceRsqrt tiling;
//     tiling.totalLength = static_cast<uint32_t>(numel);           // N4*N3*N2*N1
//     tiling.tileNum = (tiling.totalLength + 8191) / 8192;         // 与设备侧 TILE_LENGTH 一致
//
//     void *tilingDev = nullptr;
//     aclrtMalloc(&tilingDev, sizeof(tiling), ACL_MEM_MALLOC_NORMAL_ONLY);
//     aclrtMemcpy(tilingDev, sizeof(tiling), &tiling, sizeof(tiling), ACL_MEMCPY_HOST_TO_DEVICE);
//
//     // 2. 启动 kernel (单核版本; 多核可按 block 分段后改 kernel)
//     inplace_rsqrt<<<1, nullptr, stream>>>(static_cast<uint8_t *>(selfDev),
//                                           static_cast<uint8_t *>(tilingDev));
//     aclrtSynchronizeStream(stream);
//     aclrtFree(tilingDev);
// }
//
// 说明:
// - 原位: kernel 直接把 1/sqrt(x) 写回 self 的 GlobalMemory, 无额外输出
// - 4 维 shape 无需展开处理, 连续内存按一维 numel 遍历即可
// - 输入元素个数应为偶数(与 2 字节 fp16 对齐), 尾块由 ValidCount 处理
