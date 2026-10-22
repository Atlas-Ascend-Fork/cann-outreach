/*!
 * \file add_custom_tiling.h
 * \brief add_custom kernel UT（CPU 仿真）用的 tiling 数据定义
 *
 * 在 CPU（tikicpulib）上直接包含 op_kernel/add_custom.cpp 时，
 * 用 memcpy 从 GM 上的 tiling 字节串还原 AddCustomTilingData。
 */

#ifndef _ADD_CUSTOM_TILING_H_
#define _ADD_CUSTOM_TILING_H_

#include <cstdint>
#include <cstring>
#include "../../../op_kernel/add_custom_tiling_data.h"
#include "tikicpulib.h"
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "graph/c_types.h"
#include "ascendc/host_api/tiling/template_argument.h"

#ifndef __aicore__
#define __aicore__
#endif

#ifndef __gm__
#define __gm__
#endif

#ifndef __ubuf__
#define __ubuf__
#endif

// 若 CANN CPU 头未提供 REGISTER_TILING_DEFAULT，退化为空实现（真实语义仅 NPU 编译需要）
#ifndef REGISTER_TILING_DEFAULT
#define REGISTER_TILING_DEFAULT(T)
#endif

inline void InitTilingData(uint8_t* tiling, AddCustomTilingData* constData)
{
    memcpy(constData, tiling, sizeof(AddCustomTilingData));
}

#define GET_TILING_DATA_WITH_STRUCT(tilingStruct, tilingData, tilingArg) \
    tilingStruct tilingData;                                             \
    InitTilingData(tilingArg, &tilingData)

#define GET_TILING_DATA(tilingData, tilingArg) \
    AddCustomTilingData tilingData;            \
    InitTilingData(tilingArg, &tilingData)

#endif // _ADD_CUSTOM_TILING_H_
