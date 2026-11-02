/*!
 * \file sqrt_tiling_key.h
 * \brief Sqrt 算子 tilingKey 模板参数定义
 */

#ifndef __SQRT_TILING_KEY_H__
#define __SQRT_TILING_KEY_H__

#include "ascendc/host_api/tiling/template_argument.h"

// 计算模式：fp16 与 fp32 各编译一份核函数实例，由 tilingKey 在运行期选中
#define SQRT_SCH_MODE_FP16 0
#define SQRT_SCH_MODE_FP32 1

ASCENDC_TPL_ARGS_DECL(
    Sqrt,
    ASCENDC_TPL_UINT_DECL(schMode, 1, ASCENDC_TPL_UI_LIST, SQRT_SCH_MODE_FP16, SQRT_SCH_MODE_FP32));

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(
    ASCENDC_TPL_UINT_SEL(schMode, ASCENDC_TPL_UI_LIST, SQRT_SCH_MODE_FP16, SQRT_SCH_MODE_FP32)));

#endif // __SQRT_TILING_KEY_H__