/*!
 * \file sqrt_tiling_key.h
 * \brief Sqrt 算子 tilingKey 模板参数定义
 */

#ifndef __SQRT_TILING_KEY_H__
#define __SQRT_TILING_KEY_H__

#include "ascendc/host_api/tiling/template_argument.h"

// 计算模式：0 = fp16 升 fp32 计算，1 = fp32 直接计算
#define SQRT_CALC_MODE_FP16_UPCAST 0
#define SQRT_CALC_MODE_FP32 1

ASCENDC_TPL_ARGS_DECL(
    Sqrt,
    ASCENDC_TPL_UINT_DECL(calcMode, 1, ASCENDC_TPL_UI_LIST, SQRT_CALC_MODE_FP16_UPCAST, SQRT_CALC_MODE_FP32));

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(
    ASCENDC_TPL_UINT_SEL(calcMode, ASCENDC_TPL_UI_LIST, SQRT_CALC_MODE_FP16_UPCAST, SQRT_CALC_MODE_FP32)));

#endif // __SQRT_TILING_KEY_H__