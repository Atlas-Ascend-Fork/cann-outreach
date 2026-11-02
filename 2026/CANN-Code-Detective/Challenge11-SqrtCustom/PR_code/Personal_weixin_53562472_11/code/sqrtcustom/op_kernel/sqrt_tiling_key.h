/*!
 * \file sqrt_tiling_key.h
 * \brief Tiling 模板参数定义
 */

#ifndef __SQRT_TILING_KEY_H__
#define __SQRT_TILING_KEY_H__

#include "ascendc/host_api/tiling/template_argument.h"

// schMode 取值：0 -> float16 实例，1 -> float32 实例
#define SQRT_TPL_SCH_MODE_0 0
#define SQRT_TPL_SCH_MODE_1 1

ASCENDC_TPL_ARGS_DECL(
    Sqrt,
    ASCENDC_TPL_UINT_DECL(schMode, 1, ASCENDC_TPL_UI_LIST, SQRT_TPL_SCH_MODE_0, SQRT_TPL_SCH_MODE_1));

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(
    ASCENDC_TPL_UINT_SEL(schMode, ASCENDC_TPL_UI_LIST, SQRT_TPL_SCH_MODE_0, SQRT_TPL_SCH_MODE_1)));

#endif
