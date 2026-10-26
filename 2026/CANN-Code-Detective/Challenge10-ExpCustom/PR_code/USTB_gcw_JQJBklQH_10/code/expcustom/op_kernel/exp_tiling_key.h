/*!
 * \file exp_tiling_key.h
 * \brief Tiling 模板参数定义
 */

#ifndef __EXP_TILING_KEY_H__
#define __EXP_TILING_KEY_H__

#include "ascendc/host_api/tiling/template_argument.h"

/* schMode 取值：不同 dtype 走不同的 kernel 模板实例 */
#define EXP_TPL_SCH_MODE_FLOAT16 0
#define EXP_TPL_SCH_MODE_FLOAT32 1

ASCENDC_TPL_ARGS_DECL(
    Exp,
    ASCENDC_TPL_UINT_DECL(schMode, 1, ASCENDC_TPL_UI_LIST, EXP_TPL_SCH_MODE_FLOAT16, EXP_TPL_SCH_MODE_FLOAT32));

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(
    ASCENDC_TPL_UINT_SEL(schMode, ASCENDC_TPL_UI_LIST, EXP_TPL_SCH_MODE_FLOAT16, EXP_TPL_SCH_MODE_FLOAT32)));

#endif
