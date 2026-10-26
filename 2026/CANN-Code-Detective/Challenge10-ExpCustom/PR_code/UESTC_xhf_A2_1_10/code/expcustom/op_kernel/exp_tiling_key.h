/*!
 * \file exp_tiling_key.h
 * \brief Tiling 模板参数定义
 */

#ifndef __EXP_TILING_KEY_H__
#define __EXP_TILING_KEY_H__

#include "ascendc/host_api/tiling/template_argument.h"

#define EXP_MODE_HALF 0
#define EXP_MODE_FLOAT 1

ASCENDC_TPL_ARGS_DECL(
    Exp,
    ASCENDC_TPL_UINT_DECL(schMode, 1, ASCENDC_TPL_UI_LIST, EXP_MODE_HALF, EXP_MODE_FLOAT));

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(
    ASCENDC_TPL_UINT_SEL(schMode, ASCENDC_TPL_UI_LIST, EXP_MODE_HALF, EXP_MODE_FLOAT)));

#endif
