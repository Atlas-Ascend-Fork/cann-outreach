/*!
 * \file exp_tiling_key.h
 * \brief Tiling 模板参数定义
 */

#ifndef __EXP_TILING_KEY_H__
#define __EXP_TILING_KEY_H__

#include "ascendc/host_api/tiling/template_argument.h"

#define EXP_TPL_SCH_MODE_0 0
#define EXP_TPL_SCH_MODE_1 1
#define EXP_TPL_SCH_MODE_2 2
#define EXP_TPL_SCH_MODE_3 3

ASCENDC_TPL_ARGS_DECL(
    Exp,
    ASCENDC_TPL_UINT_DECL(schMode, 2, ASCENDC_TPL_UI_LIST,
        EXP_TPL_SCH_MODE_0, EXP_TPL_SCH_MODE_1, EXP_TPL_SCH_MODE_2, EXP_TPL_SCH_MODE_3));

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(
    ASCENDC_TPL_UINT_SEL(schMode, ASCENDC_TPL_UI_LIST,
        EXP_TPL_SCH_MODE_0, EXP_TPL_SCH_MODE_1, EXP_TPL_SCH_MODE_2, EXP_TPL_SCH_MODE_3)));

#endif
