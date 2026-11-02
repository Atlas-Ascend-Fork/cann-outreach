/*!
 * \file exp_tiling_key.h
 * \brief Tiling 模板参数定义
 */

#ifndef __EXP_TILING_KEY_H__
#define __EXP_TILING_KEY_H__

#include "ascendc/host_api/tiling/template_argument.h"

ASCENDC_TPL_ARGS_DECL(
    Exp,
    // 自定义 DTYPE 必须使用可由预处理器展开的数值。
    // 0 = float32，1 = float16；不可在此传入 C_DataType 枚举标识符。
    ASCENDC_TPL_DTYPE_DECL(DT_X, 0, 1));

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(
    ASCENDC_TPL_DTYPE_SEL(DT_X, 0, 1)));

#endif
