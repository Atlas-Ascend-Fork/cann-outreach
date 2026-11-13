/*!
 * \file exp_tiling_key.h
 * \brief Tiling 模板参数定义
 *
 * schMode 为 1 bit 位宽的单取值模板参数，取值必须落在 [0, 1] 内，
 * 否则取值超出声明位宽会在打包 tilingKey 时被截断，导致 host 下发的
 * key 与 kernel 侧分支不匹配（FP32 静默不执行）。
 * 取值约定与全仓其余算子一致：0 = 前半段分支，1 = 后半段分支。
 */

#ifndef __EXP_TILING_KEY_H__
#define __EXP_TILING_KEY_H__

#include "ascendc/host_api/tiling/template_argument.h"

#define EXP_TPL_SCH_HALF 0   // float16 输入
#define EXP_TPL_SCH_FLOAT 1  // float32 输入

ASCENDC_TPL_ARGS_DECL(
    Exp,
    ASCENDC_TPL_UINT_DECL(schMode, 1, ASCENDC_TPL_UI_LIST, EXP_TPL_SCH_HALF, EXP_TPL_SCH_FLOAT));

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(
    ASCENDC_TPL_UINT_SEL(schMode, ASCENDC_TPL_UI_LIST, EXP_TPL_SCH_HALF, EXP_TPL_SCH_FLOAT)));

#endif
