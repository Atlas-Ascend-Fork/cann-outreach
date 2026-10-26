/*!
 * \file exp_tiling_key.h
 * \brief Exp 算子 tiling 模板参数定义
 *
 * 官方单测 tests/ut/op_kernel/test_exp.cpp 中使用 ICPU_SET_TILING_KEY(1) +
 * ICPU_RUN_KF((exp<1>), ...)，tests/ut/op_host/test_exp_tiling.cpp 中
 * DT_FLOAT 期望 expectTilingKey == 1UL。
 * 因此这里用 uint 型模板参数 schMode 表达 dtype 分支：
 *   schMode = 0 -> float16
 *   schMode = 1 -> float32
 * 与 host 侧 exp_tiling.cpp 中 GET_TPL_TILING_KEY(...) 的取值一一对应。
 */

#ifndef __EXP_TILING_KEY_H__
#define __EXP_TILING_KEY_H__

#include "ascendc/host_api/tiling/template_argument.h"

#define EXP_TPL_SCH_MODE_0 0
#define EXP_TPL_SCH_MODE_1 1

ASCENDC_TPL_ARGS_DECL(
    Exp,
    ASCENDC_TPL_UINT_DECL(schMode, 1, ASCENDC_TPL_UI_LIST, EXP_TPL_SCH_MODE_0, EXP_TPL_SCH_MODE_1));

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(
    ASCENDC_TPL_UINT_SEL(schMode, ASCENDC_TPL_UI_LIST, EXP_TPL_SCH_MODE_0, EXP_TPL_SCH_MODE_1)));

#endif
