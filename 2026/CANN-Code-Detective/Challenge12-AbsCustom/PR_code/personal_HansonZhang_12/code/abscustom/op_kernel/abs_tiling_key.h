/*!
 * \file abs_tiling_key.h
 * \brief Abs 算子 tiling 模板参数定义
 *
 * 官方单测 tests/ut/op_kernel/test_abs.cpp 中使用 ICPU_SET_TILING_KEY(1) +
 * ICPU_RUN_KF((abs<1>), ...)，tests/ut/op_host/test_abs_tiling.cpp 中
 * DT_FLOAT 期望 expectTilingKey == 1UL。
 * ★ 因此这里用 uint 型模板参数 schMode 表达 dtype 分支（而不是模板自带的
 *   ASCENDC_TPL_DATATYPE_DECL）：kernel 入口必须写成 template <uint32_t schMode>，
 *   才能接受官方单测里的非类型实参 abs<1>；
 *   模板自带的 DATATYPE 形式无法被字面量 1 实例化。
 *   schMode = 0 -> float16
 *   schMode = 1 -> float32
 * 与 host 侧 abs_tiling.cpp 中 GET_TPL_TILING_KEY(...) 的取值一一对应。
 * ★ ASCENDC_TPL_UINT_DECL 的第二个参数是**最大位宽（bit width）**，不是默认值
 *   （官方《Tiling 模板编程》参数说明：args1 = 最大位宽，模板参数的个数不能超过最大位宽；
 *    官方宏展开为 ParamStruct{#x, ASCENDC_TPL_UINT, bw, {__VA_ARGS__}, "DECL"}）。
 *   这里取 bw = 1：schMode 只有 0 / 1 两个取值，1 个 bit 即可编码，
 *   编码后的 tilingKey 就是 0 / 1，与官方单测的 abs<1>、expectTilingKey == 1UL 一致。
 *   ★ 顺带说明：模板自带的 ASCENDC_TPL_DATATYPE_DECL(DT_X, C_DT_FLOAT, C_DT_FLOAT16)
 *     同样按 8 bit 编码，但列表中**第一个**类型 C_DT_FLOAT 会编码成 0，
 *     与官方单测期望的 float32 -> tilingKey == 1UL 相反 —— 这也是必须改用 UINT 形式的原因之一。
 */

#ifndef __ABS_TILING_KEY_H__
#define __ABS_TILING_KEY_H__

#include "ascendc/host_api/tiling/template_argument.h"

#define ABS_TPL_SCH_MODE_0 0
#define ABS_TPL_SCH_MODE_1 1

ASCENDC_TPL_ARGS_DECL(
    Abs,
    ASCENDC_TPL_UINT_DECL(schMode, 1, ASCENDC_TPL_UI_LIST, ABS_TPL_SCH_MODE_0, ABS_TPL_SCH_MODE_1));

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(
    ASCENDC_TPL_UINT_SEL(schMode, ASCENDC_TPL_UI_LIST, ABS_TPL_SCH_MODE_0, ABS_TPL_SCH_MODE_1)));

#endif
