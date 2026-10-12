/*!
 * \file tanh_tiling_key.h
 * \brief Tanh tiling key
 */

#ifndef __TANH_TILING_KEY_H__
#define __TANH_TILING_KEY_H__

#include "ascendc/host_api/tiling/template_argument.h"

#define TANH_TPL_SCH_MODE 1

ASCENDC_TPL_ARGS_DECL(
    Tanh,
    ASCENDC_TPL_UINT_DECL(
        schMode,
        ASCENDC_TPL_8_BW,
        ASCENDC_TPL_UI_LIST,
        TANH_TPL_SCH_MODE));

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(
            schMode,
            ASCENDC_TPL_UI_LIST,
            TANH_TPL_SCH_MODE)));

#endif