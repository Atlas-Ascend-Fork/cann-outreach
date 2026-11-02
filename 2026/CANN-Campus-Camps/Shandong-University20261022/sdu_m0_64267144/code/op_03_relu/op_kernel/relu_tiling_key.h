#ifndef RELU_TILING_KEY_H
#define RELU_TILING_KEY_H

#include "ascendc/host_api/tiling/template_argument.h"

// Explicit integer selector: 0 = float32, 1 = float16.
ASCENDC_TPL_ARGS_DECL(
    Relu,
    ASCENDC_TPL_UINT_DECL(
        DT_X, ASCENDC_TPL_8_BW, ASCENDC_TPL_UI_LIST, 0, 1));

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(DT_X, ASCENDC_TPL_UI_LIST, 0, 1)));

#endif