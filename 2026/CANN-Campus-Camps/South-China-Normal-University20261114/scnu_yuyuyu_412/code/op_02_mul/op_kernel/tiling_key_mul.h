#ifndef TILING_KEY_MUL_H
#define TILING_KEY_MUL_H

#include "ascendc/host_api/tiling/template_argument.h"

#define MUL_TPL_FP32 0  // ge::DT_FLOAT
#define MUL_TPL_FP16 1  // ge::DT_FLOAT16

ASCENDC_TPL_ARGS_DECL(Mul,
    ASCENDC_TPL_DTYPE_DECL(DT_X, MUL_TPL_FP32, MUL_TPL_FP16)
);

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DTYPE_SEL(DT_X, MUL_TPL_FP32)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DTYPE_SEL(DT_X, MUL_TPL_FP16)
    )
);
#endif
