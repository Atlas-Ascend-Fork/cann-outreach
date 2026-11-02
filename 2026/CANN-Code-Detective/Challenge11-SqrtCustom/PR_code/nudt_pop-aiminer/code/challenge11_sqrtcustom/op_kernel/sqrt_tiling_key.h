#ifndef SQRT_CUSTOM_TILING_KEY_H
#define SQRT_CUSTOM_TILING_KEY_H

#include "ascendc/host_api/tiling/template_argument.h"

// Dispatch IDs, independent of ge::DataType. ID 1 retains sqrt<1> in the supplied UT.
constexpr int SQRT_FLOAT16 = 0;
constexpr int SQRT_FLOAT32 = 1;

ASCENDC_TPL_ARGS_DECL(Sqrt,
    ASCENDC_TPL_UINT_DECL(DATA_TYPE, 8, ASCENDC_TPL_UI_LIST, 0, 1)
);
ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(DATA_TYPE, ASCENDC_TPL_UI_LIST, 0)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(DATA_TYPE, ASCENDC_TPL_UI_LIST, 1)
    )
);

#endif
