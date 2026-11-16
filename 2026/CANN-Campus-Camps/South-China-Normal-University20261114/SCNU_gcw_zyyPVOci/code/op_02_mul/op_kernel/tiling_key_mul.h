#ifndef TILING_KEY_MUL_H
#define TILING_KEY_MUL_H

#include "ascendc/host_api/tiling/template_argument.h"

// 声明模板参数：只按输入数据类型 D_T_X 分两支
ASCENDC_TPL_ARGS_DECL(
    MulTemplate,
    ASCENDC_TPL_DATATYPE_DECL(D_T_X, C_DT_FLOAT16, C_DT_FLOAT)
);

// 声明合法组合：x/y/z 同类型，要么全 float16，要么全 float32
ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(D_T_X, C_DT_FLOAT16)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(D_T_X, C_DT_FLOAT)
    )
);

#endif  // TILING_KEY_MUL_H