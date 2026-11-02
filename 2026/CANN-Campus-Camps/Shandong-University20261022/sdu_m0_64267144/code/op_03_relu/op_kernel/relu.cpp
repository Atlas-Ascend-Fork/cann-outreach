#include "kernel_operator.h"
#include "relu.h"
#include "relu_tiling_data.h"
#include "relu_tiling_key.h"

// Host uses 0 for float32 and 1 for float16. Keep the entry argument
// numeric, matching ASCENDC_TPL_UINT_DECL in relu_tiling_key.h.
template <uint32_t DTYPE_ID>
struct ReluDataTypeMap;

template <>
struct ReluDataTypeMap<0> {
    using Type = float;
};

template <>
struct ReluDataTypeMap<1> {
    using Type = half;
};

template <uint32_t DT_X>
__global__ __aicore__ void relu(
    GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(ReluTilingData);
    GET_TILING_DATA_WITH_STRUCT(ReluTilingData, tilingData, tiling);
    using ElementType = typename ReluDataTypeMap<DT_X>::Type;
    NsRelu::Relu<ElementType> op;
    op.Init(x, y, &tilingData);
    op.Process();
}