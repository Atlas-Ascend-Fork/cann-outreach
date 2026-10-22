# op_host 侧: 编译算子原型注册库(cust_op_proto)与 tiling 库(cust_optiling)
set(CANN_INC_DIR ${ASCEND_CANN_PACKAGE_PATH}/include)
set(CANN_LIB_DIR ${ASCEND_CANN_PACKAGE_PATH}/lib64)
set(OPS_SRC ${CMAKE_CURRENT_SOURCE_DIR}/add0123.cpp)

# ---- 原型注册库 (OP_PROTO_LIB) ----
add_library(cust_op_proto SHARED ${OPS_SRC})
target_include_directories(cust_op_proto PRIVATE
    ${CANN_INC_DIR}
    ${CANN_INC_DIR}/graph
    ${CANN_INC_DIR}/exe_graph
    ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(cust_op_proto PRIVATE OP_PROTO_LIB)
target_compile_options(cust_op_proto PRIVATE -fvisibility=hidden)
target_link_directories(cust_op_proto PRIVATE ${CANN_LIB_DIR})
target_link_libraries(cust_op_proto PRIVATE
    exe_graph
    register
    tiling_api
    -Wl,--whole-archive
    rt2_registry
    -Wl,--no-whole-archive)
set_target_properties(cust_op_proto PROPERTIES OUTPUT_NAME cust_opsproto_rt2.0)

# ---- tiling 库 (OP_TILING_LIB) ----
add_library(cust_optiling SHARED ${OPS_SRC})
target_include_directories(cust_optiling PRIVATE
    ${CANN_INC_DIR}
    ${CANN_INC_DIR}/graph
    ${CANN_INC_DIR}/exe_graph
    ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(cust_optiling PRIVATE OP_TILING_LIB)
target_compile_options(cust_optiling PRIVATE -fvisibility=hidden)
target_link_directories(cust_optiling PRIVATE ${CANN_LIB_DIR})
target_link_libraries(cust_optiling PRIVATE
    exe_graph
    register
    tiling_api
    -Wl,--whole-archive
    rt2_registry
    -Wl,--no-whole-archive)
set_target_properties(cust_optiling PROPERTIES OUTPUT_NAME cust_opmaster_rt2.0)

install(TARGETS cust_op_proto
        LIBRARY DESTINATION op_proto/lib)
install(TARGETS cust_optiling
        LIBRARY DESTINATION op_impl/ai_core/tbe/op_tiling/lib)
