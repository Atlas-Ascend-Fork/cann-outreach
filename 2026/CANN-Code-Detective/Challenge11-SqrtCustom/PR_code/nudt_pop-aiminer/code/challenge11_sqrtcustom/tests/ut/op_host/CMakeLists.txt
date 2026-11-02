set(OP_NAME sqrt)
set(UT_EXE ${OP_NAME}_op_host_ut)

if(NOT DEFINED SOC_VERSION)
    set(SOC_VERSION "ascend910b")
endif()
string(TOLOWER "${SOC_VERSION}" SOC_VERSION)

message(STATUS "Building UT for SOC_VERSION: ${SOC_VERSION}")

set(OP_HOST_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../op_host)

set(TILING_SRC ${OP_HOST_DIR}/${OP_NAME}_tiling.cpp)
set(TILING_DATA_H ${CMAKE_CURRENT_SOURCE_DIR}/../../../op_kernel/${OP_NAME}_tiling_data.h)

set(OP_HOST_SRCS
    ${OP_HOST_DIR}/${OP_NAME}_def.cpp
    ${OP_HOST_DIR}/${OP_NAME}_infershape.cpp
    ${TILING_SRC}
)

set(UT_TEST_SRCS
    test_${OP_NAME}_tiling.cpp
)

set(UT_MAIN_SRC ${CMAKE_CURRENT_SOURCE_DIR}/test_op_host_main.cpp)

set(OP_HOST_SO ${OP_NAME}_op_host_ut_lib)

add_library(${OP_HOST_SO} SHARED
    ${OP_HOST_SRCS}
)

target_include_directories(${OP_HOST_SO} PRIVATE
    ${UT_COMMON_INCLUDE_DIRS}
    ${OP_HOST_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../op_kernel
)

target_compile_definitions(${OP_HOST_SO} PRIVATE
    BUILD_SOC_VERSION=${SOC_VERSION}
    _GLIBCXX_USE_CXX11_ABI=0
)

target_link_libraries(${OP_HOST_SO} PRIVATE
    -Wl,--no-as-needed
    -Wl,--whole-archive
        ${CANN_UT_LIB_PATH}/librt2_registry.a
        ${CANN_UT_LIB_PATH}/libtiling_api.a
    -Wl,--no-whole-archive
    ${CANN_UT_LIB_PATH}/libopp_registry.so
    ${CANN_UT_LIB_PATH}/libregister.so
    ${CANN_UT_LIB_PATH}/libmmpa.so
)

add_executable(${UT_EXE}
    ${UT_MAIN_SRC}
    ${UT_TEST_SRCS}
    ${UT_COMMON_SRCS}
)

target_include_directories(${UT_EXE} PRIVATE
    ${UT_COMMON_INCLUDE_DIRS}
    ${OP_HOST_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../op_kernel
)

target_compile_definitions(${UT_EXE} PRIVATE
    BUILD_SOC_VERSION=${SOC_VERSION}
    _GLIBCXX_USE_CXX11_ABI=0
)

target_compile_options(${UT_EXE} PUBLIC
    -fPIE
    -fno-access-control
)

target_link_directories(${UT_EXE} PRIVATE
    ${UT_COMMON_LIB_DIRS}
    ${CMAKE_CURRENT_BINARY_DIR}
)

target_link_libraries(${UT_EXE} PRIVATE
    gtest
    gtest_main
    -Wl,--no-as-needed
    ${CANN_UT_LIB_PATH}/libmetadef.so
    ${CANN_UT_LIB_PATH}/libunified_dlog.so
    ${CANN_UT_LIB_PATH}/libopp_registry.so
    ${CANN_UT_LIB_PATH}/libregister.so
    ${CANN_UT_LIB_PATH}/libgraph.so
    ${CANN_UT_LIB_PATH}/libgraph_base.so
    ${CANN_UT_LIB_PATH}/libplatform.so
    ${CANN_UT_LIB_PATH}/libc_sec.so
    ${CANN_UT_LIB_PATH}/libmmpa.so
    dl
    pthread
)

set_target_properties(${UT_EXE} PROPERTIES
    INSTALL_RPATH "$ORIGIN;${CANN_UT_LIB_PATH}"
    BUILD_WITH_INSTALL_RPATH TRUE
)
