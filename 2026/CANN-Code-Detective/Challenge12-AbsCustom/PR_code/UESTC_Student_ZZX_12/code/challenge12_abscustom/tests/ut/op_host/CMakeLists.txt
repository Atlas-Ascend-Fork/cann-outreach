set(OP_NAME abs)
set(UT_EXE ${OP_NAME}_op_host_ut)

if(DEFINED ENV{ASCEND_HOME_PATH} AND EXISTS "$ENV{ASCEND_HOME_PATH}")
    set(CANN_HOME "$ENV{ASCEND_HOME_PATH}")
elseif(EXISTS "/usr/local/Ascend/cann-8.5.0")
    set(CANN_HOME "/usr/local/Ascend/cann-8.5.0")
else()
    message(FATAL_ERROR "Cannot find CANN installation")
endif()

if(NOT DEFINED SOC_VERSION)
    set(SOC_VERSION "Ascend910B")
endif()

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
        "${CANN_HOME}/lib64/librt2_registry.a"
        "${CANN_HOME}/lib64/libtiling_api.a"
    -Wl,--no-whole-archive
    "${CANN_HOME}/lib64/libopp_registry.so"
    "${CANN_HOME}/lib64/libregister.so"
    "${CANN_HOME}/lib64/libmmpa.so"
)

add_executable(${UT_EXE}
    ${UT_MAIN_SRC}
    ${UT_TEST_SRCS}
    ${UT_COMMON_SRCS}
)

target_compile_features(${UT_EXE} PRIVATE cxx_std_17)

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
    "${CANN_HOME}/lib64/libmetadef.so"
    "${CANN_HOME}/lib64/libunified_dlog.so"
    "${CANN_HOME}/lib64/libopp_registry.so"
    "${CANN_HOME}/lib64/libregister.so"
    "${CANN_HOME}/lib64/libgraph.so"
    "${CANN_HOME}/lib64/libgraph_base.so"
    "${CANN_HOME}/lib64/libplatform.so"
    "${CANN_HOME}/lib64/libc_sec.so"
    "${CANN_HOME}/lib64/libmmpa.so"
    dl
    pthread
)

set_target_properties(${UT_EXE} PROPERTIES
    INSTALL_RPATH "$ORIGIN;${CANN_HOME}/lib64"
    BUILD_WITH_INSTALL_RPATH TRUE
)

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9)
    target_link_libraries(${UT_EXE} PRIVATE stdc++fs)
endif()
