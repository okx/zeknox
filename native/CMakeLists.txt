cmake_minimum_required(VERSION 3.18)

include(CheckLanguage)
check_language(CXX)
enable_language(CXX)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()


# Options
option(USE_CUDA "Enable CUDA code (ON by default)" ON)
set(CUDA_ARCH "86;89" CACHE STRING "CUDA architecture")
option(USE_AVX "Enable AVX acceleration" OFF)
option(USE_AVX512 "Enable AVX512 (and AVX) acceleration" OFF)
option(BUILD_TESTS "Build tests" OFF)
option(BUILD_MSM "Include msm" OFF)
option(G2_ENABLED "Enable G2" OFF) # whether enable msm on G2 curve; it takes very long to build; OFF by default

set(EXTRA_DEFAULT_FLAGS "-DEXPOSE_C_INTERFACE -DTAKE_RESPONSIBILITY_FOR_ERROR_MESSAGE")
if(BUILD_MSM)
    set(EXTRA_DEFAULT_FLAGS "-DBUILD_MSM ${EXTRA_DEFAULT_FLAGS}")
    if(G2_ENABLED)
        set(EXTRA_DEFAULT_FLAGS "-DG2_ENABLED ${EXTRA_DEFAULT_FLAGS}")
    endif()
endif()

set(CMAKE_CXX_FLAGS_DEBUG "-Wall -g ${EXTRA_DEFAULT_FLAGS}")
set(CMAKE_CXX_FLAGS_RELEASE "-Wall -O3 ${EXTRA_DEFAULT_FLAGS}")
set(CMAKE_DEBUG_POSTFIX d)

find_package(OpenMP)

set(LIB_CXX_NAME "cryptocudacpp")
set(LIB_CUDA_NAME "cryptocudacuda")
set(LIB_NAME "cryptocuda")
set(EXE_TESTS_CUDA_NAME "tests.exe")
set(EXE_TESTS_CXX_NAME "tests_cpp.exe")

if (CURVE STREQUAL "BN254")
    set(CURVE_FEATURE "FEATURE_BN254")
endif ()

if(NOT CURVE)
    set(CURVE_FEATURE "FEATURE_GOLDILOCKS")
endif()
message("zeknox CURVE_FEATURE=${CURVE_FEATURE}, BUILD_MSM=${BUILD_MSM}")

# CUDA support
if(USE_CUDA)

# enable CUDA debug
if(CMAKE_BUILD_TYPE MATCHES Debug)
message("debug mode (enable CUDA debug)")
set(CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS} "-g -G -maxrregcount=168")
set(CUDA_NVCC_FLAGS "-g -G -maxrregcount=168")
endif()

if(NOT CMAKE_CUDA_TOOLKIT_ROOT_DIR)
  set(CMAKE_CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda")
  set(CMAKE_CUDA_COMPILER "${CMAKE_CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc")
endif()
if(NOT CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH})
endif()
check_language(CUDA)
enable_language(CUDA)
project(cryptocuda VERSION 1.0.0 LANGUAGES CUDA CXX)

# specify cuda runtime lib
set(CMAKE_CUDA_RUNTIME_LIBRARY, "Static")

set(LIB_INC ${PROJECT_SOURCE_DIR})
set(DEP_GOLDILOCKS_DIR "${PROJECT_SOURCE_DIR}/depends/goldilocks/src")

# 1. Build C++ objects (no -DUSE_CUDA)
set(LIB_CXX_SRC
    keccak/keccak.cu
    poseidon/poseidon.cpp
    poseidon/element_bn128.cu
    poseidon/poseidon_bn128.cu
    poseidon2/poseidon2.cu
    monolith/monolith.cu
    merkle/merkle.cpp
    merkle/merkle_avx.cpp)

add_library(${LIB_CXX_NAME} OBJECT ${LIB_CXX_SRC})
target_compile_definitions(${LIB_CXX_NAME} PRIVATE FEATURE_GOLDILOCKS)
set_target_properties(${LIB_CXX_NAME} PROPERTIES CUDA_ARCHITECTURES "${CUDA_ARCH}")
set_target_properties(${LIB_CXX_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(${LIB_CXX_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(${LIB_CXX_NAME} PROPERTIES DEBUG_POSTFIX ${CMAKE_DEBUG_POSTFIX})
target_include_directories(${LIB_CXX_NAME} PRIVATE ${LIB_INC})
target_compile_options(${LIB_CXX_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-dc>)
if(OpenMP_CXX_FOUND)
    target_link_libraries(${LIB_CXX_NAME} PUBLIC OpenMP::OpenMP_CXX)
endif()
if(USE_AVX)
    target_include_directories(${LIB_CXX_NAME} PUBLIC ${DEP_GOLDILOCKS_DIR})
    target_compile_definitions(${LIB_CXX_NAME} PUBLIC __USE_AVX__ __NO_GMP__)
    target_compile_options(${LIB_CXX_NAME} PUBLIC -mavx2)
endif()
if(USE_AVX512)
    target_include_directories(${LIB_CXX_NAME} PUBLIC ${DEP_GOLDILOCKS_DIR})
    target_compile_definitions(${LIB_CXX_NAME} PUBLIC __USE_AVX__ __AVX512__ __NO_GMP__)
    target_compile_options(${LIB_CXX_NAME} PUBLIC -march=native)
endif()

# 2. Build CUDA objects (add -DUSE_CUDA)
set(LIB_CUDA_SRC
    keccak/keccak.cu
    poseidon/poseidon.cu
    poseidon/element_bn128.cu
    poseidon/poseidon_bn128.cu
    poseidon2/poseidon2.cu
    monolith/monolith.cu
    merkle/merkle.cu
    ntt/ntt.cu
    utils/all_gpus.cu)

if (BUILD_MSM)
    list(APPEND LIB_CUDA_SRC msm/msm.cu)
endif()

add_library(${LIB_CUDA_NAME} OBJECT ${LIB_CUDA_SRC})

target_compile_definitions(${LIB_CUDA_NAME} PRIVATE ${CURVE_FEATURE} USE_CUDA ${EXTRA_DEFAULT_FLAGS})
target_include_directories(${LIB_CUDA_NAME} PRIVATE ${LIB_INC})
target_include_directories(${LIB_CUDA_NAME} PRIVATE ${PROJECT_SOURCE_DIR}/depends/blst/src)
target_compile_definitions(${LIB_CUDA_NAME} PRIVATE -D${CURVE_FEATURE} -D__ADX__) ## __ADX__ is required for blst
set_target_properties(${LIB_CUDA_NAME} PROPERTIES CUDA_ARCHITECTURES "${CUDA_ARCH}")
set_target_properties(${LIB_CUDA_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(${LIB_CUDA_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(${LIB_CUDA_NAME} PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_compile_options(${LIB_CUDA_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-dc>)
set_target_properties(${LIB_CUDA_NAME} PROPERTIES DEBUG_POSTFIX ${CMAKE_DEBUG_POSTFIX})
target_link_libraries(${LIB_CUDA_NAME} PRIVATE cudart)

# 3. Build the final library
add_library(${LIB_NAME} STATIC lib.cu $<TARGET_OBJECTS:${LIB_CUDA_NAME}> $<TARGET_OBJECTS:${LIB_CXX_NAME}>)
target_compile_definitions(${LIB_NAME} PRIVATE ${CURVE_FEATURE} USE_CUDA ${EXTRA_DEFAULT_FLAGS})
message(STATUS "FINAL_LIB_INCLUDE ${PROJECT_SOURCE_DIR}/depends/blst/src")
target_include_directories(${LIB_NAME} PRIVATE ${PROJECT_SOURCE_DIR}/depends/blst/src)
target_compile_definitions(${LIB_NAME} PRIVATE -D${CURVE_FEATURE} -D__ADX__) ## __ADX__ is required for blst
target_include_directories(${LIB_NAME} PRIVATE ${LIB_INC})
set_target_properties(${LIB_NAME} PROPERTIES CUDA_ARCHITECTURES "${CUDA_ARCH}")
set_target_properties(${LIB_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(${LIB_NAME} PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_compile_options(${LIB_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-dc>)
target_link_libraries(${LIB_NAME} PRIVATE cudart)

# Build tests?
if (BUILD_TESTS)
add_executable(${EXE_TESTS_CUDA_NAME} tests/tests.cu $<TARGET_OBJECTS:${LIB_CXX_NAME}> $<TARGET_OBJECTS:${LIB_CUDA_NAME}>)
target_compile_definitions(${EXE_TESTS_CUDA_NAME} PRIVATE USE_CUDA)
target_include_directories(${EXE_TESTS_CUDA_NAME} PRIVATE ${LIB_INC})
set_target_properties(${EXE_TESTS_CUDA_NAME} PROPERTIES CUDA_ARCHITECTURES "${CUDA_ARCH}")
set_target_properties(${EXE_TESTS_CUDA_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(${EXE_TESTS_CUDA_NAME} PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_compile_options(${EXE_TESTS_CUDA_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-dc>)
target_link_libraries(${EXE_TESTS_CUDA_NAME} PRIVATE gtest gomp)
set_target_properties(${EXE_TESTS_CUDA_NAME} PROPERTIES DEBUG_POSTFIX ${CMAKE_DEBUG_POSTFIX})
if(OpenMP_CXX_FOUND)
    target_link_libraries(${EXE_TESTS_CUDA_NAME} PUBLIC OpenMP::OpenMP_CXX)
endif()
if(USE_AVX)
    target_include_directories(${EXE_TESTS_CUDA_NAME} PRIVATE ${DEP_GOLDILOCKS_DIR})
    target_compile_definitions(${EXE_TESTS_CUDA_NAME} PRIVATE __USE_AVX__ __NO_GMP__)
    target_compile_options(${EXE_TESTS_CUDA_NAME} PRIVATE -mavx2)
endif()
if(USE_AVX512)
    target_include_directories(${EXE_TESTS_CUDA_NAME} PRIVATE ${DEP_GOLDILOCKS_DIR})
    target_compile_definitions(${EXE_TESTS_CUDA_NAME} PRIVATE __USE_AVX__ __AVX512__ __NO_GMP__)
    target_compile_options(${EXE_TESTS_CUDA_NAME} PRIVATE -march=native)
endif()
endif() # BUILD_TESTS

else() # USE_CUDA
# only C++ support

set(LIB_INC
${PROJECT_SOURCE_DIR})

project(cryptocudacxx VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_FLAGS "-Wall -I${PROJECT_SOURCE_DIR} ${EXTRA_DEFAULT_FLAGS}")

set(LIB_CXX_SRC
    keccak/keccak.cu
    poseidon/poseidon.cpp
    poseidon/element_bn128.cu
    poseidon/poseidon_bn128.cu
    poseidon2/poseidon2.cu
    monolith/monolith.cu
    merkle/merkle.cpp
    merkle/merkle_avx.cpp)

set_source_files_properties(${LIB_CXX_SRC} PROPERTIES LANGUAGE CXX)
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  add_definitions("-x c++")
endif()

add_library(${LIB_CXX_NAME} STATIC ${LIB_CXX_SRC})
target_compile_definitions(${LIB_CXX_NAME} PRIVATE FEATURE_GOLDILOCKS)
set_target_properties(${LIB_CXX_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(${LIB_CXX_NAME} PROPERTIES DEBUG_POSTFIX ${CMAKE_DEBUG_POSTFIX})
target_include_directories(${LIB_CXX_NAME} PRIVATE ${LIB_INC})
if(OpenMP_CXX_FOUND)
    target_link_libraries(${LIB_CXX_NAME} PRIVATE OpenMP::OpenMP_CXX)
endif()
if(USE_AVX)
    target_include_directories(${LIB_CXX_NAME} PRIVATE ${DEP_GOLDILOCKS_DIR})
    target_compile_definitions(${LIB_CXX_NAME} PRIVATE __USE_AVX__ __NO_GMP__)
    target_compile_options(${LIB_CXX_NAME} PRIVATE -mavx2)
endif()
if(USE_AVX512)
    target_include_directories(${LIB_CXX_NAME} PRIVATE ${DEP_GOLDILOCKS_DIR})
    target_compile_definitions(${LIB_CXX_NAME} PRIVATE __USE_AVX__ __AVX512__ __NO_GMP__)
    target_compile_options(${LIB_CXX_NAME} PRIVATE -march=native)
endif()

# Tests target
if (BUILD_TESTS)
add_executable(${EXE_TESTS_CXX_NAME} tests/tests.cu)
set_source_files_properties(tests/tests.cu PROPERTIES LANGUAGE CXX)
target_link_libraries(${EXE_TESTS_CXX_NAME} PRIVATE ${LIB_CXX_NAME} gtest)
set_target_properties(${EXE_TESTS_CXX_NAME} PROPERTIES DEBUG_POSTFIX ${CMAKE_DEBUG_POSTFIX})
target_include_directories(${EXE_TESTS_CXX_NAME} PRIVATE ${LIB_INC})
if(USE_AVX)
    target_include_directories(${EXE_TESTS_CXX_NAME} PRIVATE ${DEP_GOLDILOCKS_DIR})
    target_compile_definitions(${EXE_TESTS_CXX_NAME} PRIVATE __USE_AVX__ __NO_GMP__)
    target_compile_options(${EXE_TESTS_CXX_NAME} PRIVATE -mavx2)
endif()
if(USE_AVX512)
    target_include_directories(${EXE_TESTS_CXX_NAME} PRIVATE ${DEP_GOLDILOCKS_DIR})
    target_compile_definitions(${EXE_TESTS_CXX_NAME} PRIVATE __USE_AVX__ __AVX512__ __NO_GMP__)
    target_compile_options(${EXE_TESTS_CXX_NAME} PRIVATE -march=native)
endif()
endif() # BUILD_TESTS

endif() # USE_CUDA
