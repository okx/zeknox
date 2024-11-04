cmake_minimum_required(VERSION 3.22.1)

if(NOT CMAKE_CUDA_TOOLKIT_ROOT_DIR)
  set(CMAKE_CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda")
  set(CMAKE_CUDA_COMPILER "${CMAKE_CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc")
endif()
if(NOT CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH})
endif()
set(CUDA_INCLUDE_DIRS ${CMAKE_CUDA_TOOLKIT_ROOT_DIR}/include)
enable_language(CUDA)

project(cryptography_cuda_cpp LANGUAGES CXX CUDA ASM)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

include(cmake/platform.cmake)

set(USE_ASM    ON CACHE BOOL "Use asm implementation for Fr and Fq")
set(USE_OPENMP ON CACHE BOOL "Use OpenMP")
set(BUILD_MSM OFF CACHE BOOL "Build MSM")
set(G2_ENABLED OFF CACHE BOOL "enable G2 curve")

if(BUILD_MSM)
    message("Build MSM")
    add_definitions(-DBUILD_MSM)
endif()
if(G2_ENABLED)
    message("Build MSM G2 Enabled")
    add_definitions(-DG2_ENABLED)
endif()

if (${CMAKE_VERSION} VERSION_LESS "3.24.0")
    set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH})
else()
    set(CMAKE_CUDA_ARCHITECTURES native) # on 3.24+, on earlier it is ignored, and the target is not passed
endif ()

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 70)
endif()


if(CMAKE_BUILD_TYPE STREQUAL "Release")
    message("Build on Release Mode")
    set(CMAKE_CUDA_FLAGS_RELEASE "")  # without this setting, the msm_g2 will panic with memory segmentation
else()
    message("Build on Debug Mode")
    set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -G -O0")
endif()
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")

set(CUDA_SEPARABLE_COMPILATION ON)

if (NOT EXISTS ${GMP_INCLUDE_FILE_FULLPATH})
    message("WARNING: ${GMP_INCLUDE_FILE_FULLPATH} is not found and so system ${GMP_INCLUDE_FILE} is used.")
endif()

if (NOT EXISTS ${GMP_LIB_FILE_FULLPATH})
    message("WARNING: ${GMP_LIB_FILE_FULLPATH} is not found and so system ${GMP_LIB_FILE} is used.")
    set(GMP_LIB gmp)
endif()

include_directories(BEFORE ${GMP_INCLUDE_DIR})


if(USE_OPENMP)
    find_package(OpenMP)

    if(OpenMP_CXX_FOUND)
        if(CMAKE_HOST_SYSTEM_NAME STREQUAL "Linux")
            message("OpenMP is used")

        else()
            set(OpenMP_CXX_FOUND FALSE)
            message("OpenMP is not used")

        endif()
    endif()
endif()

get_filename_component(CPP_DIR ${CMAKE_SOURCE_DIR} DIRECTORY)
get_filename_component(PROJECT_DIR ${CPP_DIR} DIRECTORY)
message(STATUS "PROJECT_DIR is: ${PROJECT_DIR}")

set(PROJECT_CUDA_SOURCE_DIR ${PROJECT_DIR}/native)

message("USE_ASM=: ${USE_ASM}")
message("CUDA_INCLUDE_DIRS=: ${CUDA_INCLUDE_DIRS}")
message("CUDA_TOOLKIT_ROOT_DIR=: ${CUDA_TOOLKIT_ROOT_DIR}")
message("CMAKE_SOURCE_DIR=: ${CMAKE_SOURCE_DIR}")
# message("CUDA_LIBRARY=: ${CUDA_LIBRARY}")
message("PROJECT_CUDA_SOURCE_DIR=: ${PROJECT_CUDA_SOURCE_DIR}")

include_directories(SYSTEM ${CUDA_INCLUDE_DIRS})
# include_directories("depends/blst/src")  # the cuda library is dependent on blst
# message(STATUS "Include directory: ${INCLUDE_DIR}")

##============================Build CUDA Lib==============================================##
set(IS_CURVE_SUPPORTED FALSE)
set(SUPPORTED_CURVES BN254;GOLDILOCKS)
foreach (SUPPORTED_CURVE ${SUPPORTED_CURVES})
  if (CURVE STREQUAL SUPPORTED_CURVE)
    add_definitions(-DFEATURE_${CURVE})
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DFEATURE_${CURVE}")
    set(IS_CURVE_SUPPORTED TRUE)
  endif ()
endforeach()
if (NOT IS_CURVE_SUPPORTED)
  message( FATAL_ERROR "The value of CURVE variable: ${CURVE} is not one of the supported curves: ${SUPPORTED_CURVES}" )
else()
    message("value of CURVE is ${CURVE}")
endif ()

add_subdirectory(${PROJECT_CUDA_SOURCE_DIR} zeknox)
target_compile_definitions(cryptocuda PRIVATE CURVE=${CURVE})
if (BUILD_MSM)
    target_compile_definitions(cryptocuda PRIVATE BUILD_MSM)
endif()
if (G2_ENABLED)
    target_compile_definitions(cryptocuda PRIVATE G2_ENABLED)
endif()

##============================Build Test & Bench==============================================##
enable_testing()
#include(FetchContent)
#FetchContent_Declare(
#  googletest
#  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
#)
#FetchContent_MakeAvailable(googletest)
find_package(GTest REQUIRED)

link_libraries(${GMP_LIB})
add_definitions(${GMP_DEFINIONS})
if(USE_ASM)
    if(ARCH MATCHES "arm64")
            add_definitions(-DUSE_ASM -DARCH_ARM64)
    elseif(ARCH MATCHES "x86_64")
            add_definitions(-DUSE_ASM -DARCH_X86_64)
    endif()
endif()

if(USE_ASM AND ARCH MATCHES "x86_64")

    if (CMAKE_HOST_SYSTEM_NAME MATCHES "Darwin")
        set(NASM_FLAGS -fmacho64 --prefix _)
    else()
        set(NASM_FLAGS -felf64 -DPIC)
    endif()
    message("use custom command, NASM_FLAGS=${NASM_FLAGS}")

    add_custom_command(OUTPUT ${CMAKE_SOURCE_DIR}/build/fq_asm.o
        COMMAND nasm ${NASM_FLAGS} ${CMAKE_SOURCE_DIR}/field/fq.asm -o fq_asm.o
        DEPENDS ${CMAKE_SOURCE_DIR}/field/fq.asm
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/build)

    add_custom_command(OUTPUT ${CMAKE_SOURCE_DIR}/build/fr_asm.o
        COMMAND nasm ${NASM_FLAGS} ${CMAKE_SOURCE_DIR}/field/fr.asm -o fr_asm.o
        DEPENDS ${CMAKE_SOURCE_DIR}/field/fr.asm
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/build)
endif()

set(FR_SOURCES
    field/fr.hpp
    field/fr.cpp
    )
if(USE_ASM)
    if(ARCH MATCHES "arm64")
        set(FR_SOURCES ${FR_SOURCES} ../build/fr_raw_arm64.s ../build/fr_raw_generic.cpp ../build/fr_generic.cpp)
    elseif(ARCH MATCHES "x86_64")
        set(FR_SOURCES ${FR_SOURCES} ../build/fr_asm.o)
    endif()
else()
    set(FR_SOURCES ${FR_SOURCES} ../build/fr_generic.cpp ../build/fr_raw_generic.cpp)
endif()

add_library(fr STATIC ${FR_SOURCES})
set_target_properties(fr PROPERTIES POSITION_INDEPENDENT_CODE ON)

link_libraries(fr)

set(FQ_SOURCES
    field/fq.hpp
    field/fq.cpp
    )
if(USE_ASM)
    if(ARCH MATCHES "arm64")
        set(FQ_SOURCES ${FQ_SOURCES} ../build/fq_raw_arm64.s ../build/fq_raw_generic.cpp ../build/fq_generic.cpp)
    elseif(ARCH MATCHES "x86_64")
        set(FQ_SOURCES ${FQ_SOURCES} ../build/fq_asm.o)
    endif()
else()
    set(FQ_SOURCES ${FQ_SOURCES} ../build/fq_raw_generic.cpp ../build/fq_generic.cpp)
endif()

add_library(fq STATIC ${FQ_SOURCES})
set_target_properties(fq PROPERTIES POSITION_INDEPENDENT_CODE ON)

link_libraries(fq)


if(OpenMP_CXX_FOUND)
    add_definitions(-DUSE_OPENMP)
    add_compile_options(${OpenMP_CXX_FLAGS})
endif()

set(BN254_TEST_SOURCES
  test_bn254.cpp
  ${CMAKE_SOURCE_DIR}/depends/ffiasm/c/misc.cpp
  ${CMAKE_SOURCE_DIR}/depends/ffiasm/c/naf.cpp
  ${CMAKE_SOURCE_DIR}/depends/ffiasm/c/splitparstr.cpp
  ${CMAKE_SOURCE_DIR}/depends/ffiasm/c/alt_bn128.cpp
)
include_directories(
  field/
  ${CMAKE_SOURCE_DIR}/depends/ffiasm/c
)

add_executable(test_bn254 ${BN254_TEST_SOURCES})
target_include_directories(test_bn254 PRIVATE ${PROJECT_CUDA_SOURCE_DIR})
target_include_directories(test_bn254 PRIVATE ${PROJECT_CUDA_SOURCE_DIR}/depends/blst/src)
target_link_directories(test_bn254 PRIVATE ${PROJECT_CUDA_SOURCE_DIR}/depends/blst)
target_link_libraries(test_bn254 cryptocuda blst)
if(OpenMP_CXX_FOUND)
    if(CMAKE_HOST_SYSTEM_NAME STREQUAL "Linux")
        target_link_libraries(test_bn254 OpenMP::OpenMP_CXX)
    endif()
endif()
target_link_libraries(
    test_bn254
  GTest::gtest_main
)
include(GoogleTest)

gtest_discover_tests(test_bn254)

# install(TARGETS
# test_bn254
#     RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
#     BUNDLE DESTINATION ${CMAKE_INSTALL_PREFIX}/app
#     LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)

# install(FILES src/lib.h
#     DESTINATION ${CMAKE_INSTALL_PREFIX}/include)