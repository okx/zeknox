cmake_minimum_required(VERSION 3.22.1)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

include(cmake/platform.cmake)

set(USE_ASM    ON CACHE BOOL "Use asm implementation for Fr and Fq")
set(USE_OPENMP ON CACHE BOOL "Use OpenMP")
set(G2_ENABLED OFF CACHE BOOL "enable G2 curve")

if(G2_ENABLED)
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


project(cryptography_cuda_cpp LANGUAGES CXX CUDA ASM)


if(CMAKE_BUILD_TYPE STREQUAL "Release")
    message("Build on Release Mode")
    set(CMAKE_CUDA_FLAGS_RELEASE "")  # without this setting, the msm_g2 will panic with memory segmentation
else()
    message("Build on Debug Mode")
    set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -g -G -O0")
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

find_package(CUDA REQUIRED)
enable_language("CUDA")

set(PROJECT_CUDA_SOURCE_DIR ${CMAKE_SOURCE_DIR}/cuda)

message("USE_ASM=: ${USE_ASM}")
message("CUDA_INCLUDE_DIRS=: ${CUDA_INCLUDE_DIRS}")
message("CUDA_TOOLKIT_ROOT_DIR=: ${CUDA_TOOLKIT_ROOT_DIR}")
message("CMAKE_SOURCE_DIR=: ${CMAKE_SOURCE_DIR}")
message("CUDA_LIBRARY=: ${CUDA_LIBRARY}")
message("PROJECT_CUDA_SOURCE_DIR=: ${PROJECT_CUDA_SOURCE_DIR}")

include_directories(SYSTEM ${CUDA_INCLUDE_DIRS})
include_directories(${PROJECT_CUDA_SOURCE_DIR})
include_directories("depends/blst/src")  # the cuda library is dependent on blst

set(LIB_SOURCES
    src/lib.h
    src/lib.cu
    cuda/util/all_gpus.cpp
)
add_library(cryptography_cuda STATIC ${LIB_SOURCES} )

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

target_compile_definitions(cryptography_cuda PUBLIC -D__ADX__)
# target_compile_options(cryptography_cuda PRIVATE
#    "-Xcompiler=-ffunction-sections"
# )
target_link_libraries(cryptography_cuda ${CMAKE_SOURCE_DIR}/depends/blst/libblst.a)
# set_target_properties(cryptography_cuda PROPERTIES POSITION_INDEPENDENT_CODE ON)
# set_target_properties(cryptography_cuda PROPERTIES
#                             CUDA_SEPARABLE_COMPILATION ON)
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
FetchContent_MakeAvailable(googletest)

add_subdirectory(cpp)

install(TARGETS test_bn254 bench_bn254 cryptography_cuda
    RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
    BUNDLE DESTINATION ${CMAKE_INSTALL_PREFIX}/app
    LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)

install(FILES src/lib.h
    DESTINATION ${CMAKE_INSTALL_PREFIX}/include)