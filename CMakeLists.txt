cmake_minimum_required(VERSION 3.22.1)

include(cmake/platform.cmake)

set(USE_ASM    ON CACHE BOOL "Use asm implementation for Fr and Fq")
set(USE_OPENMP ON CACHE BOOL "Use OpenMP")

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 70)
endif()

project(cryptography_cuda_cpp LANGUAGES CXX CUDA ASM)
set(CMAKE_CXX_STANDARD 17)
# set(CUDA_SEPARABLE_COMPILATION ON)

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
## todo: pass feature as environment variable, or as option
target_compile_definitions(cryptography_cuda PUBLIC -DFEATURE_BN254 -D__ADX__)
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

# install(TARGETS example  test_bn128 bench_bn128 cryptography_cuda
#     RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
#     BUNDLE DESTINATION ${CMAKE_INSTALL_PREFIX}/app
#     LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)

install(FILES src/lib.h
    DESTINATION ${CMAKE_INSTALL_PREFIX}/include)