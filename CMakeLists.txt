cmake_minimum_required(VERSION 3.22.1)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 70 86)
endif()

project(cryptography_cuda_cpp LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)

find_package(CUDA REQUIRED)
enable_language("CUDA")

set(PROJECT_CUDA_SOURCE_DIR ${CMAKE_SOURCE_DIR}/cuda)


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
add_library(cryptography_cuda SHARED ${LIB_SOURCES} )
## todo: pass feature as environment variable, or as option
target_compile_definitions(cryptography_cuda PUBLIC -DFEATURE_BN128)

add_executable(${PROJECT_NAME} main.cpp)
target_link_libraries(
  ${PROJECT_NAME} 
  cryptography_cuda 
)

