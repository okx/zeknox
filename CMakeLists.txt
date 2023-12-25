cmake_minimum_required(VERSION 3.22.1)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

project(cryptography_cuda_cpp LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)

find_package(CUDA REQUIRED)
enable_language("CUDA")
include_directories(${CUDA_INCLUDE_DIRS})
message("The value of CUDA_INCLUDE_DIRS is: ${CUDA_INCLUDE_DIRS}")


set(CMAKE_SOURCE_DIR ${CMAKE_SOURCE_DIR}/cuda)
message("The value of CMAKE_SOURCE_DIR is: ${CMAKE_SOURCE_DIR}")
include_directories(SYSTEM ${CMAKE_SOURCE_DIR})  #
include_directories("depends/blst/src")


set(CUDA_SEPARABLE_COMPILATION ON)
add_library(cryptography_cuda SHARED src/lib.cu cuda/util/all_gpus.cpp)

target_compile_features(cryptography_cuda PRIVATE cuda_std_14)
target_compile_definitions(cryptography_cuda PRIVATE -DFEATURE_BN128)

set_target_properties(cryptography_cuda
        PROPERTIES
                CUDA_RUNTIME_LIBRARY Shared
                # CUDA_STANDARD 14 # this one cannot be changed by CMake
                # CUDA_SEPARABLE_COMPILATION ON # not needed for this example
)
# # set_property(TARGET CUDA_COMP PROPERTY CUDA_ARCHITECTURES 86-real 86-virtual)

add_executable(${PROJECT_NAME} cpp/main.cpp)
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)
target_link_libraries(${PROJECT_NAME} cryptography_cuda)