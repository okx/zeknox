# target_include_directories(cryptography_cuda PRIVATE "./depends/blst/src") 
add_executable(example example.cpp)
# target_include_directories(example PRIVATE "../cuda/ec/")  # the cuda library is dependent on blst
# target_include_directories(example PRIVATE "../cuda/util/")  # the cuda library is dependent on blst
# target_include_directories(example PRIVATE "/data/workspace/cliff/cryptography_cuda/depends/blst/src") 
target_include_directories(example PUBLIC "/data/workspace/cliff/cryptography_cuda/depends/blst/src") 
target_compile_definitions(example PUBLIC -DFEATURE_BN128 -D__ADX__)
target_link_libraries(
  example 
  cryptography_cuda 
)

# target_link_libraries(cryptography_cuda ${CMAKE_SOURCE_DIR}/depends/blst/libblst.a)
# target_link_libraries(example ${CMAKE_SOURCE_DIR}/depends/blst/libblst.a)

enable_testing()

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
        COMMAND nasm ${NASM_FLAGS} ${CMAKE_SOURCE_DIR}/cpp/field/fq.asm -o fq_asm.o
        DEPENDS ${CMAKE_SOURCE_DIR}/cpp/field/fq.asm
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/build)

    add_custom_command(OUTPUT ${CMAKE_SOURCE_DIR}/build/fr_asm.o
        COMMAND nasm ${NASM_FLAGS} ${CMAKE_SOURCE_DIR}/cpp/field/fr.asm -o fr_asm.o
        DEPENDS ${CMAKE_SOURCE_DIR}/cpp/field/fr.asm
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


set(BN128_TEST_SOURCES
  test_bn128.cpp
  ../depends/ffiasm/c/misc.cpp
  ../depends/ffiasm/c/naf.cpp
  ../depends/ffiasm/c/splitparstr.cpp
  ../depends/ffiasm/c/alt_bn254.cpp
)
include_directories(
  field/
  ../depends/ffiasm/c
)

add_executable(test_bn128 ${BN128_TEST_SOURCES})
target_link_libraries(test_bn128 cryptography_cuda)
if(OpenMP_CXX_FOUND)
    if(CMAKE_HOST_SYSTEM_NAME STREQUAL "Linux")
        target_link_libraries(test_bn128 OpenMP::OpenMP_CXX)
    endif()
endif()
target_link_libraries(
    test_bn128
  GTest::gtest_main
)
include(GoogleTest)
gtest_discover_tests(test_bn128)

set(BN128_BENCH_SOURCES
  bench_bn128.cpp
  ../depends/ffiasm/c/misc.cpp
  ../depends/ffiasm/c/naf.cpp
  ../depends/ffiasm/c/splitparstr.cpp
  ../depends/ffiasm/c/alt_bn254.cpp
)
include_directories(
  field/
  ../depends/ffiasm/c
)

add_executable(bench_bn128 ${BN128_BENCH_SOURCES})
target_link_libraries(bench_bn128 cryptography_cuda)
if(OpenMP_CXX_FOUND)
    if(CMAKE_HOST_SYSTEM_NAME STREQUAL "Linux")
        target_link_libraries(bench_bn128 OpenMP::OpenMP_CXX)
    endif()
endif()