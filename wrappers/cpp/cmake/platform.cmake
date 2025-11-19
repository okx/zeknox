string(TOLOWER "${TARGET_PLATFORM}" TARGET_PLATFORM)

message("Building for " ${TARGET_PLATFORM})

set(GMP_ROOT depends/gmp)

if(TARGET_PLATFORM MATCHES "aarch64")

    set(GMP_PREFIX ${GMP_ROOT}/package_aarch64)
    set(ARCH arm64)

elseif(TARGET_PLATFORM MATCHES "arm64_host")

    set(GMP_PREFIX ${GMP_ROOT}/package)
    set(ARCH arm64)

elseif(TARGET_PLATFORM MATCHES "macos_x86_64")

    set(CMAKE_OSX_ARCHITECTURES x86_64)
    set(GMP_PREFIX ${GMP_ROOT}/package_macos_x86_64)
    set(ARCH x86_64)

elseif(TARGET_PLATFORM MATCHES "macos_arm64")

    set(CMAKE_OSX_ARCHITECTURES arm64)
    set(GMP_PREFIX ${GMP_ROOT}/package_macos_arm64)
    set(ARCH arm64)

else()

    set(GMP_PREFIX ${GMP_ROOT}/package)
    set(ARCH x86_64)

endif()

if (CMAKE_HOST_SYSTEM_NAME MATCHES "Darwin")
    set(GMP_DEFINIONS -D_LONG_LONG_LIMB)
endif()


set(GMP_INCLUDE_DIR ${GMP_PREFIX}/include)
set(GMP_INCLUDE_FILE gmp.h)
set(GMP_LIB_DIR ${GMP_PREFIX}/lib)
set(GMP_LIB_FILE libgmp.a)

set(GMP_LIB_FILE_FULLPATH     ${CMAKE_SOURCE_DIR}/${GMP_LIB_DIR}/${GMP_LIB_FILE})
set(GMP_INCLUDE_FILE_FULLPATH ${CMAKE_SOURCE_DIR}/${GMP_INCLUDE_DIR}/${GMP_INCLUDE_FILE})

set(GMP_LIB ${GMP_LIB_FILE_FULLPATH})

message("CMAKE_HOST_SYSTEM_NAME=" ${CMAKE_HOST_SYSTEM_NAME})
message("CMAKE_SYSTEM_NAME=" ${CMAKE_SYSTEM_NAME})
message("ARCH=" ${ARCH})
