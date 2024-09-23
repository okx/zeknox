// Copyright 2024 OKX
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef ZEKNOX_CUDA_UTIL_RUSTERROR_H_
#define ZEKNOX_CUDA_UTIL_RUSTERROR_H_

#ifdef __cplusplus
# include <string>
# include <cstring>
#else
# include <string.h>
#endif


struct RustError { /* to be returned exclusively by value */
    int code;
    char *message;
#ifdef __cplusplus
    RustError(int e = 0) : code(e)
    {   message = nullptr;   }
    RustError(int e, const std::string& str) : code(e)
    {   message = str.empty() ? nullptr : strdup(str.c_str());   }
    RustError(int e, const char *str) : code(e)
    {   message = str==nullptr ? nullptr : strdup(str);   }
    // no destructor[!], Rust takes care of the |message|

    struct by_value {
        int code;
        char *message;
    };
    operator by_value() const { return {code, message}; }
#endif
};
#ifndef __cplusplus
typedef struct RustError RustError;
#endif

#endif