#ifndef __CRYPTO_UTIL_RUSTERROR_H__
#define __CRYPTO_UTIL_RUSTERROR_H__

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