#ifndef ZEKNOX_CUDA_UTIL_EXCEPTION_HPP_
#define ZEKNOX_CUDA_UTIL_EXCEPTION_HPP_

#include <cstdio>
#include <cstring>
#include <string>
#include <stdexcept>

class zeknox_error : public std::runtime_error {
    int _code;

    template<typename... Types>
    inline std::string fmt_errno(int errnum, const char* fmt, Types... args)
    {
        const size_t ERRLEN = 48;  // max len of the system error message
        size_t len = std::snprintf(nullptr, 0, fmt, args...);
        std::string ret(len + ERRLEN, '\0');
        std::snprintf(&ret[0], len + 1, fmt, args...);
        auto errmsg = &ret[len];  // reference to the ret string starts at len
#if defined(_WIN32)
        (void)strerror_s(errmsg, ERRLEN, errnum);
#elif defined(_GNU_SOURCE)
        auto errstr = strerror_r(errnum, errmsg, ERRLEN);  // obtain a human-readable description of an error code, as given by errnum
        if (errstr != errmsg)
            strncpy(errmsg, errstr, ERRLEN - 1);
#else
        (void)strerror_r(errnum, errmsg, ERRLEN);
#endif
        ret.resize(len + std::strlen(errmsg));
        return ret;
    }

public:
    zeknox_error(int err_code, const std::string& reason) : std::runtime_error{reason}
    {   _code = err_code;   }
    zeknox_error(int err_code, const char* msg = "") : std::runtime_error{fmt_errno(err_code, "%s", msg)}
    {   _code = err_code;   }
    template<typename... Types>
    zeknox_error(int err_code, const char* fmt, Types... args) : std::runtime_error{fmt_errno(err_code, fmt, args...)}
    {   _code = err_code;   }
    inline int code() const
    {   return _code;   }
};

template<typename... Types>
inline std::string fmt(const char* fmt, Types... args)
{
    size_t len = std::snprintf(nullptr, 0, fmt, args...);
    std::string ret(++len, '\0');
    std::snprintf(&ret.front(), len, fmt, args...);
    ret.resize(--len);
    return ret;
}

#endif
