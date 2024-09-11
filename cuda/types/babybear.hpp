#ifndef __BABYBEAR_HPP__
#define __BABYBEAR_HPP__

#include "types/int_types.h"
#include "types/monty.hpp"

typedef MontyField31<BabyBearParameters> BabyBearField;

/*
class BabyBearField
{
private:
    u32 val;

public:
    static const u32 ORDER = 0x78000001;    // 2013265921
    static const u32 ORDER_U32 = ORDER;
    static const u64 ORDER_U64 = (u64)ORDER;

    BabyBearField() { this->val = 0; }

    BabyBearField(u32 x)
    {
        assert(x < ORDER);
        this->val = x;
    }

    inline u32 value() const
    {
        return this->val;
    }

    static BabyBearField Zero()
    {
        return BabyBearField(0);
    }

    BabyBearField operator+=(const BabyBearField &rhs)
    {
        this->val = modulo_add(this->val, rhs.value());
        return *this;
    }

    BabyBearField operator+(const BabyBearField &rhs)
    {
        return BabyBearField(modulo_add(this->val, rhs.value()));
    }

    BabyBearField operator-=(const BabyBearField &rhs)
    {
        this->val = modulo_sub(this->val, rhs.value());
        return *this;
    }

    BabyBearField operator-(const BabyBearField &rhs)
    {
        return BabyBearField(modulo_sub(this->val, rhs.value()));
    }

    BabyBearField operator*(const BabyBearField &rhs)
    {
        u64 prod = (u64)this->val * (u64)rhs.value();
        u32 v = monty_reduce<BabyBearParameters>(prod);
        return BabyBearField(v);
    }

    BabyBearField operator*=(const BabyBearField &rhs)
    {
        u64 prod = (u64)this->val * (u64)rhs.value();
        this->val = monty_reduce<BabyBearParameters>(prod);
        return *this;
    }

private:
    static inline u32 modulo_add(u32 x, u32 y)
    {
        return (x < ORDER - y) ? x + y : y - (ORDER - x);
    }

    static inline u32 modulo_sub(u32 x, u32 y)
    {
        return (x > y) ? x - y : x + (ORDER - y);
    }
};
*/

#endif  // __BABYBEAR_HPP__