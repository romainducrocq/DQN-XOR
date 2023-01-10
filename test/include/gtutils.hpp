#ifndef _GTEST_UTILS_HPP
#define _GTEST_UTILS_HPP

#include <gtest/gtest.h>

#include <cmath>

namespace GTEST_UTILS
{
    inline float FPREC(float x, int n){
        return std::round(std::round(x * std::pow(10.f, n * 1.f)) / 10.f) / std::pow(10.f, (n-1) * 1.f);
    }

    inline void EXPECT_FEQ(float expected, float actual, int precision){
        EXPECT_EQ(GTEST_UTILS::FPREC(expected, precision),
                  GTEST_UTILS::FPREC(actual, precision));
    }
}

#endif