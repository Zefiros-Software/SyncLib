/**
 * @cond ___LICENSE___
 *
 * Copyright (c) 2016-2018 Zefiros Software.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * @endcond
 */
#pragma once
#ifndef __SYNCLIB_HELPER_H__
#define __SYNCLIB_HELPER_H__

#include "gtest/gtest-typed-test.h"
#include "gtest/gtest.h"

#define CONCATEXT(a, b) a##b
#define CONCAT(a, b) CONCATEXT(a, b)
#define P(prefix) CONCAT(PREFIX, prefix)

#ifndef TEST_FILES_DIR
#ifndef _MSC_VER
#   define TEST_FILES_DIR "test/test-files/"
#else
#   define TEST_FILES_DIR "../../test/test-files/"
#endif
#endif

#define TEST_FILE_DETAIL(file, suite, seperator, test, extension) file #suite #seperator #test #extension
#define TEST_FILE(suite, test) TEST_FILE_DETAIL(TEST_FILES_DIR, suite, _, test, .bin)

template< typename tT >
inline void ExpectEqual(const tT &t1, const tT &t2, const std::string &message = "")
{
    EXPECT_EQ(t1, t2) << message;
}

template<>
inline void ExpectEqual< float >(const float &f1, const float &f2, const std::string &message)
{
    EXPECT_FLOAT_EQ(f1, f2) << message;
}

template<>
inline void ExpectEqual< double >(const double &d1, const double &d2, const std::string &message)
{
    EXPECT_DOUBLE_EQ(d1, d2) << message;
}

inline void ExpectEqual(const char *c1, const char *c2, const std::string &message = "")
{
    std::string s1(c1), s2(c2);
    EXPECT_EQ(s1, s2) << message;
}

typedef uint8_t U8;
typedef uint16_t U16;
typedef uint32_t U32;
typedef uint64_t U64;
typedef int8_t S8;
typedef int16_t S16;
typedef int32_t S32;
typedef int64_t S64;
typedef typename std::make_signed<size_t>::type s_size_t;

template< typename tT >
struct SizeHelper
{
    static size_t GetSize(const tT &)
    {
        return sizeof(tT);
    }
};

template<>
struct SizeHelper< std::string >
{
    static size_t GetSize(const std::string &str)
    {
        return str.size();
    }
};

template< typename tT >
inline void ExpectEqual(const std::vector<tT> &t1, const std::vector<tT> &t2, const std::string &message = "")
{
    ExpectEqual(t1.size(), t2.size(), message);

    for (size_t i = 0, end = t1.size(); i < end; ++i)
    {
        ExpectEqual(t1[i], t2[i], " vectors differ at index " + std::to_string(i));
    }
}

extern uint32_t g_seed;

inline int GetFastRand()
{
    g_seed = (214013 * g_seed + 2531011);
    return (g_seed >> 16) & 0x7FFF;
}

template< typename tT >
inline tT GetRandom()
{
    return static_cast<tT>(GetFastRand());
}

inline float GetRandomFloatNormalized()
{
    // return with max an arbitrary number
    return static_cast<float>(static_cast<double>(GetFastRand()) / 0x7FFF);
}

template<>
inline float GetRandom<float>()
{
    return static_cast<float>(GetRandomFloatNormalized() * 100000.0 - 50000.0);
}

template<>
inline bool GetRandom< bool >()
{
    // return with max an arbitrary number
    return (GetRandom<uint32_t>() % 2) == 1;
}

inline double GetRandomDoubleNormalized()
{
    // return with max an arbitrary number
    return static_cast<double>(GetFastRand()) / 0x7FFF;
}

template<>
inline double GetRandom<double>()
{
    return static_cast<float>(GetRandomDoubleNormalized() * 1000000.0 - 500000.0);
}

std::string GenerateRandomString(bool random = true, uint32_t index = 0);

template<>
inline std::string GetRandom<std::string>()
{
    return GenerateRandomString();
}

template< typename tT, typename tSeed = uint32_t >
tSeed MakeSeed(uint32_t gen)
{
    return static_cast<tSeed>(gen * (std::is_same< tT, std::string >::value ? 32 : sizeof(tT)));
}

#define  PP_COMMA_DIRECT() ,
#define PP_COMMA() PP_COMMA_DIRECT()

#define SERIALISATION_TEST2(test, name, testClass, testClass2, type, init1, init2)    \
TEST(P(test), type ## name ## _stream)                                              \
{                                                                                       \
    testClass tc1(init1);                                                             \
    testClass2 tc2(init2);                                                            \
    SimpleSerialiseDeserialiseStream(tc1, tc2);                                       \
    tc1.TestEqual(tc2);                                                               \
}                                                                                       \
TEST(P(test), type ## name ## _file)                                                \
{                                                                                       \
    testClass tc1(init1);                                                             \
    testClass2 tc2(init2);                                                            \
    SimpleSerialiseDeserialiseFile(tc1, tc2);                                         \
    tc1.TestEqual(tc2);                                                               \
}                                                                                       \
TEST(P(test), type ## name ## _backwards)                                           \
{                                                                                       \
    testClass tc1(init1);                                                             \
    testClass2 tc2(init2);                                                            \
    std::string file = TEST_FILE(test, type ## name);                                 \
    SimpleSerialiseDeserialiseBackwards(file, tc1, tc2);                              \
    tc1.TestEqual(tc2);                                                               \
}

#define SERIALISATION_TEST(test, name, testClass, type, init1, init2)                 \
TEST(P(test), type ## name ## _stream)                                              \
{                                                                                       \
    testClass tc1(init1), tc2(init2);                                               \
    SimpleSerialiseDeserialiseStream(tc1, tc2);                                       \
    tc1.TestEqual(tc2);                                                               \
}                                                                                       \
TEST(P(test), type ## name ## _file)                                                \
{                                                                                       \
    testClass tc1(init1), tc2(init2);                                               \
    SimpleSerialiseDeserialiseFile(tc1, tc2);                                         \
    tc1.TestEqual(tc2);                                                               \
}                                                                                       \
TEST(P(test), type ## name ## _backwards)                                           \
{                                                                                       \
    testClass tc1(init1), tc2(init2);                                               \
    std::string file = TEST_FILE(test, type ## name);                                 \
    SimpleSerialiseDeserialiseBackwards(file, tc1, tc2);                              \
    tc1.TestEqual(tc2);                                                               \
}

#endif