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
#include "helper.h"

#include "sync/util/ranges/range.h"
#include "sync/env/shared.h"

#include <stdint.h>
#include <thread>
#include <tuple>
#include <numeric>

//#define TEST_RANGE(type)

template<typename tT, bool tIsSigned = std::numeric_limits<tT>::is_signed>
struct RangeHelper
{
    static tT MakeStrictlyPostive(const tT &value)
    {
        return value > 0 ? value : 1;
    }
};

template<typename tT>
struct RangeHelper<tT, true>
{
    static tT MakeStrictlyPostive(const tT &value)
    {
        return value < 0 ? -value : (value > 0 ? value : 1);
    }
};

template<typename tT, typename tMax>
tT GetRandomRange(const tMax &max)
{
    return GetRandom<tT>() % static_cast<tT>(max);
}

template<typename tT>
struct RandomHelper
{
    template<typename tMax>
    static tT GetRandom(const tMax &max)
    {
        return ::GetRandom<tT>() % static_cast<tT>(max);
    }
};

template<>
struct RandomHelper<float>
{
    template<typename tMax>
    static float GetRandom(const tMax &max)
    {
        return GetRandomFloatNormalized() * max;
    }
};

template<>
struct RandomHelper<double>
{
    template<typename tMax>
    static double GetRandom(const tMax &max)
    {
        return GetRandomDoubleNormalized() * max;
    }
};

template<typename tT>
void ExpectRangeMaxReached(const tT &actual, const tT &expect)
{
    EXPECT_EQ(expect, actual);
}

template<>
void ExpectRangeMaxReached<float>(const float &actual, const float &expect)
{
    EXPECT_GE(actual, expect);
    EXPECT_LE(actual, expect + 1.0);
}

template<>
void ExpectRangeMaxReached<double>(const double &actual, const double &expect)
{
    EXPECT_GE(actual, expect);
    EXPECT_LE(actual, expect + 1.0);
}

template<typename tT>
void TestRange(size_t iterations = 100)
{
    g_seed = 42;

    for (size_t it = 0; it < iterations; ++it)
    {
        tT iExpect = 0;
        tT iMax = RandomHelper<tT>::GetRandom(100000) + static_cast<tT>(100);

        iMax = RangeHelper<tT>::MakeStrictlyPostive(iMax);

        for (tT i : SyncLib::Ranges::Range<tT>(iMax))
        {
            EXPECT_EQ(iExpect++, i);
        }

        ExpectRangeMaxReached(iExpect, iMax);
    }
}

template<typename tT>
void TestRangeReverse(size_t iterations = 100)
{
    g_seed = 42;

    for (size_t it = 0; it < iterations; ++it)
    {
        tT iMax = RandomHelper<tT>::GetRandom(50) + static_cast<tT>(10);

        iMax = RangeHelper<tT>::MakeStrictlyPostive(iMax);
        tT iExpect = iMax - 1;

        for (tT i : SyncLib::Ranges::Range<tT>(iMax).Reverse())
        {
            EXPECT_EQ(iExpect--, i);
        }

        tT expectedEnd = 0;
        --expectedEnd;

        ExpectRangeMaxReached<tT>(iExpect, expectedEnd);
    }
}

template<typename tT>
void TestRangeWithLower(size_t iterations = 100)
{
    g_seed = 42;

    for (size_t it = 0; it < iterations; ++it)
    {
        tT u = RandomHelper<tT>::GetRandom(100000) + static_cast<tT>(100);
        tT v = RandomHelper<tT>::GetRandom(100);

        u = RangeHelper<tT>::MakeStrictlyPostive(u);
        v = RangeHelper<tT>::MakeStrictlyPostive(v);

        tT iMin = std::min(u, v);
        tT iMax = std::max(u, v);
        tT iExpect = iMin;

        for (tT i : SyncLib::Ranges::Range<tT>(iMin, iMax))
        {
            EXPECT_EQ(iExpect++, i);
        }

        ExpectRangeMaxReached(iExpect, iMax);
    }
}

template<typename tT>
void TestRangeWithLowerReverse(size_t iterations = 100)
{
    g_seed = 42;

    for (size_t it = 0; it < iterations; ++it)
    {
        tT u = RandomHelper<tT>::GetRandom(100000) + static_cast<tT>(100);
        tT v = RandomHelper<tT>::GetRandom(100);

        u = RangeHelper<tT>::MakeStrictlyPostive(u);
        v = RangeHelper<tT>::MakeStrictlyPostive(v);

        tT iMin = std::min(u, v);
        tT iMax = std::max(u, v);
        tT iExpect = iMax - 1;

        for (tT i : SyncLib::Ranges::Range<tT>(iMin, iMax).Reverse())
        {
            EXPECT_EQ(iExpect--, i);
        }

        ExpectRangeMaxReached<tT>(iExpect, iMin - 1);
    }
}

template<typename tT>
void TestStepRange(size_t iterations = 100)
{
    g_seed = 42;

    for (size_t it = 0; it < iterations; ++it)
    {
        tT iExpect = 0;
        tT iMax = RandomHelper<tT>::GetRandom(100000) + static_cast<tT>(100);
        tT step = RandomHelper<tT>::GetRandom(17);

        step = RangeHelper<tT>::MakeStrictlyPostive(step);
        iMax = RangeHelper<tT>::MakeStrictlyPostive(iMax);

        for (tT i : SyncLib::Ranges::Range<tT>(0, iMax, step))
        {
            EXPECT_EQ(iExpect, i);
            iExpect += step;
        }

        ExpectRangeMaxReached<tT>(iExpect, (iMax + step - 1) / step * step);
    }
}

#define TEST_RANGE(type)    \
TEST(P(Range), type)        \
{                           \
    TestRange<type>();      \
}

#define TEST_RANGE_REVERSE(type)    \
TEST(P(RangeReverse), type)         \
{                                   \
    TestRangeReverse<type>();       \
}

#define TEST_RANGE_WITH_LOWER(type)     \
TEST(P(RangeWithLower), type)           \
{                                       \
    TestRangeWithLower<type>();         \
}

#define TEST_RANGE_WITH_LOWER_REVERSE(type) \
TEST(P(RangeWithLowerReverse), type)        \
{                                           \
    TestRangeWithLowerReverse<type>();      \
}

#define TEST_STEP_RANGE(type)    \
TEST(P(StepRange), type)        \
{                           \
    TestStepRange<type>();      \
}

#define TEST_INTEGER_TYPES(TESTABLE_MACRO)  \
TESTABLE_MACRO(U8);                     \
TESTABLE_MACRO(U16);                    \
TESTABLE_MACRO(U32);                    \
TESTABLE_MACRO(U64);                    \
TESTABLE_MACRO(size_t);                 \
TESTABLE_MACRO(S8);                     \
TESTABLE_MACRO(S16);                    \
TESTABLE_MACRO(S32);                    \
TESTABLE_MACRO(S64);                    \
TESTABLE_MACRO(s_size_t);

TEST_INTEGER_TYPES(TEST_RANGE);
TEST_INTEGER_TYPES(TEST_RANGE_REVERSE);
TEST_INTEGER_TYPES(TEST_RANGE_WITH_LOWER);
TEST_INTEGER_TYPES(TEST_RANGE_WITH_LOWER_REVERSE);
TEST_INTEGER_TYPES(TEST_STEP_RANGE);

template<typename tT>
void TestPutValue(size_t p)
{
    using tEnv = SyncLib::Environments::SharedMemoryBSP;

    for (size_t offset : SyncLib::Ranges::Range(p))
    {
        tEnv env;

        env.Run(p, [](tEnv & env, const size_t offset)
        {
            size_t s = env.Rank();
            size_t p = env.Size();

            tT unshared = GetRandom<tT>();
            auto shared = env.ShareValue<tT>(GetRandom<tT>());
            auto sharedValidation = env.ShareValue<tT>(GetRandom<tT>());

            shared.Put((s + offset) % p, unshared);
            env.Sync();
            sharedValidation.Put((s - offset + p) % p, shared.Value());
            env.Sync();

            EXPECT_EQ(unshared, sharedValidation.Value());
        }, offset);
    }
}

template<typename tT>
void TestPutArrayValue(size_t p)
{
    using tEnv = SyncLib::Environments::SharedMemoryBSP;

    for (size_t offset : SyncLib::Ranges::Range(p))
    {
        tEnv env;

        env.Run(p, [](tEnv & env, const size_t offset)
        {
            size_t s = env.Rank();
            size_t p = env.Size();

            std::vector<tT> unshared(p);
            auto shared = env.ShareArray<tT>(p);
            auto sharedValidation = env.ShareArray<tT>(p);

            for (auto &x : unshared)
            {
                x = GetRandom<tT>();
            }


            for (auto &x : shared)
            {
                x = GetRandom<tT>();
            }


            for (auto &x : sharedValidation)
            {
                x = GetRandom<tT>();
            }

            {
                size_t t = (s + offset) % p;

                for (size_t i : SyncLib::Ranges::Range(p))
                {
                    shared.PutValue(t, unshared[i], i);
                }
            }

            env.Sync();

            {
                size_t t = (s - offset + p) % p;

                for (size_t i : SyncLib::Ranges::Range(p))
                {
                    sharedValidation.PutValue(t, shared[i], i);
                }
            }
            env.Sync();

            EXPECT_EQ(unshared, sharedValidation.Value());
        }, offset);
    }
}

template<typename tT>
void TestPutArray(size_t p)
{
    using tEnv = SyncLib::Environments::SharedMemoryBSP;

    for (size_t offset : SyncLib::Ranges::Range(p))
    {
        tEnv env;

        env.Run(p, [](tEnv & env, const size_t offset)
        {
            size_t s = env.Rank();
            size_t p = env.Size();

            std::vector<tT> unshared(p);
            auto shared = env.ShareArray<tT>(p);
            auto sharedValidation = env.ShareArray<tT>(p);

            for (auto &x : unshared)
            {
                x = GetRandom<tT>();
            }


            for (auto &x : shared)
            {
                x = GetRandom<tT>();
            }


            for (auto &x : sharedValidation)
            {
                x = GetRandom<tT>();
            }

            {
                size_t t = (s + offset) % p;

                for (size_t i : SyncLib::Ranges::Range(p))
                {
                    shared.Put(t, unshared.begin() + i, unshared.begin() + i + 1, i);
                }
            }

            env.Sync();

            {
                size_t t = (s - offset + p) % p;

                for (size_t i : SyncLib::Ranges::Range(p))
                {
                    sharedValidation.Put(t, shared.begin() + i, shared.begin() + i + 1, i);
                }
            }
            env.Sync();

            EXPECT_EQ(unshared, sharedValidation.Value());
        }, offset);
    }
}

template<typename tT>
void TestGetValue(size_t p)
{
    using tEnv = SyncLib::Environments::SharedMemoryBSP;

    for (size_t offset : SyncLib::Ranges::Range(p))
    {
        tEnv env;

        env.Run(p, [](tEnv & env, const size_t offset)
        {
            size_t s = env.Rank();
            size_t p = env.Size();

            auto shared = env.ShareValue<tT>(GetRandom<tT>());
            auto sharedValidation = env.ShareValue<tT>(GetRandom<tT>());
            tT unshared = GetRandom<tT>();

            sharedValidation.Get((s + offset) % p, shared.Value());
            env.Sync();
            shared.Get((s - offset + p) % p, unshared);
            env.Sync();

            EXPECT_EQ(unshared, sharedValidation.Value());
        }, offset);
    }
}

TEST(P(PutValue), uint8_t)
{
    TestPutValue<uint8_t>(4);
}

TEST(P(PutArrayValue), uint8_t)
{
    TestPutArrayValue<uint8_t>(4);
}

TEST(P(PutArray), uint8_t)
{
    TestPutArray<uint8_t>(4);
}

TEST(P(GetValue), uint8_t)
{
    TestGetValue<uint8_t>(4);
}

int main(int argc, char *argv[])
{
    testing::InitGoogleTest(&argc, argv);

    int result = RUN_ALL_TESTS();

    return result;
}