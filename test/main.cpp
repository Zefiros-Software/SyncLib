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
#include "sync/variables/abstractSharedVariable.h"
#include "sync/util/ranges/range.h"
#include "sync/util/variants.h"
#include "sync/env/shared.h"

#include "helper.h"

#include "fmt/format.h"

#include <stdint.h>
#include <variant>
#include <numeric>
#include <thread>
#include <tuple>
#include "sync/buffers/sendQueue.h"

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

// TEST_INTEGER_TYPES(TEST_RANGE);
// TEST_INTEGER_TYPES(TEST_RANGE_REVERSE);
// TEST_INTEGER_TYPES(TEST_RANGE_WITH_LOWER);
// TEST_INTEGER_TYPES(TEST_RANGE_WITH_LOWER_REVERSE);
// TEST_INTEGER_TYPES(TEST_STEP_RANGE);

template<typename tT>
class SyncLibTypedTest
    : public ::testing::Test
{
public:

    static void SetUpTestCase()
    {
    }

    static void TearDownTestCase()
    {
    }

    //protected:

    SyncLibTypedTest()
        : mP(std::thread::hardware_concurrency() / 2),
          mEnv(mP)
    {
    }

    size_t mP;

    using tType = tT;
    using tEnv = SyncLib::Environments::SharedMemoryBSP;
    tEnv mEnv;
    tT mValueUnshared;
};

template<typename tT>
void TestPutValue(size_t p)
{
    using tEnv = SyncLib::Environments::SharedMemoryBSP;

    for (size_t offset : SyncLib::Ranges::Range(p))
    {
        tEnv env;
        env.Run(p, [&](tEnv & env, const size_t offset)
        {
            size_t s = env.Rank();
            size_t p = env.Size();

            tT unshared = GetRandom<tT>();
            tEnv::SharedValue<tT> shared(env, GetRandom<tT>());
            tEnv::SharedValue<tT> sharedValidation(env, GetRandom<tT>());

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
            auto shared = tEnv::SharedArray<tT>(env, p);
            auto sharedValidation = tEnv::SharedArray<tT>(env, p);

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
            auto shared = tEnv::SharedArray<tT>(env, p);
            auto sharedValidation = tEnv::SharedArray<tT>(env, p);

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

            auto shared = tEnv::SharedValue<tT>(env, GetRandom<tT>());
            auto sharedValidation = tEnv::SharedValue<tT>(env, GetRandom<tT>());
            tT unshared = GetRandom<tT>();

            sharedValidation.Get((s + offset) % p, shared.Value());
            env.Sync();
            shared.Get((s - offset + p) % p, unshared);
            env.Sync();

            EXPECT_EQ(unshared, sharedValidation.Value());
        }, offset);
    }
}

template<typename tT>
void TestGetArrayValue(size_t p)
{
    using tEnv = SyncLib::Environments::SharedMemoryBSP;

    for (size_t offset : SyncLib::Ranges::Range(p))
    {
        tEnv env;

        env.Run(p, [](tEnv & env, const size_t offset)
        {
            size_t s = env.Rank();
            size_t p = env.Size();

            auto shared = tEnv::SharedArray<tT>(env, p);
            auto sharedValidation = tEnv::SharedArray<tT>(env, p);
            std::vector<tT> unshared(p);

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
                    sharedValidation.GetValue(t, shared[i], i);
                }
            }

            env.Sync();

            {
                size_t t = (s - offset + p) % p;

                for (size_t i : SyncLib::Ranges::Range(p))
                {
                    shared.GetValue(t, unshared[i], i);
                }
            }

            env.Sync();

            EXPECT_EQ(unshared, sharedValidation.Value());
        }, offset);
    }
}

template<typename tT>
void TestSendValueSingle(size_t p)
{
    using tEnv = SyncLib::Environments::SharedMemoryBSP;

    for (size_t offset : SyncLib::Ranges::Range(p))
    {
        tEnv env;

        env.Run(p, [](tEnv & env, const size_t offset)
        {
            size_t s = env.Rank();
            size_t p = env.Size();

            tT initial = GetRandom<tT>();
            tT received = GetRandom<tT>();
            tT receivedValidation = GetRandom<tT>();
            tEnv::SendQueue<tT> queue(env);

            queue.Send((s + offset) % p, initial);
            env.Sync();

            std::tie(received) = *queue.begin();
            queue.Send((s - offset + p) % p, received);
            env.Sync();

            std::tie(receivedValidation) = *queue.begin();

            EXPECT_EQ(initial, receivedValidation);
        }, offset);
    }
}

template<typename tT>
void TestSendValueAll(size_t p)
{
    using tEnv = SyncLib::Environments::SharedMemoryBSP;

    tEnv env;

    env.Run(p, [](tEnv & env)
    {
        size_t s = env.Rank();
        size_t p = env.Size();

        std::vector<tT> initial(p);
        std::vector<tT> received(p);
        std::vector<tT> receivedValidation(p);

        for (auto &x : initial)
        {
            x = GetRandom<tT>();
        }

        tEnv::SendQueue<size_t, tT> queue(env);

        for (size_t t : SyncLib::Ranges::Range(p))
        {
            queue.Send(t, s, initial[t]);
        }

        env.Sync();

        for (auto[t, rec] : queue)
        {
            received[t] = rec;
        }

        for (size_t t : SyncLib::Ranges::Range(p))
        {
            queue.Send(t, s, received[t]);
        }

        env.Sync();

        for (auto[t, rec] : queue)
        {
            receivedValidation[t] = rec;
        }

        EXPECT_EQ(initial, receivedValidation);
    });
}

template<typename tT>
void TestGetArray(size_t p)
{
    using tEnv = SyncLib::Environments::SharedMemoryBSP;

    for (size_t offset : SyncLib::Ranges::Range(p))
    {
        tEnv env;

        env.Run(p, [](tEnv & env, const size_t offset)
        {
            size_t s = env.Rank();
            size_t p = env.Size();

            auto shared = tEnv::SharedArray<tT>(env, p);
            auto sharedValidation = tEnv::SharedArray<tT>(env, p);
            std::vector<tT> unshared(p);

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
                    sharedValidation.Get(t, shared.begin() + i, shared.begin() + i + 1, i);
                }
            }

            env.Sync();

            {
                size_t t = (s - offset + p) % p;

                for (size_t i : SyncLib::Ranges::Range(p))
                {
                    shared.Get(t, unshared.begin() + i, unshared.begin() + i + 1, i);
                }
            }

            env.Sync();

            EXPECT_EQ(unshared, sharedValidation.Value());
        }, offset);
    }
}



using NumericTypes
    =  ::testing::Types<uint8_t, uint16_t, uint32_t, uint64_t, size_t, int8_t, int16_t, int32_t, int64_t, std::make_signed_t<size_t>, float, double>;
TYPED_TEST_CASE(SyncLibTypedTest, NumericTypes);

TYPED_TEST(SyncLibTypedTest, PutValue)
{
    TestPutValue<typename TestFixture::tType>(TestFixture::mP);
};

TYPED_TEST(SyncLibTypedTest, GetValue)
{
    TestGetValue<typename TestFixture::tType>(TestFixture::mP);
};

TYPED_TEST(SyncLibTypedTest, PutArrayValue)
{
    TestPutArrayValue<typename TestFixture::tType>(TestFixture::mP);
};

TYPED_TEST(SyncLibTypedTest, GetArrayValue)
{
    TestGetArrayValue<typename TestFixture::tType>(TestFixture::mP);
};

TYPED_TEST(SyncLibTypedTest, PutArray)
{
    TestPutArray<typename TestFixture::tType>(TestFixture::mP);
};

TYPED_TEST(SyncLibTypedTest, GetArray)
{
    TestGetArray<typename TestFixture::tType>(TestFixture::mP);
};

TYPED_TEST(SyncLibTypedTest, SendValueSingle)
{
    TestSendValueSingle<typename TestFixture::tType>(TestFixture::mP);
};

TYPED_TEST(SyncLibTypedTest, SendValueAll)
{
    TestSendValueAll<typename TestFixture::tType>(TestFixture::mP);
};

int main(int argc, char *argv[])
{
    //     fmt::print("Size of ValueVariants: {}\n", sizeof(ValueVariants));
    //     fmt::print("Size of VectorVariants: {}\n", sizeof(VectorVariants));
    //     fmt::print("Size of AllVariants: {}\n", sizeof(AllVariants));
    //     fmt::print("Size of std::vector: {}\n", sizeof(std::vector<size_t>));
    //     fmt::print("Size of SharedValue: {}\n",
    //                sizeof(SyncLib::Internal::SharedArray<uint64_t, SyncLib::Environments::SharedMemoryBSP>));
    //
    //
    //     fmt::print("Count in ValueVariants: {}\n", std::variant_size_v<ValueVariants>);
    //     fmt::print("Count in VectorVariants: {}\n", std::variant_size_v<VectorVariants>);
    //     fmt::print("Count in AllVariants: {}\n", std::variant_size_v<AllVariants>);
    //
    //     size_t z = 1ull;
    //     AllVariants x = z;
    //     AllVariants y = &z;
    //
    //     if (std::holds_alternative<uint64_t>(x))
    //     {
    //         fmt::print("x holds U64 with value {}\n", std::get<uint64_t>(x));
    //     }
    //     else if (std::holds_alternative<uint32_t>(x))
    //     {
    //         fmt::print("x holds U32 with value {}\n", std::get<uint32_t>(x));
    //     }
    //
    //     if (HoldsAlternative<uint64_t>(y))
    //     {
    //         auto &yRef = GetVariantReference<uint64_t>(y);
    //         fmt::print("y holds U64 with value {}\n", yRef);
    //         yRef = 2;
    //         fmt::print("y holds U64 with value {}, original now holds {}\n", yRef, z);
    //     }
    //     else if (HoldsAlternative<uint32_t>(y))
    //     {
    //         auto &yRef = GetVariantReference<uint32_t>(y);
    //         fmt::print("y holds U32 with value {}\n", yRef);
    //         yRef = 2;
    //         fmt::print("y holds U32 with value {}, original now holds {}\n", yRef, z);
    //     }
    //
    //     using tEnv = SyncLib::Environments::SharedMemoryBSP;
    //     tEnv env(2);
    //     env.Run(2, [](tEnv & env)
    //     {
    //         size_t s = env.Rank();
    //         tEnv::SendQueue<char, size_t> queue(env);
    //         queue.Send(1 - s, static_cast<char>('a' + s), s);
    //         queue.Send(1 - s, static_cast<char>('a' + s), s);
    //         env.Sync();
    //
    //         for (auto [u, v] : queue)
    //         {
    //             fmt::print("s: {}, u: {}, v: {}\n", s, u, v);
    //         }
    //     });
    //
    //     fmt::print("Done\n");
    //
    //     system("pause");
    //     exit(EXIT_SUCCESS);

    testing::InitGoogleTest(&argc, argv);

    int result = RUN_ALL_TESTS();

    return result;
}