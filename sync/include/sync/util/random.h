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
#ifndef __SYNCLIB_RANDOM_H__
#define __SYNCLIB_RANDOM_H__

namespace SyncLib
{
    namespace Util
    {
        class LinearCongruentialRandom
        {
        public:
            LinearCongruentialRandom(const size_t seed, const size_t a = 1103515245, const size_t c = 12345)
                : mCurrent(seed)
                , mA(a)
                , mC(c)
            {
            }

            size_t Next()
            {
                mCurrent *= mA;
                mCurrent += mC;
                mCurrent &= 0x7FFFFFFF;

                return mCurrent >> 16;
            }

            double NextDouble()
            {
                double x = 1.0 * Next();
                x /= 0x7FFF;
                x -= 0.5;
                x *= 2;

                return x;
            }

            using result_type = size_t;

            constexpr static size_t max()
            {
                return 0x7FFF;
            };

            constexpr static size_t min()
            {
                return 0;
            };

            size_t operator()()
            {
                return Next();
            }

        private:
            size_t mCurrent;
            size_t mA, mC;
        };
    } // namespace Util
} // namespace SyncLib

#endif