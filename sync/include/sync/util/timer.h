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
#ifndef __SYNCLIB_TIMER_H__
#define __SYNCLIB_TIMER_H__

#include <chrono>

namespace SyncLib
{
    namespace Util
    {
        template<typename tResolution = std::chrono::seconds>
        class Timer
        {
        public:

            Timer()
                : mTic(Now())
            {}

            void Tic()
            {
                mTic = Now();
            }

            double Toc()
            {
                throw;
            }

            double TocTic()
            {
                double dur = Toc();
                Tic();
                return dur;
            }

        private:

            std::chrono::high_resolution_clock::time_point mTic;

            std::chrono::high_resolution_clock::time_point Now()
            {
                return std::chrono::high_resolution_clock::now();
            }
        };

        template<>
        inline double Timer<std::chrono::hours>::Toc()
        {
            return std::chrono::duration_cast<std::chrono::seconds>(Now() - mTic).count() / 3600.0;
        }

        template<>
        inline double Timer<std::chrono::minutes>::Toc()
        {
            return std::chrono::duration_cast<std::chrono::milliseconds>(Now() - mTic).count() / 3600.0e3;
        }

        template<>
        inline double Timer<std::chrono::seconds>::Toc()
        {
            return std::chrono::duration_cast<std::chrono::microseconds>(Now() - mTic).count() / 1.0e6;
        }

        template<>
        inline double Timer<std::chrono::milliseconds>::Toc()
        {
            return std::chrono::duration_cast<std::chrono::nanoseconds>(Now() - mTic).count() / 1.0e6;
        }

        template<>
        inline double Timer<std::chrono::microseconds>::Toc()
        {
            return std::chrono::duration_cast<std::chrono::nanoseconds>(Now() - mTic).count() / 1.0e3;
        }

        template<>
        inline double Timer<std::chrono::nanoseconds>::Toc()
        {
            return static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(Now() - mTic).count());
        }
    }
}

#endif