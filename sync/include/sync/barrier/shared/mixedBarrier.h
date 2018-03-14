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
#ifndef __SYNCLIB_MIXEDBARRIER_H__
#define __SYNCLIB_MIXEDBARRIER_H__

#include <stdint.h>
#include <assert.h>
#include <atomic>
#include <chrono>
#include <thread>
#include <mutex>

namespace SyncLib
{
    namespace Internal
    {
        template<size_t tMaxSpin>
        class MixedBarrier
        {
        public:

            explicit MixedBarrier(size_t count)
                : mCurrentCon(&mConVar1),
                  mPreviousCon(&mConVar2),
                  mCount(count),
                  mSpaces(count),
                  mGeneration(0)
            {
            }

            void Wait()
            {
                const size_t myGeneration = mGeneration;

                if (!--mSpaces)
                {
                    Reset();
                }
                else
                {
                    while (mGeneration == myGeneration)
                    {
                        for (size_t spin = 0; spin < tMaxSpin && mGeneration == myGeneration; ++spin);

                        if (mGeneration == myGeneration)
                        {
                            std::unique_lock<std::mutex> lock(mMutex);

                            if (mGeneration == myGeneration)
                            {
                                using namespace std::chrono_literals;
                                mCurrentCon->wait_for(lock, 10ns);
                            }
                        }
                    }
                }
            }

            void Resize(size_t count)
            {
                assert(mSpaces == mCount && "Count should be equal to Spaces on resize");

                mCount = count;
                mSpaces = count;
                mGeneration = 0;
            }

        private:

            std::mutex mMutex;
            std::condition_variable mConVar1;
            std::condition_variable mConVar2;

            std::condition_variable *mCurrentCon;
            std::condition_variable *mPreviousCon;

            std::atomic_size_t mSpaces;
            std::atomic_size_t mGeneration;
            size_t mCount;

            void Reset()
            {
                std::unique_lock<std::mutex> lock(mMutex);

                mSpaces = mCount;
                ++mGeneration;
                std::condition_variable *tmpCon = mCurrentCon;
                mCurrentCon = mPreviousCon;
                mPreviousCon = tmpCon;

                tmpCon->notify_all();
            }
        };
    }
}

#endif