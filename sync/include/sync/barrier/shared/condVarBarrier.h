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
#ifndef __SYNCLIB_CONDVARBARRIER_H__
#define __SYNCLIB_CONDVARBARRIER_H__

#include <condition_variable>
#include <mutex>

namespace SyncLibInternal
{

    class CondVarBarrier
    {
    public:
        explicit CondVarBarrier(const size_t count)
            : mCurrentCon(&mConVar1)
            , mPreviousCon(&mConVar2)
            , mCount(count)
            , mMax(count)
        {
        }

        void Resize(const size_t count)
        {
            mCount = count;
            mMax = count;
        }

        void Wait()
        {
            std::unique_lock<std::mutex> lock(mMutex);

            if (--mCount == 0)
            {
                Reset();
            }
            else
            {
                mCurrentCon->wait(lock);
            }
        }

    private:
        std::mutex mMutex;
        std::condition_variable mConVar1;
        std::condition_variable mConVar2;

        std::condition_variable *mCurrentCon;
        std::condition_variable *mPreviousCon;

        size_t mCount;
        size_t mMax;

        void Reset()
        {
            mCount = mMax;
            std::condition_variable *tmpCon = mCurrentCon;
            mCurrentCon = mPreviousCon;
            mPreviousCon = tmpCon;

            tmpCon->notify_all();
        }
    };
} // namespace SyncLibInternal

#endif
