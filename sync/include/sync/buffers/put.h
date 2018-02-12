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
#ifndef __SYNCLIB_PUT_H__
#define __SYNCLIB_PUT_H__

#include <vector>

namespace SyncLib
{
    namespace Internal
    {
        class CommunicationBuffer
        {
        public:

            CommunicationBuffer(size_t initialSize = 1024 * 512)
                : mData(initialSize),
                  mCursor(0),
                  mSize(initialSize)
            {
            }

            char *Reserve(const size_t &size)
            {
                if (mSize - mCursor < size)
                {
                    Grow(size);
                }

                char *cursor = &mData[mCursor];
                mCursor += size;
                return cursor;
            }

            void Clear()
            {
                mCursor = 0;
            }

            const char *Begin() const
            {
                return &mData[0];
            }

            const char *End() const
            {
                return &mData[mCursor];
            }

        private:

            std::vector<char> mData;
            size_t mCursor;
            size_t mPutEnd;
            size_t mSendEnd;
            size_t mSize;

            void Grow(size_t size)
            {
                mSize = mSize * 3 / 2 + size;
                mData.resize(mSize);
            }
        };
    }
}

#endif