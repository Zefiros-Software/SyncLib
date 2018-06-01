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
#ifndef __SYNCLIB_SHAREDARRAY_H__
#define __SYNCLIB_SHAREDARRAY_H__

#include "sync/variables/abstractSharedVariable.h"

#include <algorithm>
#include <vector>

namespace SyncLibInternal
{
    template <typename tT, typename tEnv>
    class SharedArray : public AbstractSharedVariable<tEnv>
    {
    public:
        using tParent = AbstractSharedVariable<tEnv>;
        using iterator = typename std::vector<tT>::iterator;
        using tDataTypeEnum = DataTypeEnum<std::vector<tT>>;
        using tEnvironmentType = tEnv;

        template <typename... tArgs>
        SharedArray(tEnv &env, tArgs &&... args)
            : tParent(env)
            , mValues(std::forward<tArgs>(args)...)
        {
        }

        void PutValue(size_t target, const tT &value, size_t offset)
        {
            constexpr size_t requestSize = sizeof(DataType) + // DataType enum
                                           sizeof(size_t) +   // index
                                           sizeof(size_t) +   // size
                                           sizeof(size_t) +   // offset
                                           sizeof(tT);        // value

            char *cursor = tParent::GetTargetPutBuffer(target).Reserve(requestSize);
            tParent::WriteData(cursor, tDataTypeEnum::GetEnum(), tParent::mIndex, sizeof(tT), offset, value);
        }

        void GetValue(size_t target, tT &destination, size_t offset = 0)
        {
            constexpr size_t requestSize = sizeof(DataType) + // DataType enum
                                           sizeof(size_t) +   // index
                                           sizeof(size_t) +   // size
                                           sizeof(size_t);    // offset

            {
                char *cursor = tParent::GetTargetGetRequests(target).Reserve(requestSize);
                tParent::WriteData(cursor, tDataTypeEnum::GetEnum(), tParent::mIndex, sizeof(tT), offset);
            }

            {
                char *cursor = tParent::GetTargetGetDestinations(target).Reserve(sizeof(tT *));
                tParent::WriteData(cursor, &destination);
            }

            tParent::AddGetBufferSize(target, sizeof(tT));
        }

        void BroadcastValue(const tT &value, const size_t offset)
        {
            for (size_t target = 0, end = tParent::mEnv.Size(); target < end; ++target)
            {
                PutValue(target, value, offset);
            }
        }

        void BroadcastValue(const size_t offset)
        {
            BroadcastValue(mValues[offset], offset);
        }

        template <typename tIterator>
        void Put(size_t target, tIterator beginIt, tIterator endIt, size_t offset)
        {
            const size_t count = endIt - beginIt;
            size_t requestSize = sizeof(DataType) +  // DataType enum
                                 sizeof(size_t) +    // index
                                 sizeof(size_t) +    // offset
                                 sizeof(size_t) +    // size
                                 sizeof(tT) * count; // values

            char *cursor = tParent::GetTargetPutBuffer(target).Reserve(requestSize);
            cursor = tParent::WriteData(cursor, tDataTypeEnum::GetEnum(), tParent::mIndex, sizeof(tT) * count, offset);
            tParent::WriteIter(cursor, beginIt, endIt);
        }

        template <typename tDestIterator>
        void Get(size_t target, tDestIterator beginIt, tDestIterator endIt, size_t offset)
        {
            const size_t count = endIt - beginIt;
            size_t size = sizeof(tT) * count;
            constexpr size_t requestSize = sizeof(DataType) + // DataType enum
                                           sizeof(size_t) +   // index
                                           sizeof(size_t) +   // size
                                           sizeof(size_t);    // offset

            {
                char *cursor = tParent::GetTargetGetRequests(target).Reserve(requestSize);
                tParent::WriteData(cursor, tDataTypeEnum::GetEnum(), tParent::mIndex, size, offset);
            }

            {
                char *cursor = tParent::GetTargetGetDestinations(target).Reserve(sizeof(tT *));
                tParent::WriteData(cursor, &(*beginIt));
            }

            tParent::AddGetBufferSize(target, size);
        }

        void Broadcast()
        {
            for (size_t t : Range(tParent::mEnv.Size()))
            {
                Put(t, mValues.begin(), mValues.end(), 0);
            }
        }

        std::vector<tT> &Value()
        {
            return mValues;
        }

        void FinalisePut(const char *data, const size_t &offset, const size_t &size)
        {
            const size_t count = size / sizeof(tT);
            const tT *cursorT = reinterpret_cast<const tT *>(data);

            std::copy_n(cursorT, count, mValues.begin() + offset);
        }

        void BufferGet(char *cursor, const size_t &offset, const size_t &size)
        {
            const size_t count = size / sizeof(tT);
            tT *cursorT = reinterpret_cast<tT *>(cursor);
            tT *endCursor = std::copy_n(mValues.begin() + offset, count, cursorT);
            Assert((endCursor - cursorT) * sizeof(tT) == size, "Written beyond reserved size");
        }

        iterator begin()
        {
            return mValues.begin();
        }

        iterator end()
        {
            return mValues.end();
        }

        tT &operator[](const size_t index)
        {
            return mValues[index];
        }

    private:
        std::vector<tT> mValues;
    };
} // namespace SyncLibInternal

#endif
