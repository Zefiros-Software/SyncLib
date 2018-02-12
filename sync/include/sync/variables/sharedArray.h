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

#include "sync/util/ranges/range.h"

#include "sync/variables/abstractSharedVariable.h"

#include <algorithm>
#include <stdint.h>
#include <vector>

namespace SyncLib
{
    namespace Internal
    {
        template<typename tT, typename tEnv>
        class SharedArray
            : public AbstractSharedVariable<tEnv>
        {
        public:

            using tParent = AbstractSharedVariable<tEnv>;
            using iterator = typename std::vector<tT>::iterator;
            using tDataTypeEnum = DataTypeEnum<std::vector<tT>>;
            using tEnvironmentType = tEnv;

            template<typename... tArgs>
            SharedArray(tEnv &env, tArgs &&... args)
                : tParent(env),
                  mValues(std::forward<tArgs>(args)...)
            {}

            void PutValue(size_t target, const tT &value, size_t offset)
            {
                constexpr size_t requestSize = sizeof(DataType) +   // DataType enum
                                               sizeof(size_t) +     // index
                                               sizeof(size_t) +     // offset
                                               sizeof(size_t) +     // size
                                               sizeof(tT);          // value

                char *cursor = tParent::GetTargetPutBuffer(target).Reserve(requestSize);
                tParent::WriteData(cursor, tDataTypeEnum::value, tParent::mIndex, sizeof(tT), offset, value);
            }

            void BroadcastValue(const tT &value, size_t offset)
            {
                for (size_t target : Range(tParent::mEnv.Size()))
                {
                    PutValue(target, value, offset);
                }
            }

            void BroadcastValue(size_t offset)
            {
                BroadcastValue(mValues[offset], offset);
            }

            template<typename tIterator>
            void Put(size_t target, tIterator beginIt, tIterator endIt, size_t offset)
            {
                size_t count = endIt - beginIt;
                size_t requestSize = sizeof(DataType) +     // DataType enum
                                     sizeof(size_t) +       // index
                                     sizeof(size_t) +       // offset
                                     sizeof(size_t) +       // size
                                     sizeof(tT) * count;    // values

                char *cursor = tParent::GetTargetPutBuffer(target).Reserve(requestSize);
                cursor = tParent::WriteData(cursor, tDataTypeEnum::value, tParent::mIndex, sizeof(tT) * count, offset);
                tParent::WriteIter(cursor, beginIt, endIt);
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

            iterator begin()
            {
                return mValues.begin();
            }

            iterator end()
            {
                return mValues.end();
            }

            tT &operator[](size_t index)
            {
                return mValues[index];
            }

        private:

            std::vector<tT> mValues;
        };
    }
}

#endif