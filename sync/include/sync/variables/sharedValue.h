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
#ifndef __SYNCLIB_SHAREDVALUE_H__
#define __SYNCLIB_SHAREDVALUE_H__

#include "sync/variables/abstractSharedVariable.h"

#include "sync/util/variants.h"

namespace SyncLibInternal
{
    template <typename tT, typename tEnv>
    class SharedValue : public AbstractSharedVariable<tEnv>
    {
    public:
        using tParent = AbstractSharedVariable<tEnv>;
        using tDataTypeEnum = DataTypeEnum<tT>;
        using tEnvironmentType = tEnv;

        template <typename... tArgs>
        SharedValue(tEnv &env, tArgs &&... args)
            : tParent(env)
            , mValueHolder(std::forward<tArgs>(args)...)
            , mValue(GetVariantReference<tT>(mValueHolder))
        {
        }

        SharedValue &operator=(const tT &value)
        {
            mValue = value;
            return *this;
        }

        operator const tT &()
        {
            return mValue;
        }

        void Put(size_t target, const tT &value)
        {
            constexpr size_t requestSize = sizeof(DataType) + // DataType enum
                                           sizeof(size_t) +   // index
                                           sizeof(size_t) +   // size
                                           sizeof(tT);        // value

            char *cursor = tParent::GetTargetPutBuffer(target).Reserve(requestSize);
            tParent::WriteData(cursor, tDataTypeEnum::GetEnum(), tParent::mIndex, sizeof(tT), value);
        }

        void Put(size_t target)
        {
            Put(target, mValue);
        }

        void Broadcast(const tT &value)
        {
            for (size_t t = 0, endT = tParent::mEnv.Size(); t < endT; ++t)
            {
                Put(t, value);
            }
        }

        void Broadcast()
        {
            Broadcast(mValue);
        }

        void Get(size_t target, tT &destination)
        {
            constexpr size_t requestSize = sizeof(DataType) + // DataType enum
                                           sizeof(size_t) +   // index
                                           sizeof(size_t);    // size

            {
                char *cursor = tParent::GetTargetGetRequests(target).Reserve(requestSize);
                tParent::WriteData(cursor, tDataTypeEnum::GetEnum(), tParent::mIndex, sizeof(tT));
            }

            {
                char *cursor = tParent::GetTargetGetDestinations(target).Reserve(sizeof(tT *));
                tParent::WriteData(cursor, &destination);
            }

            tParent::AddGetBufferSize(target, sizeof(tT));
        }

        tT &Value()
        {
            return mValue;
        }

        void FinalisePut(const char *data) const
        {
            mValue = *reinterpret_cast<const tT *>(data);
        }

        void BufferGet(char *buffer) const
        {
            *reinterpret_cast<tT *>(buffer) = mValue;
        }

    private:
        std::variant<tT, tT *> mValueHolder;

        tT &mValue;
    };
} // namespace SyncLibInternal

#endif
