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
#ifndef __SYNCLIB_COMMUNICATIONHELPERS_H__
#define __SYNCLIB_COMMUNICATIONHELPERS_H__

#include "sync/variables/sharedArray.h"
#include "sync/variables/sharedValue.h"

namespace SyncLibInternal
{
    template <bool tIsArray = false>
    struct FinalisePutHelper
    {
        template <typename tT, typename tSharedVarialbe>
        static void Finalise(tSharedVarialbe &sharedVarialble, const char *cursor)
        {
            using tEnv = typename tSharedVarialbe::tEnvironmentType;
            reinterpret_cast<SharedValue<tT, tEnv> &>(sharedVarialble).FinalisePut(cursor);
        }
    };

    template <>
    struct FinalisePutHelper<true>
    {
        template <typename tT, typename tSharedVarialbe, typename... tArgs>
        static void Finalise(tSharedVarialbe &sharedVarialble, tArgs &&... args)
        {
            using tEnv = typename tSharedVarialbe::tEnvironmentType;
            reinterpret_cast<SharedArray<tT, tEnv> &>(sharedVarialble).FinalisePut(std::forward<tArgs>(args)...);
        }
    };

    template <bool tIsArray = false>
    struct BufferGetHelper
    {
        template <typename tT, typename tSharedVarialbe>
        static void Finalise(tSharedVarialbe &sharedVarialble, char *buffer)
        {
            using tEnv = typename tSharedVarialbe::tEnvironmentType;
            reinterpret_cast<SharedValue<tT, tEnv> &>(sharedVarialble).BufferGet(buffer);
        }
    };

    template <>
    struct BufferGetHelper<true>
    {
        template <typename tT, typename tSharedVarialbe, typename... tArgs>
        static void Finalise(tSharedVarialbe &sharedVarialble, tArgs &&... args)
        {
            using tEnv = typename tSharedVarialbe::tEnvironmentType;
            reinterpret_cast<SharedArray<tT, tEnv> &>(sharedVarialble).BufferGet(std::forward<tArgs>(args)...);
        }
    };

    template <bool tIsArray = false>
    struct FinaliseGetHelper
    {
        template <typename tT>
        static size_t Finalise(const char *buffer, const char *destinations)
        {
            tT &destination = **reinterpret_cast<tT *const *>(destinations);
            destination = *reinterpret_cast<const tT *>(buffer);
            return sizeof(tT *);
        }
    };

    template <>
    struct FinaliseGetHelper<true>
    {
        template <typename tT>
        static size_t Finalise(const char *buffer, const char *destinations, const size_t &size)
        {
            const size_t count = size / sizeof(tT);
            tT *destination = *reinterpret_cast<tT *const *>(destinations);
            const tT *bufferT = reinterpret_cast<const tT *>(buffer);

            std::copy_n(bufferT, count, destination);
            return sizeof(tT *);
        }
    };

    struct FinaliseHelper
    {
        template <typename tResult, typename tHelper, typename... tArgs>
        FORCEINLINE static tResult Finalise(const DataType &dataType, tArgs &&... args)
        {
            switch (dataType)
            {
            case DataType::U8:
                return tHelper::template Finalise<uint8_t>(std::forward<tArgs>(args)...);

            case DataType::U16:
                return tHelper::template Finalise<uint16_t>(std::forward<tArgs>(args)...);

            case DataType::U32:
                return tHelper::template Finalise<uint32_t>(std::forward<tArgs>(args)...);

            case DataType::U64:
                return tHelper::template Finalise<uint64_t>(std::forward<tArgs>(args)...);

            case DataType::S8:
                return tHelper::template Finalise<int8_t>(std::forward<tArgs>(args)...);

            case DataType::S16:
                return tHelper::template Finalise<int16_t>(std::forward<tArgs>(args)...);

            case DataType::S32:
                return tHelper::template Finalise<int32_t>(std::forward<tArgs>(args)...);

            case DataType::S64:
                return tHelper::template Finalise<int64_t>(std::forward<tArgs>(args)...);

            case DataType::Float:
                return tHelper::template Finalise<float>(std::forward<tArgs>(args)...);

            case DataType::Double:
                return tHelper::template Finalise<double>(std::forward<tArgs>(args)...);

            case DataType::Serialisable:
            case DataType::String:
            default:
                throw;
            }
        }
    };
} // namespace SyncLibInternal

#endif
