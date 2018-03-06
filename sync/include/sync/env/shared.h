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
#ifndef __SYNCLIB_ENV_SHARED_H__
#define __SYNCLIB_ENV_SHARED_H__

#include "sync/barrier/shared/condVarBarrier.h"
#include "sync/barrier/shared/mixedBarrier.h"
#include "sync/barrier/shared/spinBarrier.h"

#include "sync/buffers/processor.h"

#include "sync/defines.h"

#include "sync/variables/sharedArray.h"
#include "sync/variables/sharedValue.h"

#include "sync/util/functionTraits.h"
#include "sync/util/algorithm.h"
#include "sync/util/dataType.h"
#include "sync/util/threads.h"

#include <type_traits>
#include <iostream>
#include <stdint.h>
#include <stdio.h>
#include <vector>

namespace SyncLib
{
    namespace Internal
    {
        template<bool tIsArray = false>
        struct FinalisePutHelper
        {
            template<typename tT, typename tSharedVarialbe>
            static void Finalise(tSharedVarialbe &sharedVarialble, const char *cursor)
            {
                using tEnv = typename tSharedVarialbe::tEnvironmentType;
                reinterpret_cast<SharedValue<tT, tEnv> &>(sharedVarialble).FinalisePut(cursor);
            }
        };

        template<>
        struct FinalisePutHelper<true>
        {
            template<typename tT, typename tSharedVarialbe, typename... tArgs>
            static void Finalise(tSharedVarialbe &sharedVarialble, tArgs &&... args)
            {
                using tEnv = typename tSharedVarialbe::tEnvironmentType;
                reinterpret_cast<SharedArray<tT, tEnv> &>(sharedVarialble).FinalisePut(std::forward<tArgs>(args)...);
            }
        };


        template<bool tIsArray = false>
        struct BufferGetHelper
        {
            template<typename tT, typename tSharedVarialbe>
            static void Finalise(tSharedVarialbe &sharedVarialble, char *buffer)
            {
                using tEnv = typename tSharedVarialbe::tEnvironmentType;
                reinterpret_cast<SharedValue<tT, tEnv> &>(sharedVarialble).BufferGet(buffer);
            }
        };

        template<>
        struct BufferGetHelper<true>
        {
            template<typename tT, typename tSharedVarialbe, typename... tArgs>
            static void Finalise(tSharedVarialbe &sharedVarialble, tArgs &&... args)
            {
                using tEnv = typename tSharedVarialbe::tEnvironmentType;
                reinterpret_cast<SharedArray<tT, tEnv> &>(sharedVarialble).BufferGet(std::forward<tArgs>(args)...);
            }
        };

        template<bool tIsArray = false>
        struct FinaliseGetHelper
        {
            template<typename tT>
            static size_t Finalise(const char *buffer, const char *destinations)
            {
                tT &destination = **reinterpret_cast<tT *const *>(destinations);
                destination = *reinterpret_cast<const tT *>(buffer);
                return sizeof(tT *);
            }
        };

        template<typename tResult>
        struct RunZero
        {
            template<typename tLambda, typename tEnv>
            static tResult Run(tEnv &env, std::vector<std::thread> &threads, const tLambda &lambda)
            {
                tResult res = lambda(0);

                env.JoinThreads(threads);

                return res;
            }
        };

        template<>
        struct RunZero<void>
        {
            template<typename tLambda, typename tEnv>
            static void Run(tEnv &env, std::vector<std::thread> &threads, const tLambda &lambda)
            {
                lambda(0);
                env.JoinThreads(threads);
            }
        };
    }

    namespace Environments
    {
        class SharedMemoryBSP
        {
        public:

            template<typename tEnv>
            friend class Internal::AbstractSharedVariable;
            template<typename tResult>
            friend struct Internal::RunZero;

            SharedMemoryBSP(size_t size = 4)
                : mBarrier(size),
                  mSize(size)
            {
            }

            template<typename tFunc, typename... tArgs>
            auto Run(size_t p, const tFunc &func, tArgs &&... args)
            {
                auto lambda = [&](size_t s)
                {
                    GetRank(&s);
                    //Internal::PinThread(s);

                    return func(*this, std::forward<tArgs>(args)...);
                };

                std::vector<std::thread> threads;
                threads.reserve(p - 1);
                mProcessorBuffers.clear();
                mProcessorBuffers.resize(p);

                for (auto &buffer : mProcessorBuffers)
                {
                    for (auto &requests : buffer.requests)
                    {
                        requests.putBuffer.Clear();
                        requests.getBuffer.Clear();
                        requests.getRequests.Clear();
                        requests.getDestinations.Clear();
                    }

                    buffer.requests.resize(p);
                }

                mBarrier.Resize(p);
                mSize = p;

                for (size_t s : SyncLib::Ranges::Range<size_t>(1, p))
                {
                    threads.emplace_back(lambda, s);
                }

                using tResult = typename SyncLib::Internal::FunctionTraits<decltype(lambda)>::result::type;

                return Internal::RunZero<tResult>::Run(*this, threads, lambda);
            }

            void Sync()
            {
                const size_t s = Rank();

                mBarrier.Wait();

                Util::SplitFor<size_t>(0, s, mSize, [&](size_t t)
                {
                    BufferGet(t, s);
                });

                mBarrier.Wait();

                Util::SplitFor<size_t>(0, s, mSize, [&](size_t t)
                {
                    ProcessPut(t, s);
                });

                Util::SplitFor<size_t>(0, s, mSize, [&](size_t t)
                {
                    ProcessGet(t, s);
                });

                mBarrier.Wait();

            }

            inline void ProcessPut(const size_t &t, const size_t &s)
            {
                auto &putFromT = mProcessorBuffers[t].requests[s].putBuffer;

                for (const char *cursor = putFromT.Begin(), *end = putFromT.End(); cursor < end;)
                {
                    const Internal::DataType dataType = static_cast<Internal::DataType>(*cursor++);
                    const Internal::DataType isArray = dataType & Internal::DataType::Array;

                    const size_t *sCursor = reinterpret_cast<const size_t *>(cursor);
                    const size_t index = *sCursor++;
                    const size_t size = *sCursor++;

                    Internal::AbstractSharedVariable<SharedMemoryBSP> &sharedVariable = *mProcessorBuffers[s].variables[index];

                    if (isArray == Internal::DataType::Array)
                    {
                        const size_t offset = *sCursor++;
                        cursor += 3 * sizeof(size_t);
                        Finalise<void, Internal::FinalisePutHelper<true>>(dataType ^ isArray, sharedVariable, cursor, offset, size);
                    }
                    else
                    {
                        cursor += 2 * sizeof(size_t);
                        Finalise<void, Internal::FinalisePutHelper<false>>(dataType ^ isArray, sharedVariable, cursor);
                    }

                    cursor += size;
                }

                putFromT.Clear();
            }

            inline void BufferGet(const size_t &t, const size_t &s)
            {
                // We've received get requests from t
                auto &getFromT = mProcessorBuffers[t].requests[s].getRequests;
                // We buffer them locally
                auto &getBuffer = mProcessorBuffers[s].requests[t].getBuffer;
                // Cleanup the previous synchronization
                getBuffer.Clear();

                for (const char *cursor = getFromT.Begin(), *end = getFromT.End(); cursor < end;)
                {
                    const Internal::DataType dataType = static_cast<Internal::DataType>(*cursor++);
                    const Internal::DataType isArray = dataType & Internal::DataType::Array;

                    const size_t *sCursor = reinterpret_cast<const size_t *>(cursor);
                    const size_t index = *sCursor++;
                    const size_t size = *sCursor++;

                    Internal::AbstractSharedVariable<SharedMemoryBSP> &sharedVariable = *mProcessorBuffers[s].variables[index];

                    if (isArray == Internal::DataType::Array)
                    {
                        const size_t offset = *sCursor++;
                        cursor += 3 * sizeof(size_t);
                        offset;
                        throw;
                    }
                    else
                    {
                        cursor += 2 * sizeof(size_t);
                        Finalise<void, Internal::BufferGetHelper<false>>(dataType ^ isArray, sharedVariable, getBuffer.Reserve(size));
                    }
                }
            }

            inline void ProcessGet(const size_t &t, const size_t &s)
            {
                // We want to get data from processor t
                auto &getFromT = mProcessorBuffers[s].requests[t].getRequests;
                // We've received their buffer
                auto &getBuffer = mProcessorBuffers[t].requests[s].getBuffer;
                // We want to store it in our local destination
                auto &getDestinations = mProcessorBuffers[s].requests[t].getDestinations;

                const char *bufferCursor = getBuffer.Begin();
                const char *destinationCursor = getDestinations.Begin();

                for (const char *cursor = getFromT.Begin(), *end = getFromT.End(); cursor < end;)
                {
                    const Internal::DataType dataType = static_cast<Internal::DataType>(*cursor++);
                    const Internal::DataType isArray = dataType & Internal::DataType::Array;

                    const size_t *sCursor = reinterpret_cast<const size_t *>(cursor);
                    // Skip index
                    ++sCursor;
                    const size_t size = *sCursor++;

                    if (isArray == Internal::DataType::Array)
                    {
                        const size_t offset = *sCursor++;
                        cursor += 3 * sizeof(size_t);
                        offset;
                        throw;
                    }
                    else
                    {
                        cursor += 2 * sizeof(size_t);
                        destinationCursor += Finalise<size_t, Internal::FinaliseGetHelper<false>>(dataType ^ isArray, bufferCursor, destinationCursor);
                        bufferCursor += size;
                    }
                }

                getFromT.Clear();
                getDestinations.Clear();
            }

            size_t Rank()
            {
                return GetRank();
            }

            size_t Size()
            {
                return mSize;
            }

            size_t MaxSize()
            {
                return std::thread::hardware_concurrency();
            }

            template<typename tT, typename... tArgs>
            Internal::SharedArray<tT, SharedMemoryBSP> ShareArray(tArgs &&... args)
            {
                return Internal::SharedArray<tT, SharedMemoryBSP>(*this, std::forward<tArgs>(args)...);
            }

            template<typename tT, typename... tArgs>
            Internal::SharedValue<tT, SharedMemoryBSP> ShareValue(tArgs &&... args)
            {
                return Internal::SharedValue<tT, SharedMemoryBSP>(*this, std::forward<tArgs>(args)...);
            }

        private:

            Internal::SpinBarrier mBarrier;
            // Internal::MixedBarrier<50000> mBarrier;
            // Internal::CondVarBarrier mBarrier;
            size_t mSize;

            std::vector<Internal::ProcessorBuffers<SharedMemoryBSP>> mProcessorBuffers;

            size_t GetRank(size_t *rank = nullptr)
            {
                thread_local static size_t s = 0;

                if (rank)
                {
                    s = *rank;
                }

                return s;
            }

            template<typename tResult, typename tHelper, typename... tArgs>
            SYNCLIB_FORCEINLINE tResult Finalise(const Internal::DataType &dataType, tArgs &&...args)
            {
                switch (dataType)
                {
                case Internal::DataType::U8:
                    return tHelper::template Finalise<uint8_t>(std::forward<tArgs>(args)...);

                case Internal::DataType::U16:
                    return tHelper::template Finalise<uint16_t>(std::forward<tArgs>(args)...);

                case Internal::DataType::U32:
                    return tHelper::template Finalise<uint32_t>(std::forward<tArgs>(args)...);

                case Internal::DataType::U64:
                    return tHelper::template Finalise<uint64_t>(std::forward<tArgs>(args)...);

                case Internal::DataType::S8:
                    return tHelper::template Finalise<int8_t>(std::forward<tArgs>(args)...);

                case Internal::DataType::S16:
                    return tHelper::template Finalise<int16_t>(std::forward<tArgs>(args)...);

                case Internal::DataType::S32:
                    return tHelper::template Finalise<int32_t>(std::forward<tArgs>(args)...);

                case Internal::DataType::S64:
                    return tHelper::template Finalise<int64_t>(std::forward<tArgs>(args)...);

                case Internal::DataType::Float:
                    return tHelper::template Finalise<float>(std::forward<tArgs>(args)...);

                case Internal::DataType::Double:
                    return tHelper::template Finalise<double>(std::forward<tArgs>(args)...);

                case Internal::DataType::Serialisable:
                case Internal::DataType::String:
                default:
                    throw;
                    break;
                }
            }

            size_t Register(Internal::AbstractSharedVariable<SharedMemoryBSP> &sharedVariable)
            {
                auto &variables = mProcessorBuffers[Rank()].variables;
                size_t index = variables.size();
                variables.push_back(&sharedVariable);
                return index;
            }

            Internal::CommunicationBuffer &GetTargetPutBuffer(size_t target)
            {
                return mProcessorBuffers[Rank()].requests[target].putBuffer;
            }

            Internal::CommunicationBuffer &GetTargetGetBuffer(size_t target)
            {
                return mProcessorBuffers[Rank()].requests[target].getBuffer;
            }

            Internal::CommunicationBuffer &GetTargetGetRequests(size_t target)
            {
                return mProcessorBuffers[Rank()].requests[target].getRequests;
            }

            Internal::CommunicationBuffer &GetTargetGetDestinations(size_t target)
            {
                return mProcessorBuffers[Rank()].requests[target].getDestinations;
            }

            static void JoinThreads(std::vector<std::thread> &threads)
            {
                for (auto &t : threads)
                {
                    t.join();
                }
            }
        };
    }
}
#endif // !__SYNCLIB_ENV_H__
