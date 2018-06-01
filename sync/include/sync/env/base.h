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
#ifndef __SYNCLIB_DISTRIBUTED_H__
#define __SYNCLIB_DISTRIBUTED_H__

#include "sync/variables/communicationHelpers.h"
#include "sync/variables/sharedArray.h"
#include "sync/variables/sharedValue.h"

#include "sync/bindings/mpi.h"
#include "sync/buffers/processor.h"
#include "sync/buffers/sendQueue.h"
#include "sync/util/algorithm.h"

namespace SyncLib
{
    namespace Environments
    {
        template <typename tBackend>
        class BaseBSP
        {
        public:
            template <typename>
            friend class SyncLibInternal::AbstractSharedVariable;

            friend struct SyncLibInternal::EnvHelper;

            using tEnv = BaseBSP<tBackend>;
            using tBuffer = SyncLibInternal::DistributedCommunicationBuffer;
            using tDataType = SyncLibInternal::DataType;
            using tBackendType = tBackend;

            template <typename... tArgs>
            using SendQueue = SyncLibInternal::SendQueue<tEnv, tArgs...>;

            template <typename tT>
            using SharedValue = SyncLibInternal::SharedValue<tT, tEnv>;

            template <typename tT>
            using SharedArray = SyncLibInternal::SharedArray<tT, tEnv>;

            using AbstractSendQueue = SyncLibInternal::AbstractSendQueue<tEnv>;
            using AbstractSharedVariable = SyncLibInternal::AbstractSharedVariable<tEnv>;

            template <typename... tArgs>
            BaseBSP(tArgs &&... args)
                : mBackend(std::forward<tArgs>(args)...)
                , mGetRequests(mBackend.Size())
                , mGetBuffers(mBackend.Size())
                , mPutBuffers(mBackend.Size())
                , mGetDestinations(mBackend.Size())
            {
            }

            template <typename tFunc, typename... tArgs>
            inline auto Run(const tFunc &func, tArgs &&... args)
            {
                return mBackend.Run(*this, func, std::forward<tArgs>(args)...);
            }

            size_t Rank() const
            {
                return mBackend.Rank();
            }

            size_t Size() const
            {
                return mBackend.Size();
            }

            void Barrier()
            {
                mBackend.Barrier();
            }

            static size_t MaxSize()
            {
                return tBackend::MaxSize();
            }

            static size_t MaxVirtualSize()
            {
                return tBackend::MaxVirtualSize();
            }

            tEnv &Split(size_t part, size_t rank)
            {
                return mBackend.Split(part, rank);
            }

            void UnSplit(tEnv &env)
            {
                mBackend.UnSplit(env);
            }

            void Resize(size_t size)
            {
                mBackend.Resize(size);
            }

            void Sync()
            {
                // const size_t s = Rank();
                const size_t p = Size();

                for (auto queue : mSendQueues)
                {
                    if (queue)
                    {
                        queue->ClearReceiveBuffer();
                    }
                }

                for (size_t t = 0; t < Size(); ++t)
                {
                    mGetRequests.sendSizes[t] = static_cast<int>(mGetRequests.sendBuffers[t].Size());
                    mPutBuffers.sendSizes[t] = static_cast<int>(mPutBuffers.sendBuffers[t].Size());
                }

                size_t sqCount = mSendQueues.size();

                mSendQueueInfo.Clear();
                {
                    for (size_t i = 0; i < sqCount; ++i)
                    {
                        const size_t sqOffset = i * p;

                        if (mSendQueues[i])
                        {
                            const char *sqBufferBegin = mSendQueues[i]->GetTargetBuffer(0).Begin();

                            for (size_t t = 0; t < p; ++t)
                            {
                                const char *sqBuffer = mSendQueues[i]->GetTargetBuffer(t).Begin();

                                mSendQueueInfo.sendSizes[sqOffset + t] = static_cast<int>(mSendQueues[i]->GetTargetSize(t));
                                mSendQueueInfo.sendDisplacements[sqOffset + t] = static_cast<int>(sqBuffer - sqBufferBegin);
                                mSendQueueInfo.sendCounts[sqOffset + t] = static_cast<int>(mSendQueues[i]->GetTargetCount(t));
                            }
                        }
                    }
                }
                mBackend.ExchangeSizes(*this);

                mBackend.SynchroniseGetRequestsPutBuffersSendQueues(*this);

                mBackend.WaitForGetRequests();
                mGetBuffers.ClearSendBuffers();

                for (size_t t = 0; t < p; ++t)
                {
                    BufferGet(t);
                }

                mBackend.SynchroniseGetBuffers(*this);
                mBackend.WaitForPutBuffers();

                for (size_t t = 0; t < p; ++t)
                {
                    ProcessPut(t);
                }

                mBackend.WaitForGetBuffers();

                for (size_t t = 0; t < p; ++t)
                {
                    ProcessGet(t);
                }

                mBackend.WaitForSendQueues(sqCount);

                mGetRequests.ClearSendBuffers();
                mPutBuffers.ClearSendBuffers();

                for (size_t t = 0; t < p; ++t)
                {
                    mGetDestinations[t].Clear();
                }

                for (auto queue : mSendQueues)
                {
                    if (queue)
                    {
                        queue->ClearSendBuffers();
                    }
                }
            }

            template <typename... tArgs>
            void Print(tArgs &&... args)
            {
                mBackend.Print(std::forward<tArgs>(args)...);
            }

        private:
            tBackend mBackend;

            struct
            {
                std::vector<int> sendSizes;
                std::vector<int> sendDisplacements;
                std::vector<int> sendCounts;
                std::vector<int> receiveSizes;
                std::vector<int> receiveDisplacements;
                std::vector<int> receiveCounts;

                void Grow(size_t p)
                {
                    Grow(p, sendSizes);
                    Grow(p, sendDisplacements);
                    Grow(p, sendCounts);
                    Grow(p, receiveSizes);
                    Grow(p, receiveDisplacements);
                    Grow(p, receiveCounts);
                }

                void Clear()
                {
                    std::fill(sendSizes.begin(), sendSizes.end(), 0);
                    std::fill(sendDisplacements.begin(), sendDisplacements.end(), 0);
                    std::fill(sendCounts.begin(), sendCounts.end(), 0);
                    std::fill(receiveSizes.begin(), receiveSizes.end(), 0);
                    std::fill(receiveDisplacements.begin(), receiveDisplacements.end(), 0);
                    std::fill(receiveCounts.begin(), receiveCounts.end(), 0);
                }

            private:
                void Grow(const size_t p, std::vector<int> &info) const
                {
                    info.resize(info.size() + p);
                }
            } mSendQueueInfo;

            tBuffer mGetRequests;
            tBuffer mGetBuffers;
            tBuffer mPutBuffers;
            std::vector<SyncLibInternal::CommunicationBuffer> mGetDestinations;

            std::vector<AbstractSharedVariable *> mVariables;
            std::vector<AbstractSendQueue *> mSendQueues;

            size_t RegisterSendQueue(AbstractSendQueue *queue)
            {
                const size_t size = mSendQueues.size();
                mSendQueues.push_back(queue);
                mSendQueueInfo.Grow(Size());
                return size;
            }

            void DisableSendQueue(const size_t index)
            {
                mSendQueues[index] = nullptr;
            }

            size_t RegisterSharedVariable(AbstractSharedVariable *variable)
            {
                auto &variables = mVariables;
                const size_t size = variables.size();
                variables.push_back(variable);
                return size;
            }

            void DisableSharedVariable(size_t index)
            {
                mVariables[index] = nullptr;
            }

            void AddGetBufferSize(const size_t target, const size_t size) const
            {
                mGetRequests.sendSizes[target] += static_cast<int>(size);
            }

            SyncLibInternal::CommunicationBuffer &GetTargetPutBuffer(size_t target)
            {
                return mPutBuffers.sendBuffers[target];
            }

            SyncLibInternal::CommunicationBuffer &GetTargetGetBuffer(size_t target)
            {
                return mGetBuffers.sendBuffers[target];
            }

            SyncLibInternal::CommunicationBuffer &GetTargetGetRequests(size_t target)
            {
                return mGetRequests.sendBuffers[target];
            }

            SyncLibInternal::CommunicationBuffer &GetTargetGetDestinations(size_t target)
            {
                return mGetDestinations[target];
            }

            inline void BufferGet(const size_t &t)
            {
                // We've received get requests from t
                const int displacement = mGetRequests.receiveDisplacements[t];
                const int receiveSize = mGetRequests.receiveSizes[t];

                // We buffer them locally
                auto &getBuffer = mGetBuffers.sendBuffers[t];
                // Cleanup the previous synchronization
                getBuffer.Clear();

                for (const char *cursor = mGetRequests.receiveBuffer.Begin() + displacement, *end = cursor + receiveSize; cursor < end;)
                {
                    const tDataType dataType = static_cast<tDataType>(*cursor++);
                    const tDataType isArray = dataType & tDataType::Array;

                    const size_t *sCursor = reinterpret_cast<const size_t *>(cursor);
                    const size_t index = *sCursor++;
                    const size_t size = *sCursor++;

                    AbstractSharedVariable &sharedVariable = *mVariables[index];

                    if (isArray == tDataType::Array)
                    {
                        const size_t offset = *sCursor;
                        cursor += 3 * sizeof(size_t);
                        SyncLibInternal::FinaliseHelper::Finalise<void, SyncLibInternal::BufferGetHelper<true>>(dataType ^ isArray, sharedVariable,
                                                                                                                getBuffer.Reserve(size), offset,
                                                                                                                size);
                    }
                    else
                    {
                        cursor += 2 * sizeof(size_t);
                        SyncLibInternal::FinaliseHelper::Finalise<void, SyncLibInternal::BufferGetHelper<false>>(dataType ^ isArray, sharedVariable,
                                                                                                                 getBuffer.Reserve(size));
                    }
                }
            }

            void ProcessPut(const size_t &t)
            {
                const int displacement = mPutBuffers.receiveDisplacements[t];
                const int receiveSize = mPutBuffers.receiveSizes[t];

                for (const char *cursor = mPutBuffers.receiveBuffer.Begin() + displacement, *end = cursor + receiveSize; cursor < end;)
                {
                    const tDataType dataType = static_cast<tDataType>(*cursor++);
                    const tDataType isArray = dataType & tDataType::Array;

                    const size_t *sCursor = reinterpret_cast<const size_t *>(cursor);
                    const size_t index = *sCursor++;
                    const size_t size = *sCursor++;

                    AbstractSharedVariable &sharedVariable = *mVariables[index];

                    if (isArray == tDataType::Array)
                    {
                        const size_t offset = *sCursor;
                        cursor += 3 * sizeof(size_t);
                        SyncLibInternal::FinaliseHelper::Finalise<void, SyncLibInternal::FinalisePutHelper<true>>(dataType ^ isArray, sharedVariable,
                                                                                                                  cursor, offset, size);
                    }
                    else
                    {
                        cursor += 2 * sizeof(size_t);
                        SyncLibInternal::FinaliseHelper::Finalise<void, SyncLibInternal::FinalisePutHelper<false>>(dataType ^ isArray, sharedVariable,
                                                                                                                   cursor);
                    }

                    cursor += size;
                }
            }

            void ProcessGet(const size_t &t)
            {
                const int displacement = mGetRequests.receiveDisplacements[t];
                const int receiveSize = mGetRequests.receiveSizes[t];

                const char *bufferCursor = mGetBuffers.receiveBuffer.Begin() + mGetBuffers.receiveDisplacements[t];
                const char *destinationCursor = mGetDestinations[t].Begin();

                for (const char *cursor = mGetRequests.receiveBuffer.Begin() + displacement, *end = cursor + receiveSize; cursor < end;)
                {
                    const tDataType dataType = static_cast<tDataType>(*cursor++);
                    const tDataType isArray = dataType & tDataType::Array;

                    const size_t *sCursor = reinterpret_cast<const size_t *>(cursor);
                    // Skip index
                    ++sCursor;
                    const size_t size = *sCursor;

                    if (isArray == tDataType::Array)
                    {
                        cursor += 3 * sizeof(size_t);
                        destinationCursor +=
                            SyncLibInternal::FinaliseHelper::Finalise<size_t, SyncLibInternal::FinaliseGetHelper<true>>(dataType ^ isArray, bufferCursor,
                                                                                                                        destinationCursor, size);
                        bufferCursor += size;
                    }
                    else
                    {
                        cursor += 2 * sizeof(size_t);
                        destinationCursor +=
                            SyncLibInternal::FinaliseHelper::Finalise<size_t, SyncLibInternal::FinaliseGetHelper<false>>(dataType ^ isArray, bufferCursor,
                                                                                                                         destinationCursor);
                        bufferCursor += size;
                    }
                }
            }
        };
    } // namespace Environments

} // namespace SyncLib

#endif
