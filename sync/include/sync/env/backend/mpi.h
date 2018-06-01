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
#ifndef __SYNCLIB_BACKEND_MPI_H__
#define __SYNCLIB_BACKEND_MPI_H__

#include "sync/env/base.h"

namespace SyncLibInternal
{
    class MPIBackend
    {
    public:
        using tEnv = SyncLib::Environments::BaseBSP<MPIBackend>;
        using tBuffer = DistributedCommunicationBuffer;
        using tNonBlockingRequest = SyncLib::MPI::NonBlockingRequest;

        MPIBackend(::SyncLib::MPI::Comm &comm)
            : mCommHolder(&comm)
            , mComm(comm)
        {
        }

        MPIBackend(::SyncLib::MPI::Comm &&comm)
            : mCommHolder(std::move(comm))
            , mComm(std::get<0>(mCommHolder))
        {
            comm.TakeOwnership();
        }

        MPIBackend(const int argc, char **argv)
            : mCommHolder(::SyncLib::MPI::Comm{ argc, argv })
            , mComm(std::get<::SyncLib::MPI::Comm>(mCommHolder))
        {
        }

        ~MPIBackend()
        {
            for (tEnv *subEnv : mSubEnvs)
            {
                delete subEnv;
            }
        }

        template <typename tFunc, typename... tArgs>
        inline auto Run(tEnv &env, const tFunc &func, tArgs &&... args)
        {
            return func(env, std::forward<tArgs>(args)...);
        }

        size_t Rank() const
        {
            return mComm.Rank();
        }

        size_t Size() const
        {
            return mComm.Size();
        }

        void Barrier() const
        {
            mComm.Barrier();
        }

        static size_t MaxSize()
        {
            return ::SyncLib::MPI::Comm(MPI_COMM_WORLD).Size();
        }

        static size_t MaxVirtualSize()
        {
            return tEnv::MaxSize();
        }

        tEnv &Split(const size_t part, const size_t rank)
        {
            mSubEnvs.push_back(new tEnv(mComm.Split(part, rank)));
            return *mSubEnvs.back();
        }

        void Resize(size_t /*size*/) const
        {
            throw std::runtime_error("implement this");
        }

        template <typename... tArgs>
        void Print(tArgs &&... args)
        {
            mComm.Print(std::forward<tArgs>(args)...);
        }

        void ExchangeSizes(tEnv &env)
        {
            mGetRequestSizesWaiter = ExchangeSizes(EnvHelper::GetGetRequests<tBuffer>(env));
            mGetBufferSizesWaiter = ExchangeSizes(EnvHelper::GetGetBuffers<tBuffer>(env));
            mPutBufferSizesWaiter = ExchangeSizes(EnvHelper::GetPutBuffers<tBuffer>(env));

            const size_t p = mComm.Size();
            mSendQueueSizeRequests.clear();
            mSendQueueCountRequests.clear();

            const size_t sqCount = EnvHelper::GetSendQueues(env).size();
            auto &sendInfo = EnvHelper::GetSendInfo(env);

            for (size_t i = 0; i < sqCount; ++i)
            {
                const size_t sqOffset = i * p;

                mSendQueueSizeRequests.emplace_back(mComm.AllToAllNonBlocking(&sendInfo.sendSizes[sqOffset], &sendInfo.receiveSizes[sqOffset],
                                                                              1));
                mSendQueueCountRequests.emplace_back(mComm.AllToAllNonBlocking(&sendInfo.sendCounts[sqOffset], &sendInfo.receiveCounts[sqOffset],
                                                                               1));
            }
        }

        void SynchroniseGetRequestsPutBuffersSendQueues(tEnv &env)
        {
            SynchroniseGetRequests(env);
            SynchronisePutBuffers(env);
            SynchroniseSendQueues(env);
        }

        void SynchroniseGetRequests(tEnv &env)
        {
            mGetRequestWaiter = SynchroniseBuffer(EnvHelper::GetGetRequests<tBuffer>(env), mGetRequestSizesWaiter);
        }

        void SynchronisePutBuffers(tEnv &env)
        {
            mPutBufferWaiter = SynchroniseBuffer(EnvHelper::GetPutBuffers<tBuffer>(env), mPutBufferSizesWaiter);
        }

        void SynchroniseGetBuffers(tEnv &env)
        {
            mGetBufferWaiter = SynchroniseBuffer(EnvHelper::GetGetBuffers<tBuffer>(env), mGetBufferSizesWaiter);
        }

        void WaitForGetRequests()
        {
            mGetRequestWaiter.Wait();
        }

        void WaitForPutBuffers()
        {
            mPutBufferWaiter.Wait();
        }

        void WaitForGetBuffers()
        {
            mGetBufferWaiter.Wait();
        }

        void WaitForSendQueues(const size_t sqCount)
        {
            for (size_t i = 0; i < sqCount; ++i)
            {
                mSendQueueWaiters[i].Wait();
            }
        }

        void SynchroniseSendQueues(tEnv &env)
        {
            mSendQueueWaiters.clear();
            const size_t p = mComm.Size();

            auto &sendQueues = EnvHelper::GetSendQueues(env);
            const size_t sqCount = sendQueues.size();
            auto &sendInfo = EnvHelper::GetSendInfo(env);

            for (size_t i = 0; i < sqCount; ++i)
            {
                mSendQueueSizeRequests[i].Wait();
                mSendQueueCountRequests[i].Wait();

                size_t receiveDisplacement = 0;
                size_t receiveCount = 0;
                const size_t sqOffset = i * p;

                for (size_t t = 0; t < p; ++t)
                {
                    sendInfo.receiveDisplacements[sqOffset + t] = static_cast<int>(receiveDisplacement);
                    receiveDisplacement += sendInfo.receiveSizes[sqOffset + t];
                    receiveCount += sendInfo.receiveCounts[sqOffset + t];
                }

                char *dest = sendQueues[i]->ReserveReceiveSpace(receiveCount, receiveDisplacement);
                mSendQueueWaiters.emplace_back(mComm.AllToAllVNonBlocking(sendQueues[i]->GetTargetBuffer(0).Begin(),
                                                                          &sendInfo.sendSizes[sqOffset],
                                                                          &sendInfo.sendDisplacements[sqOffset], dest, &sendInfo.receiveSizes[sqOffset],
                                                                          &sendInfo.receiveDisplacements[sqOffset]));
            }
        }

    private:
        std::variant<::SyncLib::MPI::Comm, ::SyncLib::MPI::Comm *> mCommHolder;
        ::SyncLib::MPI::Comm &mComm;

        std::vector<tEnv *> mSubEnvs;

        tNonBlockingRequest mGetRequestSizesWaiter;
        tNonBlockingRequest mGetBufferSizesWaiter;
        tNonBlockingRequest mPutBufferSizesWaiter;

        tNonBlockingRequest mGetRequestWaiter;
        tNonBlockingRequest mPutBufferWaiter;
        tNonBlockingRequest mGetBufferWaiter;

        std::vector<tNonBlockingRequest> mSendQueueSizeRequests;
        std::vector<tNonBlockingRequest> mSendQueueCountRequests;

        std::vector<tNonBlockingRequest> mSendQueueWaiters;

        tNonBlockingRequest ExchangeSizes(DistributedCommunicationBuffer &buffer) const
        {
            return mComm.AllToAllNonBlocking(buffer.sendSizes, buffer.receiveSizes, 1);
        }

        tNonBlockingRequest SynchroniseBuffer(DistributedCommunicationBuffer &buffer, tNonBlockingRequest &waiter) const
        {
            waiter.Wait();

            const size_t p = mComm.Size();
            const char *bufferBegin = buffer.sendBuffers[0].Begin();
            int receiveDisplacement = 0;

            for (size_t t = 0; t < p; ++t)
            {
                auto &bufferT = buffer.sendBuffers[t];

                buffer.sendDisplacements[t] = static_cast<int>(bufferT.Begin() - bufferBegin);
                buffer.receiveDisplacements[t] = receiveDisplacement;

                receiveDisplacement += buffer.receiveSizes[t];
            }

            buffer.receiveBuffer.Clear();
            char *dest = buffer.receiveBuffer.Reserve(receiveDisplacement);

            return mComm.AllToAllVNonBlocking(bufferBegin, buffer.sendSizes, buffer.sendDisplacements, dest, buffer.receiveSizes,
                                              buffer.receiveDisplacements);
        }
    };
} // namespace SyncLibInternal

#endif
