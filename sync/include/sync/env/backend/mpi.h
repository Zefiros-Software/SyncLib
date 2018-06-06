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
#include "sync/util/json.h"

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
            mSendQueueInfo.Clear();
        }

        MPIBackend(::SyncLib::MPI::Comm &&comm)
            : mCommHolder(std::move(comm))
            , mComm(std::get<0>(mCommHolder))
        {
            comm.TakeOwnership();
            mSendQueueInfo.Clear();
        }

        MPIBackend(const int argc, char **argv)
            : mCommHolder(::SyncLib::MPI::Comm{ argc, argv })
            , mComm(std::get<::SyncLib::MPI::Comm>(mCommHolder))
        {
            mSendQueueInfo.Clear();
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

        void CheckMainThread() const
        {
            if (!EnvHelper::isMainThread)
            {
                throw std::runtime_error("When combining MPI and SharedMemory BSP in an algorithm, only the SharedMemory processor with ID 0 should communicate");
            }
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
            ExchangeSizes(EnvHelper::GetGetRequests<tBuffer>(env), mGetRequestSizesWaiter);
            ExchangeSizes(EnvHelper::GetGetBuffers<tBuffer>(env), mGetBufferSizesWaiter);
            ExchangeSizes(EnvHelper::GetPutBuffers<tBuffer>(env), mPutBufferSizesWaiter);

            const size_t p = mComm.Size();
            mSendQueueSizeRequests.clear();
            mSendQueueCountRequests.clear();

            const size_t sqCount = EnvHelper::GetSendQueues(env).size();
            auto &sendInfo = mSendQueueInfo;//EnvHelper::GetSendInfo(env);

            mSendQueueSizeRequests.resize(sqCount);
            mSendQueueCountRequests.resize(sqCount);

            mSendQueueInfo.Resize(sqCount, Size());
            mSendQueueInfo.Clear();
            /*{
                for (size_t i = 0; i < sqCount; ++i)
                {
                }
            }*/
            auto &sendQueues = EnvHelper::GetSendQueues(env);

            for (size_t i = 0; i < sqCount; ++i)
            {
                const size_t sqOffset = i * p;

                if (sendQueues[i])
                {
                    //                     const char *sqBufferBegin = sendQueues[i]->GetTargetBuffer(0).Begin();
                    //                     MPI_Aint AintBegin;
                    //                     MPI_Get_address(sqBufferBegin, &AintBegin);

                    for (size_t t = 0; t < p; ++t)
                    {
                        const char *sqBuffer = sendQueues[i]->GetTargetBuffer(t).Begin();

                        mSendQueueInfo.sendSizes[sqOffset + t] = static_cast<int>(sendQueues[i]->GetTargetSize(t));
                        //mSendQueueInfo.sendDisplacements[sqOffset + t] = static_cast<int>(sqBuffer - sqBufferBegin);
                        //                         MPI_Aint AintAddr;
                        //                         MPI_Get_address(sqBuffer, &AintAddr);
                        //mSendQueueInfo.sendDisplacements[sqOffset + t] = MPI_Aint_diff(AintAddr, AintBegin);
                        mSendQueueInfo.sendBuffers[sqOffset + t] = sqBuffer;
                        // MPI_Get_address(sqBuffer, &mSendQueueInfo.sendDisplacements[sqOffset + t]);
                        mSendQueueInfo.sendCounts[sqOffset + t] = static_cast<int>(sendQueues[i]->GetTargetCount(t));
                    }
                }

                mComm.AllToAllNonBlocking(&sendInfo.sendSizes[sqOffset], &sendInfo.receiveSizes[sqOffset], 1, mSendQueueSizeRequests[i]);
                mComm.AllToAllNonBlocking(&sendInfo.sendCounts[sqOffset], &sendInfo.receiveCounts[sqOffset], 1, mSendQueueCountRequests[i]);
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
            SynchroniseBuffer(EnvHelper::GetGetRequests<tBuffer>(env), mGetRequestSizesWaiter, mGetRequestWaiter);
        }

        void SynchronisePutBuffers(tEnv &env)
        {
            SynchroniseBuffer(EnvHelper::GetPutBuffers<tBuffer>(env), mPutBufferSizesWaiter, mPutBufferWaiter);
        }

        void SynchroniseGetBuffers(tEnv &env)
        {
            SynchroniseBuffer(EnvHelper::GetGetBuffers<tBuffer>(env), mGetBufferSizesWaiter, mGetBufferWaiter);
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

        void WaitForSendQueues(const size_t sqCount, tEnv &env)
        {
            auto &sendQueues = EnvHelper::GetSendQueues(env);

            for (size_t i = 0; i < sqCount; ++i)
            {
                if (!sendQueues[i])
                {
                    fmt::print("Warning: sendqueue {} was null\n", i);
                    fflush(stdout);
                    continue;
                }

                mSendQueueWaiters[i].Wait();
            }
        }

        void SynchroniseSendQueues(tEnv &env)
        {
            mSendQueueWaiters.clear();
            const size_t p = mComm.Size();

            auto &sendQueues = EnvHelper::GetSendQueues(env);
            const size_t sqCount = sendQueues.size();
            auto &sendInfo = mSendQueueInfo;//EnvHelper::GetSendInfo(env);
            mSendQueueWaiters.resize(sqCount);

            for (size_t i = 0; i < sqCount; ++i)
            {
                mSendQueueSizeRequests[i].Wait();
                mSendQueueCountRequests[i].Wait();

                size_t receiveDisplacement = 0;
                size_t receiveCount = 0;
                const size_t sqOffset = i * p;

                for (size_t t = 0; t < p; ++t)
                {
                    // sendInfo.receiveDisplacements[sqOffset + t] = static_cast<int>(receiveDisplacement);
                    receiveDisplacement += sendInfo.receiveSizes[sqOffset + t];
                    receiveCount += sendInfo.receiveCounts[sqOffset + t];
                }

                if (!sendQueues[i])
                {
                    fmt::print("Warning: sendqueue {} was null\n", i);
                    fflush(stdout);
                    // mSendQueueWaiters.emplace_back(MPI_REQUEST_NULL);
                    *mSendQueueWaiters[i].request = MPI_REQUEST_NULL;
                    continue;
                }

                for (size_t t = 0; t < p; ++t)
                {

                    auto buff = sendQueues[i]->GetTargetBuffer(t).Begin();
                    /* SyncLibInternal::Assert(sendInfo.sendBuffers[sqOffset + t] == buff, "Buffer mismatches, actual {}, but predicted {}\n",
                                             buff, sendInfo.sendBuffers[sqOffset + t]);
                     SyncLibInternal::Assert(sendQueues[i]->GetTargetBuffer(t).Size() == sendInfo.sendSizes[sqOffset + t], "Sizes mismatch");*/
                }

                char *dest = sendQueues[i]->ReserveReceiveSpace(receiveCount, receiveDisplacement);
                //                 MPI_Aint AintDest;
                //                 MPI_Get_address(dest, &AintDest);
                receiveDisplacement = 0;

                for (size_t t = 0; t < p; ++t)
                {
                    // sendInfo.receiveDisplacements[sqOffset + t] = static_cast<int>(receiveDisplacement);
                    //                     MPI_Aint AintAddr;
                    //                     MPI_Get_address(dest + receiveDisplacement, &AintAddr);
                    //sendInfo.receiveDisplacements[sqOffset + t] = MPI_Aint_diff(AintAddr, AintDest);
                    sendInfo.receiveBuffers[sqOffset + t] = dest + receiveDisplacement;
                    //MPI_Get_address(dest + receiveDisplacement, &sendInfo.receiveDisplacements[sqOffset + t]);
                    receiveDisplacement += sendInfo.receiveSizes[sqOffset + t];
                    // receiveDisplacement += sendInfo.receiveSizes[sqOffset + t];
                    // receiveCount += sendInfo.receiveCounts[sqOffset + t];
                    //SyncLibInternal::Assert(receiveDisplacement <= sendQueues[i]->GetSizeBytes(), "Receive cursor went outside the receive buffer\n");
                }

                /*mComm.AllToAllVNonBlocking(sendQueues[i]->GetTargetBuffer(0).Begin(),
                                           &sendInfo.sendSizes[sqOffset],
                                           &sendInfo.sendDisplacements[sqOffset], dest, &sendInfo.receiveSizes[sqOffset],
                                           &sendInfo.receiveDisplacements[sqOffset], mSendQueueWaiters[i]);*/
                mComm.AllToAllVNonBlocking(&sendInfo.sendBuffers[sqOffset], &sendInfo.sendSizes[sqOffset], &sendInfo.receiveBuffers[sqOffset],
                                           &sendInfo.receiveSizes[sqOffset], mSendQueueWaiters[i]);
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

        struct
        {
            std::vector<int> sendSizes;
            std::vector<const void *> sendBuffers;
            std::vector<int> sendCounts;
            std::vector<int> receiveSizes;
            std::vector<void *> receiveBuffers;
            std::vector<int> receiveCounts;

            void Grow(size_t p)
            {
                Grow(p, sendSizes);
                Grow(p, sendBuffers);
                Grow(p, sendCounts);
                Grow(p, receiveSizes);
                Grow(p, receiveBuffers);
                Grow(p, receiveCounts);
            }

            void Resize(size_t count, size_t p)
            {
                Resize(count, p, sendSizes);
                Resize(count, p, sendBuffers);
                Resize(count, p, sendCounts);
                Resize(count, p, receiveSizes);
                Resize(count, p, receiveBuffers);
                Resize(count, p, receiveCounts);
            }

            void Clear()
            {
                std::fill(sendSizes.begin(), sendSizes.end(), 0);
                std::fill(sendBuffers.begin(), sendBuffers.end(), nullptr);
                std::fill(sendCounts.begin(), sendCounts.end(), 0);
                std::fill(receiveSizes.begin(), receiveSizes.end(), 0);
                std::fill(receiveBuffers.begin(), receiveBuffers.end(), nullptr);
                std::fill(receiveCounts.begin(), receiveCounts.end(), 0);
            }

        private:

            template<typename tT>
            void Grow(const size_t p, std::vector<tT> &info) const
            {
                info.resize(info.size() + p);
            }

            template<typename tT>
            void Resize(size_t count, size_t p, std::vector<tT> &info)
            {
                info.resize(count * p);
            }
        } mSendQueueInfo;

        void ExchangeSizes(DistributedCommunicationBuffer &buffer, tNonBlockingRequest &req) const
        {
            /*fmt::print("Send sizes are {} on node {}\n", json(arma::conv_to<arma::Col<int>>::from(buffer.sendSizes)).dump(), Rank());
            fmt::print("Receive sizes (before sync) are {} on node {}\n", json(arma::conv_to<arma::Col<int>>::from(buffer.sendSizes)).dump(),
                       Rank());*/

            mComm.AllToAllNonBlocking(&buffer.sendSizes[0], &buffer.receiveSizes[0], 1, req);
        }

        void SynchroniseBuffer(DistributedCommunicationBuffer &buffer, tNonBlockingRequest &waiter, tNonBlockingRequest &request) const
        {
            /*fmt::print("Waiting on node {}\n", Rank());
            fflush(stdout);*/
            waiter.Wait();
            /*fmt::print("Done waiting on node {}\n", Rank());
            fmt::print("Receive sizes (after waiting) are {} on node {}\n",
                       json(arma::conv_to<arma::Col<int>>::from(buffer.receiveSizes)).dump(), Rank());
            fflush(stdout);*/

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

            // fmt::print("Receive sizes are {} on node {}\n", json(arma::conv_to<arma::Col<int>>::from(buffer.receiveSizes)).dump(),  Rank());

            buffer.receiveBuffer.Clear();
            char *dest = buffer.receiveBuffer.Reserve(receiveDisplacement);
            /*fmt::print("Receiving {}, sending {} bytes on node {}\n", receiveDisplacement, std::accumulate(buffer.sendSizes.begin(),
                                                                                                           buffer.sendSizes.end(), 0), Rank());
            fflush(stdout);*/

            mComm.AllToAllVNonBlocking(bufferBegin, &buffer.sendSizes[0], &buffer.sendDisplacements[0], dest, &buffer.receiveSizes[0],
                                       &buffer.receiveDisplacements[0], request);
        }
    };
} // namespace SyncLibInternal

#endif
