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
#ifndef __SYNCLIB_BACKEND_SHARED_H__
#define __SYNCLIB_BACKEND_SHARED_H__

#include "sync/barrier/shared/condVarBarrier.h"
#include "sync/barrier/shared/spinBarrier.h"
#include "sync/env/base.h"
#include "sync/util/threads.h"

#ifdef IS_WINDOWS
#include <execution>
#endif
#include <thread>

namespace SyncLibInternal
{
    class SharedMemoryBackend
    {
    public:
        using tEnv = SyncLib::Environments::BaseBSP<SharedMemoryBackend>;
        using tBuffer = DistributedCommunicationBuffer;
        using tBarrier = SpinBarrier;

        SharedMemoryBackend(size_t size = 0)
            : mBarrierHolder(tBarrier{ SharedMemoryBackend::SizeOrDefault(size) })
            , mBarrier(std::get<0>(mBarrierHolder))
            , mOthersHolder(std::vector<tEnv *>(SharedMemoryBackend::SizeOrDefault(size)))
            , mOthers(std::get<0>(mOthersHolder))
            , mSize(SharedMemoryBackend::SizeOrDefault(size))
            , mSizePow2(SyncLib::Util::NextPowerOfTwo(SharedMemoryBackend::SizeOrDefault(size)))
            , mRank(0)
        {
        }

        SharedMemoryBackend(int /*argc*/, char ** /*argv*/)
            : SharedMemoryBackend()
        {
        }

        SharedMemoryBackend(SharedMemoryBackend &other, size_t rank)
            : mBarrierHolder(&other.mBarrier)
            , mBarrier(*std::get<1>(mBarrierHolder))
            , mOthersHolder(&other.mOthers)
            , mOthers(*std::get<1>(mOthersHolder))
            , mSize(other.mSize)
            , mSizePow2(other.mSizePow2)
            , mRank(rank)
        {
        }

        ~SharedMemoryBackend()
        {
            for (tEnv *env : mSubEnvs)
            {
                delete env;
            }
        }

        template <typename tFunc, typename... tArgs>
        inline auto Run(tEnv &env, const tFunc &func, tArgs &&... args)
        {
            EnvHelper::GetSendQueues(env).clear();
            EnvHelper::GetVariables(env).clear();
            mOthers.resize(env.Size());
            bool isMain = EnvHelper::isMainThread;

            // #ifdef IS_WINDOWS
            //             arma::Col<size_t> S = arma::regspace<arma::Col<size_t>>(0, env.Size() - 1);
            //             std::for_each(std::execution::par, S.begin(), S.end(), [ &, this](size_t s)
            //             {
            //                 if (s == 0)
            //                 {
            //                     mOthers[0] = &env;
            //                     EnvHelper::isMainThread = true;
            //                     //env.Barrier();
            //                     func(env, std::forward<tArgs>(args)...);
            //                 }
            //                 else
            //                 {
            //                     tEnv otherEnv(*this, s);
            //                     mOthers[s] = &otherEnv;
            //                     EnvHelper::isMainThread = false;
            //                     //otherEnv.Barrier();
            //
            //                     func(otherEnv, std::forward<tArgs>(args)...);
            //                 }
            //             });
            // #else
            auto lambda = [&](size_t s)
            {
                tEnv otherEnv(*this, s);
                mOthers[s] = &otherEnv;
                mThreadAffinityHelper.PinThread(s);
                EnvHelper::isMainThread = false;
                //otherEnv.Barrier();

                return func(otherEnv, std::forward<tArgs>(args)...);
            };

            mOthers[0] = &env;

            std::vector<std::thread> threads;

            for (size_t t = 1; t < mSize; ++t)
            {
                threads.emplace_back(lambda, t);
            }

            mThreadAffinityHelper.PinThread(0);
            //env.Barrier();
            func(env, std::forward<tArgs>(args)...);

            for (auto &thr : threads)
            {
                if (thr.joinable())
                {
                    thr.join();
                }
            }

            //#endif

            EnvHelper::isMainThread = isMain;

            for (tEnv *subEnv : mSubEnvs)
            {
                delete subEnv;
            }

            mSubEnvs.clear();
            mSubEnvInit.clear();

            return 0;
        }

        size_t Rank() const
        {
            return mRank;
        }

        size_t Size() const
        {
            return mSize;
        }

        void Barrier() const
        {
            //if (mOthers[0])
            {
                mBarrier.Wait();
            }
        }

        void CheckMainThread() const
        {
        }

        static size_t MaxSize()
        {
            return ThreadAffinityHelper().GetCoreCount();
        }

        static size_t MaxVirtualSize()
        {
            return ThreadAffinityHelper().GetThreadCount();
        }

        tEnv &Split(size_t part, size_t rank)
        {
            mSubEnvInit.emplace_back(part, rank);
            //             fmt::print("{} enters split with ({}, {})\n", mRank, part, rank);
            //             fflush(stdout);
            Barrier();
            //            fmt::print("\n");

            std::vector<std::tuple<size_t, size_t>> partMembers;

            for (tEnv *other : mOthers)
            {
                auto &otherBackend = EnvHelper::GetBackend(*other);
                auto [otherPart, otherRank] = otherBackend.mSubEnvInit.back();

                if (otherPart == part)
                {
                    partMembers.emplace_back(otherRank, otherBackend.mRank);
                }
            }

            std::sort(partMembers.begin(), partMembers.end());
            size_t newRank = std::find(partMembers.begin(), partMembers.end(), std::make_tuple(rank, mRank)) - partMembers.begin();

            if (newRank == 0)
            {
                mSubEnvs.push_back(new tEnv(partMembers.size()));
            }

            Barrier();
            //fmt::print("{} joins part {} of size {} with rank {}\n", mRank, part, partMembers.size(), newRank);

            if (newRank != 0)
            {
                auto &otherBackend = EnvHelper::GetBackend(*mOthers[std::get<1>(partMembers[0])]);
                mSubEnvs.push_back(new tEnv(EnvHelper::GetBackend(*otherBackend.mSubEnvs.back()), newRank));
            }

            tEnv &env = *mSubEnvs.back();
            auto &backend = EnvHelper::GetBackend(env);
            backend.mOthers[newRank] = &env;
            env.Barrier();

            return env;
        }

        void UnSplit(tEnv &env)
        {
            env.Barrier();
            size_t i = std::find(mSubEnvs.begin(), mSubEnvs.end(), &env) - mSubEnvs.begin();
            delete mSubEnvs[i];
            mSubEnvs.erase(mSubEnvs.begin() + i);
            mSubEnvInit.erase(mSubEnvInit.begin() + i);
        }

        void Resize(size_t size)
        {
            if (!mOthers.empty() && mOthers[0] != nullptr)
            {
                throw std::runtime_error("Cannot resize an active environment");
            }

            size = SharedMemoryBackend::SizeOrDefault(size);
            mBarrier.Resize(size);

            mOthers.resize(size);
            mSize = size;
            mSizePow2 = SyncLib::Util::NextPowerOfTwo(size);
        }

        template <typename... tArgs>
        void Print(tArgs &&... args)
        {
            fmt::print(std::forward<tArgs>(args)...);
        }

        void ExchangeSizes(tEnv &env) const
        {
            mBarrier.Wait();

            const size_t s = mRank;

            auto &getRequests = EnvHelper::GetGetRequests<tBuffer>(env);
            auto &getBuffers = EnvHelper::GetGetBuffers<tBuffer>(env);
            auto &putBuffers = EnvHelper::GetPutBuffers<tBuffer>(env);

            const auto &sendQueues = EnvHelper::GetSendQueues(env);
            const size_t sqCount = sendQueues.size();
            //auto &sendInfo = EnvHelper::GetSendInfo(env);

            for (size_t t = 0; t < mSize; ++t)
            {
                tEnv &otherEnv = *mOthers[t];
                getRequests.receiveSizes[t] = EnvHelper::GetGetRequests<tBuffer>(otherEnv).sendSizes[s];
                getBuffers.receiveSizes[t] = EnvHelper::GetGetBuffers<tBuffer>(otherEnv).sendSizes[s];
                putBuffers.receiveSizes[t] = EnvHelper::GetPutBuffers<tBuffer>(otherEnv).sendSizes[s];

                /*for (size_t i = 0; i < sqCount; ++i)
                {
                    if (sendQueues[i] == nullptr)
                    {
                        continue;
                    }

                    const size_t sqOffset = i * mSize;

                    sendInfo.receiveSizes[sqOffset + t] = EnvHelper::GetSendInfo(otherEnv).sendSizes[sqOffset + s];
                    sendInfo.receiveCounts[sqOffset + t] = EnvHelper::GetSendInfo(otherEnv).sendCounts[sqOffset + s];
                }*/
            }
        }

        void SynchroniseGetRequestsPutBuffersSendQueues(tEnv &env)
        {
            size_t s = env.Rank();

            auto &getRequests = EnvHelper::GetGetRequests<tBuffer>(env);
            auto &putBuffers = EnvHelper::GetPutBuffers<tBuffer>(env);

            char *getRequestCursor = PrepareReceiveBuffer(getRequests);
            char *putBufferCursor = PrepareReceiveBuffer(putBuffers);
            std::vector<char *> sqCursors;

            auto &sendQueues = EnvHelper::GetSendQueues(env);
            const size_t sqCount = sendQueues.size();
            //auto &sendInfo = EnvHelper::GetSendInfo(env);
            sqCursors.reserve(sqCount);

            for (size_t i = 0; i < sqCount; ++i)
            {
                if (sendQueues[i] == nullptr)
                {
                    sqCursors.emplace_back(nullptr);
                    continue;
                }

                size_t receiveDisplacement = 0;
                size_t receiveCount = 0;
                const size_t sqOffset = i * mSize;

                for (size_t t = 0; t < mSize; ++t)
                {
                    auto &sq = *EnvHelper::GetSendQueues(*mOthers[t])[i];
                    //sendInfo.receiveDisplacements[sqOffset + t] = static_cast<int>(receiveDisplacement);
                    receiveDisplacement += sq.GetTargetSize(s);//sendInfo.receiveSizes[sqOffset + t];
                    receiveCount += sq.GetTargetCount(s);//sendInfo.receiveCounts[sqOffset + t];
                }

                char *dest = sendQueues[i]->ReserveReceiveSpace(receiveCount, receiveDisplacement);
                sqCursors.emplace_back(dest);
            }

            for (size_t mask = 0; mask < mSizePow2; ++mask)
            {
                const size_t t = mRank ^ mask;

                if (t >= mSize)
                {
                    continue;
                }

                SynchroniseBuffer(getRequests, getRequestCursor, t, EnvHelper::GetGetRequests<tBuffer>(*mOthers[t]));
                SynchroniseBuffer(putBuffers, putBufferCursor, t, EnvHelper::GetPutBuffers<tBuffer>(*mOthers[t]));

                for (size_t i = 0; i < sqCount; ++i)
                {
                    if (sendQueues[i] == nullptr)
                    {
                        continue;
                    }

                    const size_t sqOffset = i * mSize;
                    auto &otherQueue = *EnvHelper::GetSendQueues(*mOthers[t])[i];
                    // auto &otherInfo = EnvHelper::GetSendInfo(*mOthers[t]);
                    std::copy_n(otherQueue.GetTargetBuffer(mRank).Begin(), // + otherInfo.sendDisplacements[sqOffset + t],
                                otherQueue.GetTargetSize(mRank), sqCursors[i]);
                    sqCursors[i] += otherQueue.GetTargetSize(mRank);
                }
            }

            mSendQueueRemoval.clear();

            for (size_t i = 0; i < sqCount; ++i)
            {
                if (sendQueues[i] == nullptr)
                {
                    for (size_t t = 0; t < mSize; ++t)
                    {
                        if (EnvHelper::GetSendQueues(*mOthers[t])[i] != nullptr)
                        {
                            continue;
                        }
                    }

                    mSendQueueRemoval.push_back(i);
                }
            }
        }

        void SynchroniseGetBuffers(tEnv &env) const
        {
            mBarrier.Wait();
            auto &getBuffers = EnvHelper::GetGetBuffers<tBuffer>(env);
            char *getBufferCursor = PrepareReceiveBuffer(getBuffers);

            for (size_t mask = 0; mask < mSizePow2; ++mask)
            {
                const size_t t = mRank ^ mask;

                if (t >= mSize)
                {
                    continue;
                }

                SynchroniseBuffer(getBuffers, getBufferCursor, t, EnvHelper::GetGetBuffers<tBuffer>(*mOthers[t]));
            }
        }

        void WaitForGetRequests() const
        {
        }

        void WaitForPutBuffers() const
        {
        }

        void WaitForGetBuffers() const
        {
        }

        void WaitForSendQueues(size_t /*sqCount*/, tEnv & /*env*/)
        {
            if (!mSendQueueRemoval.empty())
            {
                auto &sendQueues = EnvHelper::GetSendQueues(*mOthers[mRank]);

                for (auto it = mSendQueueRemoval.rbegin(), end = mSendQueueRemoval.rend(); it != end; ++it)
                {
                    sendQueues.erase(sendQueues.begin() + *it);
                }

                for (size_t i = mSendQueueRemoval[0]; i < sendQueues.size(); ++i)
                {
                    sendQueues[i]->SetIndex(i);
                }
            }
        }

    private:
        std::variant<tBarrier, tBarrier *> mBarrierHolder;
        tBarrier &mBarrier;

        std::variant<std::vector<tEnv *>, std::vector<tEnv *> *> mOthersHolder;
        std::vector<tEnv *> &mOthers;
        std::vector<size_t> mSendQueueRemoval;
        std::vector<tEnv *> mSubEnvs;
        std::vector<std::tuple<size_t, size_t>> mSubEnvInit;

        SyncLibInternal::ThreadAffinityHelper mThreadAffinityHelper;

        size_t mSize;
        size_t mSizePow2;
        size_t mRank;

        char *PrepareReceiveBuffer(tBuffer &buffer) const
        {

            const char *bufferBegin = buffer.sendBuffers[0].Begin();
            int receiveDisplacement = 0;

            for (size_t t = 0; t < mSize; ++t)
            {
                auto &bufferT = buffer.sendBuffers[t];

                buffer.sendDisplacements[t] = static_cast<int>(bufferT.Begin() - bufferBegin);
                buffer.receiveDisplacements[t] = receiveDisplacement;

                receiveDisplacement += buffer.receiveSizes[t];
            }

            buffer.receiveBuffer.Clear();
            return buffer.receiveBuffer.Reserve(receiveDisplacement);
        }

        void SynchroniseBuffer(tBuffer &myBuffer, char *myCursor, size_t other, tBuffer &otherBuffer) const
        {
            std::copy_n(otherBuffer.sendBuffers[mRank].Begin() /*+ otherBuffer.sendDisplacements[mRank]*/, myBuffer.receiveSizes[other],
                        myCursor + myBuffer.receiveDisplacements[other]);
        }

        static size_t SizeOrDefault(size_t size)
        {
            return size > 0 ? size : SharedMemoryBackend::MaxSize();
        }
    };
} // namespace SyncLibInternal

#endif
