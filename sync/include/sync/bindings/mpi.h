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
#ifndef __SYNCLIB_MPI_H__
#define __SYNCLIB_MPI_H__

#ifdef SYNCLIB_ENABLE_MPI

#include "sync/buffers/put.h"

#include "fmt/format.h"

#include <armadillo>

#include <mpi.h>

#include <stdint.h>
#include <string>

namespace SyncLib
{
    namespace MPI
    {
        template<typename tT>
        struct Types
        {
            template<typename tSize = int>
            static constexpr int Size(tSize count)
            {
                return static_cast<int>(count * sizeof(tT));
            }

            static constexpr auto Type()
            {
                return MPI_BYTE;
            }
        };

#define SYNCLIB_MPI_TYPE_OVERRIDE(type, mpiType)            \
        template<>                                          \
        struct Types<type>                                  \
        {                                                   \
            template<typename tSize = int>                        \
            static constexpr int Size(tSize count)          \
            {                                               \
                return static_cast<int>(count);             \
            }                                               \
                                                            \
            static constexpr auto Type()                    \
            {                                               \
                return mpiType;                             \
            }                                               \
        }

        SYNCLIB_MPI_TYPE_OVERRIDE(char, MPI_BYTE);
        SYNCLIB_MPI_TYPE_OVERRIDE(float, MPI_FLOAT);
        SYNCLIB_MPI_TYPE_OVERRIDE(double, MPI_DOUBLE);
        SYNCLIB_MPI_TYPE_OVERRIDE(int8_t, MPI_INT8_T);
        SYNCLIB_MPI_TYPE_OVERRIDE(int16_t, MPI_INT16_T);
        SYNCLIB_MPI_TYPE_OVERRIDE(int32_t, MPI_INT32_T);
        SYNCLIB_MPI_TYPE_OVERRIDE(int64_t, MPI_INT64_T);
        SYNCLIB_MPI_TYPE_OVERRIDE(uint8_t, MPI_UINT8_T);
        SYNCLIB_MPI_TYPE_OVERRIDE(uint16_t, MPI_UINT16_T);
        SYNCLIB_MPI_TYPE_OVERRIDE(uint32_t, MPI_UINT32_T);
        SYNCLIB_MPI_TYPE_OVERRIDE(uint64_t, MPI_UINT64_T);

#undef SYNCLIB_MPI_TYPE_OVERRIDE

        inline void Init(int argc, char **argv)
        {
            MPI_Init(&argc, &argv);
        }

        inline void Finalise()
        {
            MPI_Finalize();
        }

        inline bool Initialized()
        {
            int flag;
            MPI_Initialized(&flag);
            return flag;
        }

        class Comm
        {
        public:

            Comm(const MPI_Comm &comm)
                : mComm(comm),
                  mInitialized(false)
            {
                assert(SyncLib::MPI::Initialized());

                InitRankSize();
            }

            Comm(int argc, char **argv)
                : mInitialized(false)
            {
                if (!SyncLib::MPI::Initialized())
                {
                    SyncLib::MPI::Init(argc, argv);
                    mInitialized = true;
                }

                mComm = MPI_COMM_WORLD;

                InitRankSize();
            }

            ~Comm()
            {
                Barrier();

                if (mInitialized)
                {
                    SyncLib::MPI::Finalise();
                }
            }

            inline size_t Rank()
            {
                return mRank;
            }

            inline size_t Size()
            {
                return mSize;
            }

            template<typename tT>
            void Send(size_t target, const tT *buffer, size_t count, int tag = 0)
            {
                using TypeHelper = SyncLib::MPI::Types<tT>;
                MPI_Send(buffer, TypeHelper::Size(count), TypeHelper::Type(), static_cast<int>(target), tag, mComm);
            }

            template<typename tT>
            void Send(size_t target, const arma::Mat<tT> &buffer, int tag = 0)
            {
                using TypeHelper = SyncLib::MPI::Types<tT>;
                MPI_Send(&buffer.at(0), buffer.size(), TypeHelper::Type(), static_cast<int>(target), tag, mComm);
            }

            template<typename tT>
            void Send(size_t target, const tT &buffer, int tag = 0)
            {
                Send(target, &buffer, 1, tag);
            }

            template<typename tT>
            void Receive(size_t source, tT *buffer, size_t count, int tag = 0)
            {
                using TypeHelper = SyncLib::MPI::Types<tT>;
                MPI_Recv(buffer, TypeHelper::Size(count), TypeHelper::Type(), static_cast<int>(source), tag, mComm, MPI_STATUS_IGNORE);
            }

            template<typename tT>
            void Receive(size_t source, arma::Mat<tT> &buffer, int tag = 0)
            {
                using TypeHelper = SyncLib::MPI::Types<tT>;
                const size_t count = buffer.size();
                MPI_Recv(&buffer.at(0), TypeHelper::Size(count), TypeHelper::Type(), static_cast<int>(source), tag, mComm, MPI_STATUS_IGNORE);
            }

            template<typename tT>
            void Receive(size_t source, tT &buffer, int tag = 0)
            {
                Receive(source, &buffer, 1, tag);
            }

            template<typename tT>
            void SendReceive(size_t other, const tT *sendBuffer, size_t sendCount, tT *receiveBuffer, size_t receiveCount,
                             int sendTag = 0, int receiveTag = 0)
            {
                using TypeHelper = SyncLib::MPI::Types<tT>;
                MPI_Sendrecv(sendBuffer, TypeHelper::Size(sendCount), TypeHelper::Type(), static_cast<int>(other), static_cast<int>(sendTag),
                             receiveBuffer, TypeHelper::Size(receiveCount), TypeHelper::Type(), static_cast<int>(other), static_cast<int>(receiveTag),
                             mComm, MPI_STATUS_IGNORE);
            }

            template<typename tT>
            void SendReceive(size_t other, const arma::Mat<tT> &sendBuffer, arma::Mat<tT> &receiveBuffer,
                             int sendTag = 0, int receiveTag = 0)
            {
                using TypeHelper = SyncLib::MPI::Types<tT>;
                const size_t sendCount = sendBuffer.size();
                const size_t receiveCount = receiveBuffer.size();
                MPI_Sendrecv(&sendBuffer.at(0), TypeHelper::Size(sendCount), TypeHelper::Type(), static_cast<int>(other),
                             static_cast<int>(sendTag),
                             &receiveBuffer.at(0), TypeHelper::Size(receiveCount), TypeHelper::Type(), static_cast<int>(other),
                             static_cast<int>(receiveTag), mComm, MPI_STATUS_IGNORE);
            }

            template<typename tT>
            void SendReceive(size_t other, const tT &sendBuffer, tT &receiveBuffer, int sendTag = 0, int receiveTag = 0)
            {
                SendReceive(other, &sendBuffer, 1, &receiveBuffer, 1, sendTag, receiveTag);
            }

            template<typename tT>
            void AllToAllV(const tT *sendBuffer, const int *sendCounts, const int *sendOffsets,
                           tT       *recvBuffer, const int *recvCounts, const int *recvOffsets)
            {
                using TypeHelper = SyncLib::MPI::Types<tT>;
                MPI_Alltoallv(sendBuffer, sendCounts, sendOffsets, TypeHelper::Type(),
                              recvBuffer, recvCounts, recvOffsets, TypeHelper::Type(), mComm);
            }

            template<typename tT>
            void AllToAllV(const arma::Mat<tT> &sendBuffer, const int *sendCounts, const int *sendOffsets,
                           arma::Mat<tT> &recvBuffer, const int *recvCounts, const int *recvOffsets)
            {
                using TypeHelper = SyncLib::MPI::Types<tT>;
                MPI_Alltoallv(&sendBuffer.at(0), sendCounts, sendOffsets, TypeHelper::Type(),
                              &recvBuffer.at(0), recvCounts, recvOffsets, TypeHelper::Type(), mComm);
            }

            template<typename tT>
            void AllToAllV(const arma::Mat<tT> &sendBuffer, const arma::Mat<int> &sendCounts, const arma::Mat<int> &sendOffsets,
                           arma::Mat<tT> &recvBuffer, const arma::Mat<int> &recvCounts, const arma::Mat<int> &recvOffsets)
            {
                using TypeHelper = SyncLib::MPI::Types<tT>;
                MPI_Alltoallv(&sendBuffer.at(0), &sendCounts.at(0), &sendOffsets.at(0), TypeHelper::Type(),
                              &recvBuffer.at(0), &recvCounts.at(0), &recvOffsets.at(0), TypeHelper::Type(), mComm);
            }

            template<typename tT, typename tSize = int>
            void AllToAllV(const arma::Mat<tT> &sendBuffer, const arma::Mat<tSize> &sendCounts, const arma::Mat<tSize> &sendOffsets,
                           arma::Mat<tT> &recvBuffer, const arma::Mat<tSize> &recvCounts, const arma::Mat<tSize> &recvOffsets)
            {
                AllToAllV<tT>(sendBuffer, arma::conv_to<arma::Mat<int>>::from(sendCounts), arma::conv_to<arma::Mat<int>>::from(sendOffsets),
                              recvBuffer, arma::conv_to<arma::Mat<int>>::from(recvCounts), arma::conv_to<arma::Mat<int>>::from(recvOffsets))
            }

            template<typename tT, typename tSize = int>
            void AllToAll(const tT *sendBuffer, tT *recvBuffer, tSize count)
            {
                using TypeHelper = SyncLib::MPI::Types<tT>;
                MPI_Alltoall(sendBuffer, TypeHelper::Size(count), TypeHelper::Type(),
                             recvBuffer, TypeHelper::Size(count), TypeHelper::Type(), mComm);
            }

            template<typename tT, typename tSize = int>
            void AllToAll(tT *buffer, tSize count)
            {
                using TypeHelper = SyncLib::MPI::Types<tT>;
                MPI_Alltoall(MPI_IN_PLACE, TypeHelper::Size(count), TypeHelper::Type(),
                             buffer, TypeHelper::Size(count), TypeHelper::Type(), mComm);
            }

            template<typename tT, typename tSize = int>
            void AllToAll(const arma::Mat<tT> &sendBuffer, arma::Mat<tT> &recvBuffer, tSize count)
            {
                using TypeHelper = SyncLib::MPI::Types<tT>;
                AllToAll(&sendBuffer.at(0), &recvBuffer.at(0), count);
            }

            template<typename tT>
            void AllToAll(const arma::Mat<tT> &sendBuffer, arma::Mat<tT> &recvBuffer)
            {
                AllToAll(sendBuffer, recvBuffer, sendBuffer.size() / Size());
            }

            template<typename tT, typename tSize = int>
            void AllToAll(arma::Mat<tT> &buffer, tSize count)
            {
                AllToAll(&buffer.at(0), count);
            }

            template<typename tT>
            void AllToAll(arma::Mat<tT> &buffer)
            {
                AllToAll(&buffer.at(0), buffer.size() / Size());
            }

            template<typename tT, typename tSize = int>
            void Gather(const tT *sendBuffer, tSize sendCount, tT *recvBuffer, tSize recvCount, tSize receiver)
            {
                using TypeHelper = SyncLib::MPI::Types<tT>;
                MPI_Gather(sendBuffer, TypeHelper::Size(sendCount), TypeHelper::Type(),
                           recvBuffer, TypeHelper::Size(recvCount), TypeHelper::Type(),
                           static_cast<int>(receiver), mComm);
            }

            template<typename tT, typename tSize = int>
            void Gather(const arma::Mat<tT> &sendBuffer, tSize sendCount, arma::Mat<tT> &recvBuffer, tSize recvCount, tSize receiver)
            {
                using TypeHelper = SyncLib::MPI::Types<tT>;
                MPI_Gather(&sendBuffer.at(0), TypeHelper::Size(sendCount), TypeHelper::Type(),
                           &recvBuffer.at(0), TypeHelper::Size(recvCount), TypeHelper::Type(),
                           static_cast<int>(receiver), mComm);
            }

            template<typename tT, typename tSize = int>
            void Gather(const arma::Mat<tT> &sendBuffer, arma::Mat<tT> &recvBuffer, tSize receiver)
            {
                using TypeHelper = SyncLib::MPI::Types<tT>;
                MPI_Gather(&sendBuffer.at(0), TypeHelper::Size(sendBuffer.size()), TypeHelper::Type(),
                           &recvBuffer.at(0), TypeHelper::Size(recvBuffer.size()), TypeHelper::Type(),
                           static_cast<int>(receiver), mComm);
            }

            template<typename tT, typename tSize = int>
            void AllGather(const tT *sendBuffer, tSize sendCount, tT *recvBuffer, tSize recvCount)
            {
                using TypeHelper = SyncLib::MPI::Types<tT>;
                MPI_Allgather(sendBuffer, TypeHelper::Size(sendCount), TypeHelper::Type(),
                              recvBuffer, TypeHelper::Size(recvCount), TypeHelper::Type(), mComm);
            }

            template<typename tT, typename tSize = int>
            void AllGather(const arma::Mat<tT> &sendBuffer, tSize sendCount, arma::Mat<tT> &recvBuffer, tSize recvCount)
            {
                using TypeHelper = SyncLib::MPI::Types<tT>;
                MPI_Allgather(&sendBuffer.at(0), TypeHelper::Size(sendCount), TypeHelper::Type(),
                              &recvBuffer.at(0), TypeHelper::Size(recvCount), TypeHelper::Type(), mComm);
            }

            template<typename tT, typename tSize = int>
            void AllGather(const arma::Mat<tT> &sendBuffer, arma::Mat<tT> &recvBuffer)
            {
                using TypeHelper = SyncLib::MPI::Types<tT>;
                MPI_Allgather(&sendBuffer.at(0), TypeHelper::Size(sendBuffer.size()), TypeHelper::Type(),
                              &recvBuffer.at(0), TypeHelper::Size(sendBuffer.size()), TypeHelper::Type(), mComm);
            }

            template<typename tT, typename tSize = int>
            void Scatter(const tT *sendBuffer, tSize sendCount, tT *recvBuffer, tSize recvCount, tSize sender)
            {
                using TypeHelper = SyncLib::MPI::Types<tT>;
                MPI_Scatter(sendBuffer, TypeHelper::Size(sendCount), TypeHelper::Type(),
                            recvBuffer, TypeHelper::Size(recvCount), TypeHelper::Type(),
                            static_cast<int>(sender), mComm);
            }

            template<typename tT, typename tSize = int>
            void Scatter(const arma::Mat<tT> &sendBuffer, tSize sendCount, arma::Mat<tT> &recvBuffer, tSize recvCount, tSize sender)
            {
                using TypeHelper = SyncLib::MPI::Types<tT>;
                MPI_Scatter(&sendBuffer.at(0), TypeHelper::Size(sendCount), TypeHelper::Type(),
                            &recvBuffer.at(0), TypeHelper::Size(recvCount), TypeHelper::Type(),
                            static_cast<int>(sender), mComm);
            }

            template<typename tT, typename tSize = int>
            void Scatter(const arma::Mat<tT> &sendBuffer, arma::Mat<tT> &recvBuffer, tSize sender)
            {
                using TypeHelper = SyncLib::MPI::Types<tT>;
                MPI_Scatter(&sendBuffer.at(0), TypeHelper::Size(sendBuffer.size()), TypeHelper::Type(),
                            &recvBuffer.at(0), TypeHelper::Size(recvBuffer.size()), TypeHelper::Type(),
                            static_cast<int>(sender), mComm);
            }

            template<typename tT, typename tSize = int>
            void Broadcast(tT *buffer, tSize count, tSize sender)
            {
                using TypeHelper = SyncLib::MPI::Types<tT>;
                MPI_Bcast(buffer, TypeHelper::Size(count), TypeHelper::Type(), static_cast<int>(sender), mComm);
            }

            template<typename tT, typename tSize = int>
            void Broadcast(tT &buffer, tSize sender)
            {
                using TypeHelper = SyncLib::MPI::Types<tT>;
                MPI_Bcast(&buffer, TypeHelper::Size(1), TypeHelper::Type(), static_cast<int>(sender), mComm);
            }

            template<typename tT, typename tSize = int>
            void Broadcast(const arma::Mat<tT> &buffer, tSize sender)
            {
                using TypeHelper = SyncLib::MPI::Types<tT>;
                Broadcast(&buffer.at(0), buffer.size(), sender);
            }

            inline void Barrier()
            {
                MPI_Barrier(mComm);

                if (mRank == 0)
                {
                    for (size_t t = 1, size = Size(); t < size; ++t)
                    {
                        size_t tLength;
                        Receive(t, tLength);

                        if (tLength > 0)
                        {
                            char *buff = mLogBuffer.Reserve(tLength);
                            Receive(t, buff, tLength);
                        }
                    }

                    fwrite(mLogBuffer.Begin(), sizeof(char), mLogBuffer.Size(), stdout);
                    fflush(stdout);
                }
                else
                {
                    const size_t sLength = mLogBuffer.Size();
                    Send(0, sLength);

                    if (sLength > 0)
                    {
                        Send(0, mLogBuffer.Begin(), sLength);
                    }
                }

                mLogBuffer.Clear();
            }

            template<typename tString, typename... tArgs>
            void Print(const tString &format, tArgs &&... args)
            {
                const std::string log = fmt::format(format, std::forward<tArgs>(args)...);
                char *buff = mLogBuffer.Reserve(log.size());
                memcpy(buff, log.data(), log.size());
            }

            const MPI_Comm &GetCommunicator()
            {
                return mComm;
            }

        private:

            inline void InitRankSize()
            {
                {
                    int rank;
                    MPI_Comm_rank(mComm, &rank);
                    mRank = static_cast<size_t>(rank);
                }
                {
                    int size;
                    MPI_Comm_size(mComm, &size);
                    mSize = static_cast<size_t>(size);
                }
            }

            Internal::CommunicationBuffer mLogBuffer;
            MPI_Comm mComm;

            size_t mRank, mSize;
            bool mInitialized;
        };
    }
}
#endif // MPI_VERSION

#endif