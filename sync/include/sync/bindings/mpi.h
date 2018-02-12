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

#include <mpi.h>

#include <stdint.h>

namespace SyncLib
{
    namespace MPI
    {
        template<typename tT>
        struct Types
        {
            static constexpr int Size(size_t count)
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
            static constexpr int Size(size_t count)         \
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

        void Init(int argc, char **argv)
        {
            MPI_Init(&argc, &argv);
        }

        void Finalise()
        {
            MPI_Finalize();
        }

        class Comm
        {
        public:

            Comm(const MPI_Comm &comm)
                : mComm(comm)
            {}

            Comm(int argc, char **argv)
            {
                SyncLib::MPI::Init(argc, argv);
                mComm = MPI_COMM_WORLD;
            }

            ~Comm()
            {
                SyncLib::MPI::Finalise();
            }

            size_t Rank()
            {
                int rank;
                MPI_Comm_rank(mComm, &rank);
                return static_cast<size_t>(rank);
            }

            size_t Size()
            {
                int size;
                MPI_Comm_size(mComm, &size);
                return static_cast<size_t>(size);
            }

            template<typename tT>
            void Send(size_t target, const tT *buffer, size_t count, int tag = 0)
            {
                using TypeHelper = SyncLib::MPI::Types<tT>;
                MPI_Send(buffer, TypeHelper::Size(count), TypeHelper::Type(), target, tag, mComm);
            }

            template<typename tT>
            void Receive(size_t source, tT *buffer, size_t count, int tag = 0)
            {
                using TypeHelper = SyncLib::MPI::Types<tT>;
                MPI_Recv(buffer, TypeHelper::Size(count), TypeHelper::Type(), source, tag, mComm, MPI_STATUS_IGNORE);
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
            void Send(size_t target, const tT &buffer, int tag = 0)
            {
                Send(target, &buffer, 1, tag);
            }

            template<typename tT>
            void Receive(size_t source, tT &buffer, int tag = 0)
            {
                Receive(source, &buffer, 1, tag);
            }

            template<typename tT>
            void SendReceive(size_t other, const tT &sendBuffer, tT &receiveBuffer, int sendTag = 0, int receiveTag = 0)
            {
                SendReceive(other, &sendBuffer, 1, &receiveBuffer, 1, sendTag, receiveTag);
            }

            void Barrier()
            {
                MPI_Barrier(mComm);
            }

        private:

            MPI_Comm mComm;
        };
    }
}
#endif // MPI_VERSION

#endif