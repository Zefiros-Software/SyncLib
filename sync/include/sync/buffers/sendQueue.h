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
#ifndef __SYNCLIB_SENDQUEUE_H__
#define __SYNCLIB_SENDQUEUE_H__
#include "sync/buffers/put.h"

#include <algorithm>
#include <iterator>

namespace SyncLibInternal
{
    template <typename tT>
    struct CursorHelper
    {
        using type = const tT &;

        static type Read(const char *cursor)
        {
            return *reinterpret_cast<const tT *>(cursor);
        }

        static char *Write(char *cursor, type v)
        {
            *reinterpret_cast<tT *>(cursor) = v;
            return cursor + Size(v);
        }

        static constexpr size_t Size(const tT &)
        {
            return sizeof(tT);
        }
    };

    template <typename tT>
    struct CursorHelper<std::vector<tT>>
    {
        using type = std::vector<tT>;

        static type Read(const char *cursor)
        {
            const size_t size = *reinterpret_cast<const size_t *>(cursor);
            tT *cursorT = reinterpret_cast<tT *>(cursor + sizeof(size_t));
            return std::vector<tT>(cursorT, cursorT + size);
        }

        static char *Write(char *cursor, const type &v)
        {
            cursor = CursorHelper<size_t>::Write(cursor, v.size());
            tT *cursorT = reinterpret_cast<tT *>(cursor + sizeof(size_t));
            std::copy(v.begin(), v.end(), cursorT);
            return cursor + v.size() * sizeof(tT);
        }

        static constexpr size_t Size(const tT &v)
        {
            return sizeof(size_t) + v.size() * sizeof(tT);
        }
    };

    template <typename tT, typename... tArgs>
    struct CursorReader
    {
        using tHelper = CursorHelper<tT>;

        template <typename... tArgsDone>
        static std::tuple<tArgsDone..., typename tHelper::type, tArgs...> Read(const char *cursor, size_t &size, tArgsDone &&... done)
        {
            typename tHelper::type value = tHelper::Read(cursor);
            const size_t diff = tHelper::Size(value);
            size += diff;
            return CursorReader<tArgs...>::template Read<tArgsDone..., typename tHelper::type>(cursor + diff, size,
                                                                                               std::forward<tArgsDone>(done)..., value);
        }
    };

    template <typename tT>
    struct CursorReader<tT>
    {
        using tHelper = CursorHelper<tT>;

        template <typename... tArgsDone>
        static std::tuple<tArgsDone..., tT> Read(const char *cursor, size_t &size, tArgsDone &&... done)
        {
            typename tHelper::type value = tHelper::Read(cursor);
            size += tHelper::Size(value);
            return std::tuple<tArgsDone..., typename tHelper::type>(std::forward<tArgsDone>(done)..., value);
        }
    };

    template <typename tT, typename... tArgs>
    struct CursorWriter
    {
        static char *Write(char *cursor, const tT &v, const tArgs &... args)
        {
            using tHelper = CursorHelper<tT>;
            return CursorWriter<tArgs...>::Write(tHelper::Write(cursor, v), args...);
        }
    };

    template <typename tT>
    struct CursorWriter<tT>
    {
        static char *Write(char *cursor, const tT &v)
        {
            using tHelper = CursorHelper<tT>;
            return tHelper::Write(cursor, v);
        }
    };

    struct EnvHelper
    {
        template <typename tEnv>
        static void DisableSendQueue(tEnv &env, size_t index)
        {
            env.DisableSendQueue(index);
        }

        template <typename tEnv, typename tSendQueue>
        static size_t RegisterSendQueue(tEnv &env, tSendQueue *queue)
        {
            return env.RegisterSendQueue(queue);
        }

        template <typename tBuffer, typename tEnv>
        static tBuffer &GetGetRequests(tEnv &env)
        {
            return env.mGetRequests;
        }

        template <typename tBuffer, typename tEnv>
        static tBuffer &GetGetBuffers(tEnv &env)
        {
            return env.mGetBuffers;
        }

        template <typename tBuffer, typename tEnv>
        static tBuffer &GetPutBuffers(tEnv &env)
        {
            return env.mPutBuffers;
        }

        template <typename tEnv>
        static auto &GetSendInfo(tEnv &env)
        {
            return env.mSendQueueInfo;
        }

        template <typename tEnv>
        static std::vector<typename tEnv::AbstractSendQueue *> &GetSendQueues(tEnv &env)
        {
            return env.mSendQueues;
        }

        template <typename tEnv>
        static std::vector<typename tEnv::AbstractSharedVariable *> &GetVariables(tEnv &env)
        {
            return env.mVariables;
        }

        template <typename tEnv>
        static typename tEnv::tBackendType &GetBackend(tEnv &env)
        {
            return env.mBackend;
        }
    };

    template <typename tEnvironment>
    class AbstractSendQueue
    {
    public:
        using tEnv = tEnvironment;
        using tBuffer = SyncLibInternal::CommunicationBuffer;

        AbstractSendQueue(tEnv &env)
            : mSendBuffers(env.Size())
            , mSendCounts(env.Size(), 0)
            , mSendSizes(env.Size(), 0)
            , mReceiveBuffer(1024 * 256)
            , mEnv(env)
            , mReceiveCount(0)
        {
            mIndex = EnvHelper::RegisterSendQueue(env, this);
        }

        virtual ~AbstractSendQueue()
        {
            EnvHelper::DisableSendQueue(mEnv, mIndex);
        }

        char *ReserveReceiveSpace(size_t count, size_t size)
        {
            mReceiveCount += count;
            return mReceiveBuffer.Reserve(size);
        }

        void Receive(size_t count, const tBuffer &buffer)
        {
            if (count == 0)
            {
                return;
            }

            char *cursor = ReserveReceiveSpace(count, buffer.Size());
            std::copy_n(buffer.Begin(), buffer.Size(), cursor);
        }

        const tBuffer &GetTargetBuffer(size_t t) const
        {
            return mSendBuffers[t];
        }

        size_t GetTargetCount(size_t t) const
        {
            return mSendCounts[t];
        }

        size_t GetTargetSize(size_t t) const
        {
            return mSendSizes[t];
        }

        void ClearSendBuffers()
        {
            for (auto &buffer : mSendBuffers)
            {
                buffer.Clear();
            }

            std::fill(mSendCounts.begin(), mSendCounts.end(), 0);
            std::fill(mSendSizes.begin(), mSendSizes.end(), 0);
        }

        void ClearReceiveBuffer()
        {
            mReceiveBuffer.Clear();
            mReceiveCount = 0;
        }

        void SetIndex(size_t index)
        {
            mIndex = index;
        }

    protected:
        std::vector<tBuffer> mSendBuffers;
        std::vector<size_t> mSendCounts;
        std::vector<size_t> mSendSizes;

        tBuffer mReceiveBuffer;
        tEnv &mEnv;
        size_t mReceiveCount;
        size_t mIndex;
    };

    template <typename tT>
    constexpr bool IsNumeric = std::is_integral_v<tT> || std::is_floating_point_v<tT>;

    template <typename tT>
    using tSendQueueElement = typename std::conditional<IsNumeric<tT>, const tT &, tT>::type;

    template <typename tT, typename... tValues>
    struct SendQueueElementHelper
    {
        template <template <typename...> typename tTemplate, typename... tDone>
        using tTemplated = typename SendQueueElementHelper<tValues...>::template tTemplated<tTemplate, tDone..., tSendQueueElement<tT>>;
    };

    template <typename tT>
    struct SendQueueElementHelper<tT>
    {
        template <template <typename...> typename tTemplate, typename... tDone>
        using tTemplated = tTemplate<tDone..., tSendQueueElement<tT>>;
        // SendQueueElementHelper<tValues...>::template Templated<tTemplate, tDone..., tSendQueueElement<tT>>;
    };

    //     template<typename tT, typename... tValues>
    //     struct SendQueueIteratorHelper
    //     {
    //         using tHelper = SendQueueIteratorHelper<tValues...>;
    //
    //         template<typename... tDone>
    //         using tIterator = tHelper::tIterator<tDone..., tSendQueueElement<tT>>;
    //     };
    //
    //     template<typename tT>
    //     struct SendQueueIteratorHelper
    //     {
    //         template<typename... tDone>
    //         using tIterator = std::tuple<tDone..., tSendQueueElement<tT>>;
    //     };
    //
    //     template<typename tT, typename... tValues>
    //     using tSendQueueIteratorValue = SendQueueIteratorHelper<tValues...>::tIterator<tSendQueueElement<tT>>;

    //     template<typename... tArgs>
    //     struct SendQueueIteratorValueHelper
    //     {
    //         template<typename tT>
    //         using Extend =
    //             std::conditional_t<IsNumeric<tT>, SendQueueIteratorValueHelper<tArgs..., tT &>, SendQueueIteratorValueHelper<tArgs..., tT>>;
    //
    //         template<template<typename...>typename tTemplate>
    //         using Templated = tTemplate<const tArgs...>;
    //     };
    //
    //     template<typename tSendQueueIteratorValueHelper, typename tS, typename... tArgs>
    //     struct SendQueueIteratorValue
    //     {
    //         using type = typename SendQueueIteratorValue<typename tSendQueueIteratorValueHelper::template Extend<tS>, tArgs...>::type;
    //     };
    //
    //     template<typename tSendQueueIteratorValueHelper, typename tS>
    //     struct SendQueueIteratorValue<tSendQueueIteratorValueHelper, tS>
    //     {
    //         using type = typename tSendQueueIteratorValueHelper::template Extend<tS>;
    //     };
    //
    //     template<typename tT, typename... tArgs>
    //     struct SendQueueIteratorValueInstance
    //     {
    //         using type = typename SendQueueIteratorValue<SendQueueIteratorValueHelper<tSendQueueElement<tT>>, tArgs...>::type;
    //     };
    //
    //     template<typename tT>
    //     struct SendQueueIteratorValueInstance<tT>
    //     {
    //         using type = SendQueueIteratorValueHelper<tSendQueueElement<tT>>;
    //     };
    //
    //     template<typename... tArgs>
    //     using tSendQueueIteratorValue = typename SendQueueIteratorValueInstance<tArgs...>::type::template Templated<std::tuple>;

    template <typename tT, bool tIsNumeric = IsNumeric<tT>>
    struct CustomCursorHelper
    {
        static size_t Size(const tT &)
        {
            static_assert(tIsNumeric, "This should only be called by numeric types. To fix this, create a custom specialisation.");
            return sizeof(tT);
        }

        static const tT &Read(const char *&cursor, size_t &size)
        {
            // fmt::print("Reading normal value\n");
            const tT *value = reinterpret_cast<const tT *>(cursor);
            size += Size(*value);
            cursor += Size(*value);

            return *value;
        }

        static char *Write(char *cursor, const tT &v)
        {
            *reinterpret_cast<tT *>(cursor) = v;
            return cursor + Size(v);
        }
    };

    template <typename tT>
    struct ArmaMatCursorHelper
    {
        using tMat = arma::Mat<tT>;

        static size_t Size(const tMat &mat)
        {
            return 2 * sizeof(size_t) + mat.n_elem * sizeof(tT);
        }

        static tMat Read(const char *&cursor, size_t &size)
        {
            const size_t *cursorSizeT = reinterpret_cast<const size_t *>(cursor);
            const size_t rows = cursorSizeT[0];
            const size_t cols = cursorSizeT[1];

            const tT *cursorT = reinterpret_cast<const tT *>(cursor + size + 2 * sizeof(size_t));
            size += 2 * sizeof(size_t) + rows * cols * sizeof(tT);
            cursor += 2 * sizeof(size_t) + rows * cols * sizeof(tT);

            return tMat(const_cast<tT *>(cursorT), static_cast<arma::uword>(rows), static_cast<arma::uword>(cols), false, true);
        }

        static char *Write(char *cursor, const tMat &mat)
        {
            size_t *cursorSizeT = reinterpret_cast<size_t *>(cursor);
            cursorSizeT[0] = mat.n_rows;
            cursorSizeT[1] = mat.n_cols;

            tT *cursorT = reinterpret_cast<tT *>(cursor + 2 * sizeof(size_t));
            std::copy_n(&mat(0, 0), mat.n_elem, cursorT);

            return cursor + Size(mat);
        }
    };

    template <typename tT, typename tVec>
    struct ArmaVecCursorHelper
    {
        static size_t Size(const tVec &vec)
        {
            return sizeof(size_t) + vec.n_elem * sizeof(tT);
        }

        static tVec Read(const char *&cursor, size_t &size)
        {
            // fmt::print("Reading vector\n");
            const size_t *cursorSizeT = reinterpret_cast<const size_t *>(cursor);
            const size_t elems = cursorSizeT[0];

            const tT *cursorT = reinterpret_cast<const tT *>(cursor + sizeof(size_t));
            size += sizeof(size_t) + elems * sizeof(tT);
            cursor += sizeof(size_t) + elems * sizeof(tT);

            return tVec(const_cast<tT *>(cursorT), static_cast<arma::uword>(elems), false, true);
        }

        static char *Write(char *cursor, const tVec &vec)
        {
            size_t *cursorSizeT = reinterpret_cast<size_t *>(cursor);
            cursorSizeT[0] = vec.n_elem;

            tT *cursorT = reinterpret_cast<tT *>(cursor + sizeof(size_t));
            std::copy_n(&vec(0), vec.n_elem, cursorT);

            return cursor + Size(vec);
        }
    };

    template <typename tT>
    struct CustomCursorHelper<arma::Mat<tT>, false> : public ArmaMatCursorHelper<tT>
    {
    };

    template <typename tT>
    struct CustomCursorHelper<arma::Col<tT>, false> : public ArmaVecCursorHelper<tT, arma::Col<tT>>
    {
    };

    template <typename tT>
    struct CustomCursorHelper<arma::Row<tT>, false> : public ArmaVecCursorHelper<tT, arma::Row<tT>>
    {
    };

    template <typename tT, typename... tValues>
    struct ReverseHelper
    {
        template <template <typename...> typename tTemplate, typename... tDone>
        using tTemplated = typename ReverseHelper<tValues...>::template Templated<tTemplate, tT, tDone...>;
    };

    template <typename tT>
    struct ReverseHelper<tT>
    {
        template <template <typename...> typename tTemplate, typename... tDone>
        using tTemplated = tTemplate<tT, tDone...>;
    };

    template <typename tValueReversed, typename tValue, size_t... tI>
    tValue ReverseTupleImpl(const tValueReversed &t, std::index_sequence<tI...>)
    {
        return tValue(std::get < sizeof...(tI) - 1 - tI > (t)...);
    }

    template <typename tValueReversed, typename tValue>
    tValue ReverseTuple(const tValueReversed &t)
    {
        return ReverseTupleImpl<tValueReversed, tValue>(t, std::make_index_sequence<std::tuple_size_v<tValueReversed>>());
    }

    //  template<typename... tValues>
    //  struct CustomCursorReader
    //  {
    //      using tValue = std::tuple<typename tSendQueueElement<tValues>...>;
    //
    //      static tValue Read(const char *cursor, size_t &size)
    //      {
    //          return tValue(CustomCursorHelper<tValues>::Read(cursor, size)...);
    //      }
    //  };

    //     template<typename>
    //     struct CustomCursorReader;
    //
    //     template<typename... tValues>
    //     struct CustomCursorReader<std::tuple<tValues...>>
    //     {
    //         template<typename tValue, typename tValueReversed>
    //         static tValue Read(const char *cursor, size_t &size)
    //         {
    //             return ReverseTuple<tValueReversed, tValue>(tValueReversed(CustomCursorHelper<tValues>::Read(cursor, size)...));
    //         }
    //     };

    template <typename tT, typename... tArgs>
    struct CustomCursorWriter
    {
        using tHelper = CustomCursorHelper<tT>;

        static char *Write(char *cursor, const tT &v, const tArgs &... args)
        {
            return CustomCursorWriter<tArgs...>::Write(tHelper::Write(cursor, v), args...);
        }
    };

    template <typename tT>
    struct CustomCursorWriter<tT>
    {
        using tHelper = CustomCursorHelper<tT>;

        static char *Write(char *cursor, const tT &v)
        {
            return tHelper::Write(cursor, v);
        }
    };

    //  template<typename tValue, typename tT, typename... tArgs>
    //  struct CustomCursorReader
    //  {
    //      using tHelper = CustomCursorHelper<tT>;
    //
    //      template<typename... tDone>
    //      FORCEINLINE static tValue Read(const char *cursor, size_t &size, tDone... done)
    //      {
    //          tSendQueueElement<tT>value = tHelper::Read(cursor, size);
    //          return CustomCursorReader<tValue, tArgs...>::template Read<tDone..., tSendQueueElement<tT>>(cursor, size, done..., value);
    //      }
    //  };
    //
    //  template<typename tValue, typename tT>
    //  struct CustomCursorReader<tValue, tT>
    //  {
    //      using tHelper = CustomCursorHelper<tT>;
    //
    //      template<typename... tDone>
    //      FORCEINLINE static tValue Read(const char *cursor, size_t &size, tDone... done)
    //      {
    //          tSendQueueElement<tT>value = tHelper::Read(cursor, size);
    //          return tValue(done..., value);
    //      }
    //  };

    template <typename tValue, typename tT, typename... tArgs>
    struct CustomCursorReader
    {
        using tHelper = CustomCursorHelper<tT>;

        template <typename... tDone>
        FORCEINLINE static tValue Read(const char *cursor, size_t &size, tDone &&... done)
        {
            tSendQueueElement<tT> value = tHelper::Read(cursor, size);
            return CustomCursorReader<tValue, tArgs...>::Read(cursor, size, std::forward<tDone>(done)..., value);
        }
    };

    template <typename tValue, typename tT, typename... tArgs>
    struct CustomCursorReader<tValue, arma::Col<tT>, tArgs...>
    {
        using tHelper = CustomCursorHelper<arma::Col<tT>>;

        template <typename... tDone>
        FORCEINLINE static tValue Read(const char *cursor, size_t &size, tDone &&... done)
        {
            arma::Col<tT> value = tHelper::Read(cursor, size);
            return CustomCursorReader<tValue, tArgs...>::Read(cursor, size, std::forward<tDone>(done)..., std::move(value));
        }
    };

    template <typename tValue, typename tT>
    struct CustomCursorReader<tValue, tT>
    {
        using tHelper = CustomCursorHelper<tT>;

        template <typename... tDone>
        FORCEINLINE static tValue Read(const char *cursor, size_t &size, tDone &&... done)
        {
            return tValue(std::forward<tDone>(done)..., tHelper::Read(cursor, size));
        }
    };

    template <typename tValue, typename tT>
    struct CustomCursorReader<tValue, arma::Col<tT>>
    {
        using tHelper = CustomCursorHelper<arma::Col<tT>>;

        template <typename... tDone>
        FORCEINLINE static tValue Read(const char *cursor, size_t &size, tDone &&... done)
        {
            return tValue(std::forward<tDone>(done)..., tHelper::Read(cursor, size));
        }
    };

    template <typename... tValues>
    class SendQueueIterator //: public std::iterator<std::forward_iterator_tag, std::tuple<tSendQueueElement<tValues>...>>
    {
    public:
        SendQueueIterator(const char *cursor)
            : mCursor(cursor)
            , mCurrentSize(0)
        {
        }

        using tValue = typename SendQueueElementHelper<tValues...>::template tTemplated<std::tuple>;

        FORCEINLINE tValue operator*()
        {
            mCurrentSize = 0;
            return CustomCursorReader<tValue, tValues...>::Read(mCursor, mCurrentSize);
        }

        SendQueueIterator &operator++()
        {
            if (mCurrentSize == 0)
            {
                this->operator*();
            }

            mCursor += mCurrentSize;
            mCurrentSize = 0;
            return *this;
        }

        bool operator!=(const SendQueueIterator &other)
        {
            return other.mCursor != mCursor;
        }

        bool operator==(const SendQueueIterator &other)
        {
            return other.mCursor == mCursor;
        }

    private:
        const char *mCursor;
        size_t mCurrentSize;
    };

    template < typename tT, bool tIsNumeric = std::is_integral_v<tT> || std::is_floating_point_v<tT >>
    class SingleValueSendQueueIterator : public std::iterator<std::forward_iterator_tag, tT>
    {
    public:
        SingleValueSendQueueIterator(const char *cursor)
            : mCursor(cursor)
        {
        }

        const tT &operator*()
        {
            return *reinterpret_cast<const tT *>(mCursor);
        }

        SingleValueSendQueueIterator &operator++()
        {
            mCursor += sizeof(tT);
            return *this;
        }

        bool operator!=(const SingleValueSendQueueIterator &other)
        {
            return other.mCursor != mCursor;
        }

        bool operator==(const SingleValueSendQueueIterator &other)
        {
            return other.mCursor == mCursor;
        }

    private:
        const char *mCursor;
    };

    template <typename tT>
    class SingleValueSendQueueIterator<arma::Mat<tT>, false>
    {
    public:
        SingleValueSendQueueIterator(const char *cursor)
            : mCursor(cursor)
            , mCurrentSize(0)
        {
        }

        arma::Mat<tT> operator*()
        {
            mCurrentSize = 0;

            // Duplicate the pointer: it is modified by the Read
            const char *cursor = mCursor;
            return ArmaMatCursorHelper<tT>::Read(cursor, mCurrentSize);
        }

        SingleValueSendQueueIterator &operator++()
        {
            if (mCurrentSize == 0)
            {
                this->operator*();
            }

            mCursor += mCurrentSize;
            mCurrentSize = 0;
            return *this;
        }

        bool operator!=(const SingleValueSendQueueIterator &other)
        {
            return other.mCursor != mCursor;
        }

        bool operator==(const SingleValueSendQueueIterator &other)
        {
            return other.mCursor == mCursor;
        }

    private:
        const char *mCursor;
        size_t mCurrentSize;
    };

    template <typename tEnvironment, typename tT, typename... tValues>
    class SendQueue : public AbstractSendQueue<tEnvironment>
    {
    public:
        using tParent = AbstractSendQueue<tEnvironment>;
        using tEnv = typename tParent::tEnv;

        SendQueue(tEnv &env)
            : tParent(env)
        {
        }

        void Send(const size_t t, const tT &value1, const tValues &... values)
        {
            const std::make_signed_t<size_t> size = CustomCursorHelper<tT>::Size(value1) + (... + CustomCursorHelper<tValues>::Size(values));
            char *cursor = tParent::mSendBuffers[t].Reserve(size);
            char *endCursor = CustomCursorWriter<tT, tValues...>::Write(cursor, value1, values...);
            // char *endCursor = CursorWriter<tT, tValues...>::Write(cursor, value1, values...);
            Assert(endCursor - cursor == size, "Written beyond reserved size");
            ++tParent::mSendCounts[t];
            tParent::mSendSizes[t] += size;
        }

        void Broadcast(const tT &value1, const tValues &... values)
        {
            for (size_t t = 0, p = tParent::mEnv.Size(); t < p; ++t)
            {
                Send(t, value1, values...);
            }
        }

        //         class SendQueueIterator
        //         {
        //         public:
        //             SendQueueIterator(const char *cursor)
        //                 : mCursor(cursor)
        //                 , mCurrentSize(0)
        //             {
        //             }
        //
        //             std::tuple<tT, tValues...>operator*()
        //             {
        //                 mCurrentSize = 0;
        //                 return CursorReader<tT, tValues...>::Read(mCursor, mCurrentSize);
        //             }
        //
        //             bool operator!=(const SendQueueIterator &other) const
        //             {
        //                 return mCursor != other.mCursor;
        //             }
        //
        //             bool operator==(const SendQueueIterator &other) const
        //             {
        //                 return mCursor == other.mCursor;
        //             }
        //
        //             SendQueueIterator &operator++()
        //             {
        //                 if (mCurrentSize == 0)
        //                 {
        //                     this->operator*();
        //                 }
        //
        //                 mCursor += mCurrentSize;
        //                 mCurrentSize = 0;
        //                 return *this;
        //             }
        //
        //         private:
        //             const char *mCursor;
        //             size_t mCurrentSize;
        //         };

        using iterator = SendQueueIterator<tT, tValues...>;

        iterator begin()
        {
            return iterator(tParent::mReceiveBuffer.Begin());
        }

        iterator end()
        {
            return iterator(tParent::mReceiveBuffer.End());
        }

        size_t size() const
        {
            return tParent::mReceiveCount;
        }
    };

    template <typename tEnvironment, typename tT>
    class SendQueue<tEnvironment, tT> : public AbstractSendQueue<tEnvironment>
    {
    public:
        using tParent = AbstractSendQueue<tEnvironment>;
        using tEnv = typename tParent::tEnv;

        SendQueue(tEnv &env)
            : tParent(env)
        {
        }

        void Send(size_t t, const tT &value)
        {
            using tCursorHelper = CustomCursorHelper<tT>;
            const std::make_signed_t<size_t> size = tCursorHelper::Size(value);
            char *cursor = tParent::mSendBuffers[t].Reserve(size);

            char *endCursor = tCursorHelper::Write(cursor, value);
            Assert(endCursor - cursor == size, "Written beyond reserved size");

            ++tParent::mSendCounts[t];
            tParent::mSendSizes[t] += size;
        }

        void Broadcast(const tT &value)
        {
            for (size_t t = 0, p = tParent::mEnv.Size(); t < p; ++t)
            {
                Send(t, value);
            }
        }

        using iterator = SingleValueSendQueueIterator<tT>;

        iterator begin()
        {
            return iterator(tParent::mReceiveBuffer.Begin());
        }

        iterator end()
        {
            return iterator(tParent::mReceiveBuffer.End());
        }

        size_t size() const
        {
            return tParent::mReceiveCount;
        }

        void ToVector(std::vector<tT> &buff)
        {
            buff.resize(tParent::mReceiveCount);
            std::copy(begin(), end(), buff.begin());
        }

        auto ToVector()
        {
            std::vector<tT> buff;
            ToVector(buff);
            return std::move(buff);
        }
    };
} // namespace SyncLibInternal
#endif
