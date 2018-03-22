#pragma once
#ifndef __SYNCLIB_SENDQUEUE_H__
#define __SYNCLIB_SENDQUEUE_H__
#include "sync/buffers/put.h"

#include <algorithm>

namespace SyncLib
{
    template<typename tT>
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

    template<typename tT>
    struct CursorHelper<std::vector<tT>>
    {
        using type = std::vector<tT>;

        static type Read(const char *cursor)
        {
            size_t size = *reinterpret_cast<size_t *>(cursor);
            tT *tCursor = reinterpret_cast<tT *>(cursor + sizeof(size_t));
            return std::vector<tT>(tCursor, tCursor + size);
        }

        static char *Write(char *cursor, const type &v)
        {
            cursor = CursorHelper<size_t>::Write(cursor, v.size());
            tT *tCursor = reinterpret_cast<tT *>(cursor + sizeof(size_t));
            std::copy(v.begin(), v.end(), tCursor);
            return cursor + v.size() * sizeof(tT);
        }

        static constexpr size_t Size(const tT &v)
        {
            return sizeof(size_t) + v.size() * sizeof(tT);
        }
    };

    template<typename tT, typename... tArgs>
    struct CursorReader
    {
        using tHelper = CursorHelper<tT>;

        template<typename... tArgsDone>
        static std::tuple<tArgsDone..., tT, tArgs...> Read(const char *cursor, size_t &size, tArgsDone &&... done)
        {
            tT value = tHelper::Read(cursor);
            const size_t diff = tHelper::Size(value);
            size += diff;
            return CursorReader<tArgs...>::Read(cursor + diff, size, std::forward<tArgsDone>(done)..., value);
        }
    };

    template<typename tT>
    struct CursorReader<tT>
    {
        using tHelper = CursorHelper<tT>;

        template<typename... tArgsDone>
        static std::tuple<tArgsDone..., tT> Read(const char *cursor, size_t &size, tArgsDone &&... done)
        {
            tT value = tHelper::Read(cursor);
            size += tHelper::Size(value);
            return std::tuple<tArgsDone..., tT>(std::forward<tArgsDone>(done)..., value);
        }
    };

    template<typename tT, typename... tArgs>
    struct CursorWriter
    {
        static char *Write(char *cursor, const tT &v, const tArgs &... args)
        {
            using tHelper = CursorHelper<tT>;
            return CursorWriter<tArgs...>::Write(tHelper::Write(cursor, v), args...);
        }
    };

    template<typename tT>
    struct CursorWriter<tT>
    {
        static char *Write(char *cursor, const tT &v)
        {
            using tHelper = CursorHelper<tT>;
            return tHelper::Write(cursor, v);
        }
    };

    template<typename tEnvironment>
    class AbstractSendQueue
    {
    public:

        using tEnv = tEnvironment;

        AbstractSendQueue(tEnv &env)
            : mSendBuffers(env.Size()),
              mSendCounts(env.Size(), 0),
              mReceiveCount(0),
              mEnv(env)
        {
            mIndex = mEnv.RegisterSendQueue(this);
        }

        virtual ~AbstractSendQueue()
        {
            mEnv.DisableSendQueue(mIndex);
        }

        char *ReserveReceiveSpace(size_t count, size_t size)
        {
            mReceiveCount += count;
            return mReceiveBuffer.Reserve(size);
        }

        void Receive(size_t count, const Internal::CommunicationBuffer &buffer)
        {
            if (count == 0)
            {
                return;
            }

            char *cursor = ReserveReceiveSpace(count, buffer.Size());
            std::copy(buffer.Begin(), buffer.End(), cursor);
        }

        const Internal::CommunicationBuffer &GetTargetBuffer(size_t t) const
        {
            return mSendBuffers[t];
        }

        size_t GetTargetCount(size_t t) const
        {
            return mSendCounts[t];
        }

        void ClearSendBuffers()
        {
            for (auto &buffer : mSendBuffers)
            {
                buffer.Clear();
            }

            std::fill(mSendCounts.begin(), mSendCounts.end(), 0);
        }

        void ClearReceiveBuffer()
        {
            mReceiveBuffer.Clear();
            mReceiveCount = 0;
        }

    protected:

        std::vector<Internal::CommunicationBuffer> mSendBuffers;
        std::vector<size_t> mSendCounts;

        Internal::CommunicationBuffer mReceiveBuffer;
        tEnv &mEnv;
        size_t mReceiveCount;
        size_t mIndex;
        size_t mRank;
    };

    template<typename tEnvironment, typename... tValues>
    class SendQueue
        : public AbstractSendQueue<tEnvironment>
    {
    public:

        using tParent = AbstractSendQueue<tEnv>;

        SendQueue(tEnv &env)
            : tParent(env)
        {}

        void Send(size_t t, const tValues &... values)
        {
            const size_t size = (... + CursorHelper<tValues>::Size(values));
            char *cursor = tParent::mSendBuffers[t].Reserve(size);
            CursorWriter<tValues...>::Write(cursor, values...);
            ++mSendCounts[t];
        }

        class SendQueueIterator
        {
        public:

            SendQueueIterator(const char *cursor)
                : mCursor(cursor),
                  mCurrentSize(0)
            {}

            std::tuple<tValues...> operator*()
            {
                mCurrentSize = 0;
                return CursorReader<tValues...>::Read(mCursor, mCurrentSize);
            }

            bool operator !=(const SendQueueIterator &other) const
            {
                return mCursor != other.mCursor;
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

        private:

            const char *mCursor;
            size_t mCurrentSize;
        };

        using iterator = SendQueueIterator;

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
}
#endif