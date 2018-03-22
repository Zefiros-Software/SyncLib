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
#ifndef __SYNCLIB_ABSTRACTSHAREDVARIABLE_H__
#define __SYNCLIB_ABSTRACTSHAREDVARIABLE_H__

#include "sync/buffers/put.h"
#include "sync/util/dataType.h"

namespace SyncLib
{
    namespace Internal
    {
        template<typename tEnv>
        class AbstractSharedVariable
        {
        public:

            using tEnvironmentType = tEnv;

            AbstractSharedVariable(tEnv &env)
                : mEnv(env)
            {
                mIndex = mEnv.RegisterSharedVariable(this);
            }

            virtual ~AbstractSharedVariable()
            {
                mEnv.DisableSharedVariable(mIndex);
            }

        protected:

            size_t mIndex;
            tEnv &mEnv;

            template<typename tT>
            char *WriteData(char *cursor, const tT &data)
            {
                *reinterpret_cast<tT *>(cursor) = data;
                return cursor + sizeof(tT);
            }

            template<typename tT, typename... tArgs>
            char *WriteData(char *cursor, const tT &data, tArgs &&...args)
            {
                return WriteData(WriteData(cursor, data), std::forward<tArgs>(args)...);
            }

            template<typename tIterator>
            char *WriteIter(char *cursor, tIterator beginIt, tIterator endIt)
            {
                using tT = typename tIterator::value_type;

                tT *cursorT = reinterpret_cast<tT *>(cursor);
                std::copy(beginIt, endIt, cursorT);
                return cursor + (endIt - beginIt) * sizeof(tT);
            }

            CommunicationBuffer &GetTargetPutBuffer(size_t target)
            {
                return mEnv.GetTargetPutBuffer(target);
            }

            CommunicationBuffer &GetTargetGetBuffer(size_t target)
            {
                return mEnv.GetTargetGetBuffer(target);
            }

            CommunicationBuffer &GetTargetGetRequests(size_t target)
            {
                return mEnv.GetTargetGetRequests(target);
            }

            CommunicationBuffer &GetTargetGetDestinations(size_t target)
            {
                return mEnv.GetTargetGetDestinations(target);
            }
        };
    }
}

#endif