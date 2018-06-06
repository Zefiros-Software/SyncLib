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
#ifndef __SYNC_SYNC_H__
#define __SYNC_SYNC_H__

#include "sync/env/backend/mpi.h"
#include "sync/env/backend/shared.h"
#include "sync/env/base.h"

namespace SyncLib
{
    namespace Environments
    {
        using SharedMemoryBSP = BaseBSP<SyncLibInternal::SharedMemoryBackend>;
        using DistributedBSP = BaseBSP<SyncLibInternal::MPIBackend>;
        struct NoBSP
        {
            template<typename... tArgs>
            NoBSP(tArgs &&...)
            {}

            constexpr size_t Rank()
            {
                return 0;
            }

            constexpr void Barrier()
            {}

            template <typename tFunc, typename... tArgs>
            inline void Run(const tFunc &f, tArgs &&... args)
            {
                f(*this, std::forward<tArgs>(args)...);
            }
        };
    } // namespace Environments
} // namespace SyncLib

#endif
