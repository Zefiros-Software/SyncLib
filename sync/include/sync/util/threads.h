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
#ifndef __SYNCLIB_THREADS_H__
#define __SYNCLIB_THREADS_H__

#include "preproc/preproc.h"

#ifdef IS_WINDOWS
#endif

#include "hwloc.h"

namespace SyncLibInternal
{
    class ThreadAffinityHelper
    {
    public:
        ThreadAffinityHelper()
        {
            hwloc_topology_init(&mTopology);
            hwloc_topology_load(mTopology);

            mSocketCount = hwloc_get_nbobjs_by_type(mTopology, HWLOC_OBJ_PACKAGE);
            mCoreCount = hwloc_get_nbobjs_by_type(mTopology, HWLOC_OBJ_CORE);
            mThreadCount = hwloc_get_nbobjs_by_type(mTopology, HWLOC_OBJ_PU);
            mCoresPerSocket = mCoreCount / mSocketCount;
            mThreadsPerCore = mThreadCount / mCoreCount;
        }

        void PinThread(size_t s) const
        {
            const int S = static_cast<int>(s) % mThreadCount;

            const int socketId = (S / mCoresPerSocket) % mSocketCount;
            const int coreId = S % mCoresPerSocket;
            const int thrId = (S / mCoresPerSocket) / mSocketCount;
            /*fmt::print("Pinning {} to ({}, {}, {}) of the available ({}, {}, {})\n", s, socketId, coreId, thrId, mSocketCount,
                       mCoresPerSocket, mThreadsPerCore);*/

            hwloc_obj_t socket = hwloc_get_obj_by_type(mTopology, HWLOC_OBJ_PACKAGE, socketId);
            hwloc_obj_t core = hwloc_get_obj_inside_cpuset_by_type(mTopology, socket->cpuset, HWLOC_OBJ_CORE, coreId);
            hwloc_obj_t thr = core->children[thrId];

            hwloc_set_cpubind(mTopology, thr->cpuset, HWLOC_CPUBIND_THREAD);
        }

        int GetCoreCount() const
        {
            return mCoreCount;
        }

        int GetThreadCount() const
        {
            return mThreadCount;
        }

        ~ThreadAffinityHelper()
        {
            hwloc_topology_destroy(mTopology);
        }

    private:
        hwloc_topology_t mTopology{};
        int mSocketCount;
        int mCoreCount;
        int mCoresPerSocket;
        int mThreadCount;
        int mThreadsPerCore;
    };
} // namespace SyncLibInternal

#endif
