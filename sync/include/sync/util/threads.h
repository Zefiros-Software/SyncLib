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

#include <future>
#ifdef _WIN32
#include <windows.h>
#endif

namespace SyncLib
{
    namespace Internal
    {
        inline void PinThread(size_t s)
        {
            int maxS = static_cast<int>(std::thread::hardware_concurrency());
            int core = static_cast<int>(s) % maxS;
#ifdef _WIN32
            DWORD_PTR mask = 1;
            mask = mask << core;
            SetThreadAffinityMask(GetCurrentThread(), mask);
            SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST);
#endif // _WIN32

#if defined(__GNUC__) && !defined(__APPLE__)
            cpu_set_t cpuset;
            CPU_ZERO(&cpuset);
            CPU_SET(core, &cpuset);

            pthread_t currentThread = pthread_self();
            pthread_setaffinity_np(currentThread, sizeof(cpu_set_t), &cpuset);
#endif
        }
    }
}

#endif