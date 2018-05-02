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

#include <future>
#ifdef IS_WINDOWS
#include <windows.h>
#endif

#include <set>
#include <thread>

namespace SyncLib
{
    namespace Util
    {

        // Taken from boost
        inline unsigned PhysicalConcurrency() noexcept
        {
#if defined(IS_WINDOWS)
            unsigned cores = 0;
            DWORD size = 0;

            GetLogicalProcessorInformation(nullptr, &size);

            if (ERROR_INSUFFICIENT_BUFFER != GetLastError())
            {
                return 0;
            }

            std::vector<SYSTEM_LOGICAL_PROCESSOR_INFORMATION> buffer(size);

            if (GetLogicalProcessorInformation(&buffer.front(), &size) == FALSE)
            {
                return 0;
            }

            const size_t elements = size / sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION);

            for (size_t i = 0; i < elements; ++i)
            {
                if (buffer[i].Relationship == RelationProcessorCore)
                {
                    ++cores;
                }
            }

            return cores;
#elif defined(IS_LINUX)

            try
            {
                using namespace std;

                ifstream proc_cpuinfo("/proc/cpuinfo");

                const string_view physical_id("physical id"), core_id("core id");

                typedef std::pair<unsigned, unsigned> core_entry; // [physical ID, core id]

                std::set<core_entry> cores;

                core_entry current_core_entry;

                string line;

                while (getline(proc_cpuinfo, line))
                {
                    if (line.empty())
                    {
                        continue;
                    }

                    vector<string_view> key_val(::SyncLib::Util::Split(line, ':'));
                    // boost::split(key_val, line, boost::is_any_of(":"));

                    if (key_val.size() != 2)
                        // return hardware_concurrency();
                    {
                        continue;
                    }

                    // string key = key_val[0];
                    // string value = key_val[1];
                    // boost::trim(key);
                    // boost::trim(value);
                    string key(::SyncLib::Util::Trim(key_val[0]));
                    string value(::SyncLib::Util::Trim(key_val[1]));

                    if (key == physical_id)
                    {
                        // current_core_entry.first = boost::lexical_cast<unsigned>(value);
                        current_core_entry.first = stoi(value);
                        continue;
                    }

                    if (key == core_id)
                    {
                        // current_core_entry.second = boost::lexical_cast<unsigned>(value);
                        current_core_entry.second = stoi(value);
                        cores.insert(current_core_entry);
                        continue;
                    }
                }

                // Fall back to hardware_concurrency() in case
                // /proc/cpuinfo is formatted differently than we expect.
                // return cores.size() != 0 ? cores.size() : hardware_concurrency();
                return cores.size() != 0 ? cores.size() : std::thread::hardware_concurrency();
            }
            catch (...)
            {
                // return hardware_concurrency();
                return std::thread::hardware_concurrency();
            }

#elif defined(IS_MACOS)
            int count;
            size_t size = sizeof(count);
            return sysctlbyname("hw.physicalcpu", &count, &size, NULL, 0) ? 0 : count;
#else
            return std::thread::hardware_concurrency();
#endif
        }
    } // namespace Util
} // namespace SyncLib

namespace SyncLibInternal
{
    inline void PinThread(const size_t s)
    {
        const int maxS = static_cast<int>(std::thread::hardware_concurrency());
        const int virtiualConcurrency = maxS / SyncLib::Util::PhysicalConcurrency();
        int core = static_cast<int>(s) % maxS;
        core *= virtiualConcurrency;
        core = (core % maxS) + (core / maxS);
#if defined(IS_WINDOWS)

        DWORD_PTR mask = 1;
        mask = mask << core;
        SetThreadAffinityMask(GetCurrentThread(), mask);
        SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST);
#elif defined(IS_LINUX)
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(core, &cpuset);

        pthread_t currentThread = pthread_self();
        pthread_setaffinity_np(currentThread, sizeof(cpu_set_t), &cpuset);
#endif
    }
} // namespace SyncLibInternal

#endif