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
#include "sync/bench/timingsCollector.h"

using json = ::nlohmann::json;

SyncLib::Bench::TimingsCollector::TimingsCollector(size_t p, size_t s, size_t maxCount)
    : mTimings(p, std::vector<std::vector<double>>(maxCount)),
      mP(p),
      mS(s),
      mMaxCount(maxCount)
{
}

const std::vector<double> &SyncLib::Bench::TimingsCollector::GetCountTimings(size_t t, size_t count) const
{
    return mTimings[t][count - 1];
}

void SyncLib::Bench::TimingsCollector::AddTiming(size_t t, size_t count, double timing)
{
    mTimings[t][count - 1].push_back(timing);
}

const size_t &SyncLib::Bench::TimingsCollector::GetP() const
{
    return mP;
}

const size_t &SyncLib::Bench::TimingsCollector::GetS() const
{
    return mS;
}

const size_t &SyncLib::Bench::TimingsCollector::GetMaxCount() const
{
    return mMaxCount;
}
