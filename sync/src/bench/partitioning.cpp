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
#include "sync/bench/partitioning.h"

#include <algorithm>

std::vector<std::vector<size_t>> SyncLib::Partitioning::MakeImprovedClusterInitialisedPartitioning(const arma::mat &distances,
                                                                                                   size_t count /*= 0*/)
{
    size_t p = distances.n_rows;

    if (count > p)
    {
        count = p;
    }
    else if (count == 0)
    {
        count = static_cast<size_t>(std::sqrt(p));

        while ((p / count) * count != p)
        {
            --count;
        }
    }

    assert((p / count) * count == p);

    if (count == p)
    {
        std::vector<std::vector<size_t>> parts(p);

        for (size_t i = 0; i < p; ++i)
        {
            parts[i].push_back(i);
        }

        return parts;
    }
    else if (count > 1)
    {
        Clustering::UPGMA clustering(distances);
        clustering.MakeClustering(count);
        ClusterInitialised partitioning(clustering);
        partitioning.MakePartitioning(count);
        TripletImprover improver(partitioning);
        improver.Improve();

        std::vector<std::vector<size_t>> result(count);
        auto &parts = improver.GetParts();

        for (size_t i = 0; i < count; ++i)
        {
            result[i] = arma::conv_to<std::vector<size_t>>::from(parts[i].GetSamples());
        }

        return result;
    }
    else
    {
        std::vector<std::vector<size_t>> parts(1);

        for (size_t i = 0; i < p; ++i)
        {
            parts[0].push_back(i);
        }

        return parts;
    }
}
