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
#include "sync/bench/mpiPingpong.h"
#include "sync/util/algorithm.h"
#include "sync/util/timer.h"

using json = nlohmann::json;
using namespace SyncLib::Bench;

MPIPingPongBenchmark::MPIPingPongBenchmark(SyncLib::MPI::Comm &comm,
                                           const size_t skip /*= 4*/,
                                           const size_t maxCount /*= 256*/,
                                           const size_t packageSize /*= 64*/,
                                           const size_t repetitions /*= 40*/,
                                           const size_t rotations /*= 4*/)
    : mAggregatedTimings(comm.Size(), comm.Rank(), maxCount / skip),
      mComm(comm),
      mSkip(skip),
      mMaxCount(maxCount),
      mPackageSize(packageSize),
      mRepetitions(repetitions),
      mRotations(rotations),
      mS(comm.Rank()),
      mP(comm.Size()),
      mP2(SyncLib::Util::NextPowerOfTwo(comm.Size()))
{

}

std::tuple<double, double>MPIPingPongBenchmark::LeastSquares(const size_t h0, const size_t h1, size_t multiplier,
                                                             const arma::rowvec &t)
{
    /* This function computes the parameters g and l of the
    linear function T(h)= g*h+l that best fits
    the data points (h,t[h]) with h0<= h<= h1. */

    /* Compute sums:
    sumt  =  sum of t[h] over h0<= h<= h1
    sumth =         t[h]*h
    nh    =         1
    sumh  =         h
    sumhh =         h*h */
    const double dh0 = static_cast<double>(h0);
    const double dh1 = static_cast<double>(h1);

    const double sumt = arma::sum(t);
    const double sumth = arma::dot(t, SyncLib::Util::ArmaRange(h0, h1)) * multiplier;

    const double nh = dh1 - dh0 + 1;
    const double sumh = multiplier * (dh1 * (dh1 + 1) - (dh0 - 1) * dh0) / 2;
    const double sumhh = multiplier * multiplier * (dh1 * (dh1 + 1) * (2 * dh1 + 1) - (dh0 - 1) * dh0 * (2 * dh0 - 1)) / 6;

    /* Solve    nh*l +  sumh*g =  sumt
    sumh*l + sumhh*g =  sumth */

    const double a = nh / static_cast<double>(sumh);    // nh<= sumh

    /* subtract a times second eqn from first eqn to obtain g */
    double g = (sumt - a * sumth) / (sumh - a * sumhh);

    /* use second eqn to obtain l */
    double l = (sumth - sumhh * g) / sumh;

    return std::make_tuple(g, l);
}

void MPIPingPongBenchmark::PingPong()
{
    Util::Timer<std::chrono::microseconds>timer;

    size_t receive;

    std::vector<size_t>sendBuff(mMaxCount * mPackageSize);
    std::vector<size_t>recvBuff(mMaxCount * mPackageSize);

    TimingsCollector timingsCollector(mP, mS, mMaxCount / mSkip);

    for (size_t i = 0; i < mRotations; ++i)
    {
        for (size_t mask = 1; mask < mP2; ++mask)
        {
            const size_t t = mS ^ mask;

            if (t < mP)
            {
                mComm.SendReceive(t, mS, receive);

                for (size_t count = 4; count <= mMaxCount; count += mSkip)
                {
                    timer.Tic();

                    for (size_t j = 0; j < mRepetitions; ++j)
                    {
                        mComm.SendReceive(t, sendBuff.data(), count * mPackageSize, &recvBuff[0], count * mPackageSize);
                    }

                    const double elapsed = timer.Toc();

                    timingsCollector.AddTiming(t, count / mSkip, elapsed / mRepetitions);
                }
            }
        }
    }

    mAggregatedTimings.Aggregate(timingsCollector);
}

void MPIPingPongBenchmark::Serialise(std::ostream &out /*= std::cout*/)
{
    if (mS == 0)
    {
        json dumper = json::object();
        dumper["domain"] = SyncLib::Util::ArmaRange(mSkip, mMaxCount, mSkip);

        json data = json::array();
        {
            data.push_back(mAggregatedTimings);

            for (size_t t = 1; t < mP; ++t)
            {
                AggregatedTimings otherTimings(mP, t, mMaxCount / mSkip);
                auto &buff = otherTimings.GetBuffer();
                mComm.Receive(t, &buff.at(0), buff.size());
                data.emplace_back(otherTimings);
            }
        }

        dumper["data"] = data;

        auto [G, L] = ComputePairwise();
        dumper["G"] = G;
        dumper["L"] = L;

        out << dumper;
    }
    else
    {
        const auto &buff = mAggregatedTimings.GetBuffer();
        mComm.Send(0, &buff.at(0, 0), buff.size(), 0);
        ComputePairwise();
    }
}

std::tuple<arma::mat, arma::mat>MPIPingPongBenchmark::ComputePairwise()
{
    arma::mat G = arma::zeros(mP, mP);
    arma::mat L = arma::zeros(mP, mP);

    arma::vec gRow = arma::zeros(mP);
    arma::vec lRow = arma::zeros(mP);

    for (size_t t = 0; t < mP; ++t)
    {
        if (t == mS)
        {
            continue;
        }

        auto[g, l] = LeastSquares(1, mMaxCount / mSkip, mSkip * mPackageSize, mAggregatedTimings.GetBuffer().row(t));

        gRow.at(t) = g;
        lRow.at(t) = l;
    }

    mComm.AllGather(gRow, G);
    mComm.AllGather(lRow, L);

    return std::make_tuple(G, L);
}
