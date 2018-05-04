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
#ifndef __SYNCLIB_MPIPINGPONG_H__
#define __SYNCLIB_MPIPINGPONG_H__

#include "sync/bench/timingsCollector.h"

#include "sync/bindings/mpi.h"

#include "nlohmann/json.hpp"

#include <armadillo>

#include <iostream>

namespace SyncLib
{
    namespace Bench
    {
        class MPIPingPongBenchmark
        {
        public:
            MPIPingPongBenchmark(SyncLib::MPI::Comm &comm, size_t skip = 4, size_t maxCount = 256, size_t packageSize = 64,
                                 size_t repetitions = 40, size_t rotations = 4);

            static std::tuple<double, double> LeastSquares(size_t h0, size_t h1, size_t multiplier, const arma::rowvec &t);
            void PingPong();
            void Serialise(std::ostream &out = std::cout);
            std::tuple<arma::mat, arma::mat> ComputePairwise();

            static std::tuple<arma::mat, arma::mat> ComputePairwise(const nlohmann::json &j)
            {
                arma::vec domain = j["domain"];
                auto &data = j["data"];
                const size_t p = data.size();
                arma::mat G = arma::zeros(p, p);
                arma::mat L = arma::zeros(p, p);

                for (auto &sData : data)
                {
                    const size_t s = sData["source"];

                    for (auto &dataT : sData["data"])
                    {
                        const size_t t = dataT["target"];
                        {
                            if (t == s)
                            {
                                continue;
                            }

                            arma::rowvec times = dataT["timings"];

                            auto [g, l] = LeastSquares(1, times.size() - 1, 64, times);

                            G.at(s, t) = g;
                            L.at(s, t) = l;
                        }
                    }
                }

                return std::make_tuple(G, L);
            }

        private:
            AggregatedTimings mAggregatedTimings;
            SyncLib::MPI::Comm &mComm;

            size_t mSkip, mMaxCount, mPackageSize;
            size_t mRepetitions, mRotations;
            size_t mS, mP, mP2;
        };
    } // namespace Bench
} // namespace SyncLib

#endif