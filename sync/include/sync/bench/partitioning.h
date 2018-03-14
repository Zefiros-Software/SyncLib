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
#ifndef __SYNCLIB_PARTITIONING_H__
#define __SYNCLIB_PARTITIONING_H__

#include "sync/util/algorithm.h"

#include "fmt/format.h"

#include "nlohmann/json.hpp"

#include <armadillo>

#include <algorithm>
#include <numeric>
#include <set>

namespace SyncLib
{
    namespace Bench
    {
        class DistanceMatrix
        {
        public:

            DistanceMatrix(const arma::mat &distances)
                : mDistances(distances),
                  mSize(distances.n_rows)
            {}

            auto GetSubview()
            {
                return mDistances(0, 0, GetShape());
            }

            arma::SizeMat GetShape() const
            {
                return arma::SizeMat(mSize, mSize);
            }

            void Merge(const size_t &i, const size_t &j)
            {
                const size_t m1 = mSize - 1;

                for (size_t k = 0; k < mSize; ++k)
                {
                    mDistances(i, k) = std::max(mDistances(i, k), mDistances(j, k));
                    mDistances(k, i) = std::max(mDistances(k, i), mDistances(k, j));

                    if (k != m1 && k != j)
                    {
                        std::swap(mDistances(m1, k), mDistances(j, k));
                        std::swap(mDistances(k, m1), mDistances(k, j));
                    }
                }

                std::swap(mDistances(j, j), mDistances(m1, m1));
                std::swap(mDistances(j, m1), mDistances(m1, j));

                --mSize;
            }

        private:

            arma::mat mDistances;
            size_t mSize;
        };

        class Clusters
        {
        public:

            Clusters(const size_t &count)
                : mClusters(count),
                  mSize(count)
            {
                for (size_t c = 0; c < count; ++c)
                {
                    mClusters[c].push_back(c);
                }
            }

            const auto &GetClusters() const
            {
                return mClusters;
            }

            void Merge(const size_t &i, const size_t &j)
            {
                auto &cI = mClusters[i];
                auto &cJ = mClusters[j];

                cI.insert(cI.end(), cJ.begin(), cJ.end());
                cJ = mClusters[mSize - 1];
                mClusters.pop_back();
                --mSize;
            }

        private:

            std::vector<std::vector<size_t>> mClusters;
            size_t mSize;
        };
    }

    namespace Clustering
    {
        class AbstractClustering
        {
        public:

            AbstractClustering(const arma::mat &distances)
                : mDistances(distances)
            {}

            const arma::mat &GetDistances() const
            {
                return mDistances;
            }

        protected:

            const arma::mat &mDistances;
        };

        class UPGMA
            : public AbstractClustering
        {
        public:

            UPGMA(const arma::mat &distances)
                : AbstractClustering(distances),
                  mClusterDistances(distances),
                  mClusters(distances.n_rows)
            {
            }

            void MakeClustering(size_t count)
            {
                arma::vec diag = mClusterDistances.GetSubview().diag();
                mClusterDistances.GetSubview().diag() *= std::numeric_limits<double>::infinity();

                for (size_t m = diag.size(); m > count; --m)
                {
                    auto d = mClusterDistances.GetSubview();
                    auto [i, j] = Util::ArgMin(d, mClusterDistances.GetShape());

                    mClusterDistances.Merge(i, j);
                    mClusters.Merge(i, j);
                }
            }

            const Bench::Clusters &GetClusters() const
            {
                return mClusters;
            }

            //private:

            Bench::DistanceMatrix mClusterDistances;
            Bench::Clusters mClusters;
        };
    }

    namespace Partitioning
    {
        class ClusterInitialised
        {
        public:

            template<typename tClustering>
            ClusterInitialised(const tClustering &clustering)
                : mParts(clustering.GetClusters().GetClusters()),
                  mDistances(clustering.GetDistances())
            {
            }

            void MakePartitioning(size_t)
            {
                std::vector<size_t> sizes(mParts.size());
                size_t minSize, maxSize;

                auto imbalance = std::tie(minSize, maxSize);
                imbalance = SizeImbalance(sizes);

                for (; maxSize - minSize > 1; imbalance = SizeImbalance(sizes))
                {
                    Rebalance(maxSize);
                }
            }

            const std::vector<std::vector<size_t>> &GetParts() const
            {
                return mParts;
            }

            const arma::mat &GetDistances() const
            {
                return mDistances;
            }

        private:

            std::vector<std::vector<size_t>> mParts;
            const arma::mat &mDistances;

            void Rebalance(size_t maxSize)
            {
                auto largePart = std::find_if(mParts.begin(), mParts.end(), [&](auto part)
                {
                    return part.size() == maxSize;
                });
                const size_t m = largePart->size();

                auto [worstSample, wGlob, worstCost] = WorstSample(*largePart, m);
                size_t bestAlternative = BestAlternative(wGlob, largePart - mParts.begin(), worstCost);
                largePart->erase(largePart->begin() + worstSample);
                mParts[bestAlternative].push_back(wGlob);
            }

            std::tuple<size_t, size_t, double> WorstSample(std::vector<size_t> &largePart, const size_t m)
            {
                arma::uvec slicer(&largePart[0], m, false);
                auto d = mDistances(slicer, slicer);
                auto[i, j] = Util::ArgMax(d, arma::SizeMat(m, m));
                size_t iGlob = largePart[i];
                size_t jGlob = largePart[j];

                arma::uvec iSlice({ iGlob });
                arma::uvec jSlice({ jGlob });
                double iCost = (arma::mean(mDistances(iSlice, slicer), 1) + arma::mean(mDistances(slicer, iSlice)))[0];
                double jCost = (arma::mean(mDistances(jSlice, slicer), 1) + arma::mean(mDistances(slicer, jSlice)))[0];

                if (iCost > jCost)
                {
                    return { i, iGlob, mDistances(iGlob, jGlob) };
                }
                else
                {
                    return { j, jGlob, mDistances(iGlob, jGlob) };
                }
            }

            size_t BestAlternative(size_t worstSample, size_t worstIndex, double worstCost)
            {
                size_t bestAlternative = worstIndex;
                double bestCost = std::numeric_limits<double>::infinity();
                const size_t maxSize = mParts[worstIndex].size() - 2;

                for (size_t i = 0, iEnd = mParts.size(); i < iEnd; ++i)
                {
                    if (i == worstIndex || mParts[i].size() > maxSize)
                    {
                        continue;
                    }

                    double worstI = 0;

                    for (auto &j : mParts[i])
                    {
                        double cost = std::max(mDistances(j, worstSample), mDistances(worstSample, j));

                        if (cost > worstI)
                        {
                            worstI = cost;
                        }
                    }

                    if (worstI < bestCost)
                    {
                        bestAlternative = i;
                        bestCost = worstI;
                    }
                }

                return bestAlternative;
            }

            std::tuple<size_t, size_t> SizeImbalance(std::vector<size_t> &buffer)
            {
                std::transform(mParts.begin(), mParts.end(), buffer.begin(), std::size<std::vector<size_t>>);
                return Util::MinMax(buffer.begin(), buffer.end());
            }
        };

        class TripletImprover
        {
        public:

            class Part
            {
            public:

                Part(const std::vector<size_t> &samples, const arma::mat &distances)
                    : mSamples(samples),
                      mShape(samples.size(), samples.size()),
                      mDistances(distances)
                {
                    auto[i, j] = Util::ArgMax(mDistances(mSamples, mSamples), mShape);
                    size_t iGlob = mSamples[i];

                }

                void Replace(size_t index, size_t sample)
                {
                    mSamples[index] = sample;
                }

                double WorstCostAfterReplace(size_t index, size_t sample)
                {
                    double worstCost = 0;

                    for (size_t i = 0, iEnd = mSamples.size(); i < iEnd; ++i)
                    {
                        if (i == index)
                        {
                            continue;
                        }

                        double cost = std::max(mDistances(mSamples[i], sample), mDistances(sample, mSamples[i]));

                        if (cost > worstCost)
                        {
                            worstCost = cost;
                        }
                    }

                    return worstCost;
                }

                size_t Size() const
                {
                    return mSamples.size();
                }


                const arma::uvec &GetSamples() const
                {
                    return mSamples;
                }

                size_t &At(size_t index)
                {
                    return mSamples[index];
                }

                const size_t &At(size_t index) const
                {
                    return mSamples[index];
                }

                const arma::SizeMat &GetShape() const
                {
                    return mShape;
                }

            private:

                arma::uvec mSamples;
                arma::SizeMat mShape;
                const arma::mat &mDistances;
            };

            using Candidate = std::tuple<Part *, size_t, Part *, size_t>;

            template<typename tPartitioning>
            TripletImprover(const tPartitioning &partitioning)
                : mDistances(partitioning.GetDistances()),
                  mRandom(mRandomDevice())
            {
                for (auto &part : partitioning.GetParts())
                {
                    mParts.emplace_back(part, mDistances);
                }
            }

            void Improve()
            {
                std::vector<Part *> parts(mParts.size());

                for (size_t i = 0; i < mParts.size(); ++i)
                {
                    parts[i] = &mParts[i];
                }

                for (bool improved = true; improved;)
                {
                    improved = TryImprove(parts);

                }
            }

            bool TryImprove(std::vector<Part *> &parts)
            {
                auto[worstPart, worstSample, worstSampleGlob, worstCost] = WorstSample();

                std::shuffle(parts.begin(), parts.end(), mRandom);

                for (auto part1 : parts)
                {
                    if (part1 == worstPart)
                    {
                        continue;
                    }

                    for (auto &part2 : mParts)
                    {
                        for (size_t sample1 = 0, end1 = part1->Size(); sample1 < end1; ++sample1)
                        {
                            size_t sample1Glob = part1->At(sample1);

                            for (size_t sample2 = 0, end2 = part2.Size(); sample2 < end2; ++sample2)
                            {
                                size_t sample2Glob = part2.At(sample2);

                                if (sample2Glob == worstSampleGlob || sample2Glob == sample1Glob)
                                {
                                    continue;
                                }

                                bool hasThreeParts = !(part1 == &part2 || &part2 == worstPart);

                                double cost = hasThreeParts ? std::max(
                                {
                                    worstPart->WorstCostAfterReplace(worstSample, sample2Glob),
                                    part1->WorstCostAfterReplace(sample1, worstSampleGlob),
                                    part2.WorstCostAfterReplace(sample2, sample1Glob)
                                }) : std::max(
                                {
                                    worstPart->WorstCostAfterReplace(worstSample, sample1Glob),
                                    part1->WorstCostAfterReplace(sample1, worstSampleGlob),
                                });

                                if (cost < worstCost)
                                {
                                    if (hasThreeParts)
                                    {
                                        part2.Replace(sample2, sample1Glob);
                                        worstPart->Replace(worstSample, sample2Glob);
                                        part1->Replace(sample1, worstSampleGlob);
                                    }
                                    else
                                    {
                                        worstPart->Replace(worstSample, sample1Glob);
                                        part1->Replace(sample1, worstSampleGlob);
                                    }

                                    return true;
                                }
                            }
                        }
                    }
                }

                return false;
            }

            const std::vector<Part> &GetParts() const
            {
                return mParts;
            }

        private:

            std::random_device mRandomDevice;
            std::mt19937 mRandom;

            arma::mat mDistances;
            std::vector<Part> mParts;

            std::tuple<Part *, size_t, size_t, double> WorstSample()
            {
                double worstCost = 0;
                size_t worstSample = 0;
                size_t worstSampleGlob = 0;
                Part *worstPart = nullptr;

                for (auto &part : mParts)
                {
                    auto slicer = part.GetSamples();
                    size_t i, j;
                    std::tie(i, j) = Util::ArgMax(mDistances(slicer, slicer), arma::SizeMat(slicer.size(), slicer.size()));
                    size_t iGlob = slicer[i];
                    size_t jGlob = slicer[j];
                    double cost = std::max(mDistances(iGlob, jGlob), mDistances(jGlob, iGlob));

                    if (cost > worstCost)
                    {
                        worstCost = cost;
                        worstPart = &part;

                        arma::uvec iSlice({ iGlob });
                        arma::uvec jSlice({ jGlob });
                        double iCost = (arma::mean(mDistances(iSlice, slicer), 1) + arma::mean(mDistances(slicer, iSlice)))[0];
                        double jCost = (arma::mean(mDistances(jSlice, slicer), 1) + arma::mean(mDistances(slicer, jSlice)))[0];

                        if (iCost > jCost)
                        {
                            worstSample = i;
                            worstSampleGlob = iGlob;
                        }
                        else
                        {
                            worstSample = j;
                            worstSampleGlob = jGlob;
                        }
                    }
                }

                return std::make_tuple(worstPart, worstSample, worstSampleGlob, worstCost);
            }

            double WorstCost(Part &part)
            {
                double worstCost = 0;

                for (const size_t &i : part.GetSamples())
                {
                    for (const size_t &j : part.GetSamples())
                    {
                        const double &cost = mDistances(i, j);

                        if (cost > worstCost)
                        {
                            worstCost = cost;
                        }
                    }
                }

                return worstCost;
            }
        };

        std::vector<std::vector<size_t>> MakeImprovedClusterInitialisedPartitioning(const arma::mat &distances, size_t count = 0);
    }
}

#endif