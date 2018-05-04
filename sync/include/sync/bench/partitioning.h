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
#include "sync/util/random.h"

#include "fmt/format.h"

#include "nlohmann/json.hpp"

#include <armadillo>

#include <algorithm>
#include <set>

namespace SyncLib
{
    namespace Bench
    {
        class DistanceMatrix
        {
        public:
            DistanceMatrix(const arma::mat &distances)
                : mDistances(distances)
                , mSize(distances.n_rows)
            {
            }

            auto GetSubview()
            {
                return mDistances(0, 0, GetShape());
            }

            arma::SizeMat GetShape() const
            {
                return arma::SizeMat(mSize, mSize);
            }

            void Merge(const arma::uword &i, const arma::uword &j)
            {
                const arma::uword m1 = mSize - 1;

                for (arma::uword k = 0; k < mSize; ++k)
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
            arma::uword mSize;
        };

        class Clusters
        {
        public:
            Clusters(const arma::uword &count)
                : mClusters(count)
                , mSize(count)
            {
                for (arma::uword c = 0; c < count; ++c)
                {
                    mClusters[c].push_back(c);
                }
            }

            const auto &GetClusters() const
            {
                return mClusters;
            }

            void Merge(const arma::uword &i, const arma::uword &j)
            {
                auto &cI = mClusters[i];
                auto &cJ = mClusters[j];

                cI.insert(cI.end(), cJ.begin(), cJ.end());
                cJ = mClusters[mSize - 1];
                mClusters.pop_back();
                --mSize;
            }

        private:
            std::vector<std::vector<arma::uword>> mClusters;
            arma::uword mSize;
        };
    } // namespace Bench

    namespace Clustering
    {
        class AbstractClustering
        {
        public:
            AbstractClustering(const arma::mat &distances)
                : mDistances(distances)
            {
            }

            const arma::mat &GetDistances() const
            {
                return mDistances;
            }

        protected:
            const arma::mat &mDistances;
        };

        class UPGMA : public AbstractClustering
        {
        public:
            UPGMA(const arma::mat &distances)
                : AbstractClustering(distances)
                , mClusterDistances(distances)
                , mClusters(distances.n_rows)
            {
            }

            void MakeClustering(const arma::uword count)
            {
                arma::vec diag = mClusterDistances.GetSubview().diag();
                mClusterDistances.GetSubview().diag() *= std::numeric_limits<double>::infinity();

                for (arma::uword m = diag.size(); m > count; --m)
                {
                    const auto d = mClusterDistances.GetSubview();
                    auto [i, j] = Util::ArgMin(d, mClusterDistances.GetShape());

                    mClusterDistances.Merge(i, j);
                    mClusters.Merge(i, j);
                }
            }

            const Bench::Clusters &GetClusters() const
            {
                return mClusters;
            }

        private:
            Bench::DistanceMatrix mClusterDistances;
            Bench::Clusters mClusters;
        };
    } // namespace Clustering

    namespace Partitioning
    {
        class ClusterInitialised
        {
        public:
            template <typename tClustering>
            ClusterInitialised(const tClustering &clustering)
                : mParts(clustering.GetClusters().GetClusters())
                , mDistances(clustering.GetDistances())
            {
            }

            void MakePartitioning(arma::uword)
            {
                std::vector<arma::uword> sizes(mParts.size());
                arma::uword minSize, maxSize;

                std::tie(minSize, maxSize) = SizeImbalance(sizes);

                for (; maxSize - minSize > 1; std::tie(minSize, maxSize) = SizeImbalance(sizes))
                {
                    Rebalance(maxSize);
                }
            }

            const std::vector<std::vector<arma::uword>> &GetParts() const
            {
                return mParts;
            }

            const arma::mat &GetDistances() const
            {
                return mDistances;
            }

        private:
            std::vector<std::vector<arma::uword>> mParts;
            const arma::mat &mDistances;

            void Rebalance(arma::uword maxSize)
            {
                auto largePart = std::find_if(mParts.begin(), mParts.end(), [&](auto part)
                {
                    return part.size() == maxSize;
                });
                const arma::uword m = largePart->size();

                auto [worstSample, wGlob, worstCost] = WorstSample(*largePart, m);
                const arma::uword bestAlternative = BestAlternative(wGlob, largePart - mParts.begin(), worstCost);
                largePart->erase(largePart->begin() + worstSample);
                mParts[bestAlternative].push_back(wGlob);
            }

            std::tuple<arma::uword, arma::uword, double> WorstSample(std::vector<arma::uword> &largePart, const arma::uword m) const
            {
                // arma::uvec slicer(&largePart[0], m, false);
                const arma::uvec slicer(largePart);
                const auto d = mDistances(slicer, slicer);
                auto [i, j] = Util::ArgMax(d, arma::SizeMat(m, m));
                arma::uword iGlob = largePart[i];
                arma::uword jGlob = largePart[j];

                const arma::uvec iSlice({ iGlob });
                const arma::uvec jSlice({ jGlob });
                const double iCost = (arma::mean(mDistances(iSlice, slicer), 1) + arma::mean(mDistances(slicer, iSlice)))[0];
                const double jCost = (arma::mean(mDistances(jSlice, slicer), 1) + arma::mean(mDistances(slicer, jSlice)))[0];

                if (iCost > jCost)
                {
                    return { i, iGlob, mDistances(iGlob, jGlob) };
                }
                else
                {
                    return { j, jGlob, mDistances(iGlob, jGlob) };
                }
            }

            arma::uword BestAlternative(const arma::uword worstSample, const arma::uword worstIndex, double /*worstCost*/)
            {
                arma::uword bestAlternative = worstIndex;
                double bestCost = std::numeric_limits<double>::infinity();
                const arma::uword maxSize = mParts[worstIndex].size() - 2;

                for (arma::uword i = 0, iEnd = mParts.size(); i < iEnd; ++i)
                {
                    if (i == worstIndex || mParts[i].size() > maxSize)
                    {
                        continue;
                    }

                    double worstI = 0;

                    for (auto &j : mParts[i])
                    {
                        const double cost = std::max(mDistances(j, worstSample), mDistances(worstSample, j));

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

            std::tuple<arma::uword, arma::uword> SizeImbalance(std::vector<arma::uword> &buffer)
            {
                std::transform(mParts.begin(), mParts.end(), buffer.begin(), std::size<std::vector<arma::uword>>);
                return Util::MinMax(buffer.begin(), buffer.end());
            }
        };

        class TripletImprover
        {
        public:
            class Part
            {
            public:
                Part(const std::vector<arma::uword> &samples, const arma::mat &distances)
                    : mSamples(samples)
                    , mShape(samples.size(), samples.size())
                    , mDistances(distances)
                {
                    /*auto [i, j] = Util::ArgMax(mDistances(mSamples, mSamples), mShape);*/
                    /*arma::uword iGlob = mSamples[i];*/
                }

                void Replace(const arma::uword index, const arma::uword sample)
                {
                    mSamples[index] = sample;
                }

                double WorstCostAfterReplace(const arma::uword index, const arma::uword sample)
                {
                    double worstCost = 0;

                    for (arma::uword i = 0, iEnd = mSamples.size(); i < iEnd; ++i)
                    {
                        if (i == index)
                        {
                            continue;
                        }

                        const double cost = std::max(mDistances(mSamples[i], sample), mDistances(sample, mSamples[i]));

                        if (cost > worstCost)
                        {
                            worstCost = cost;
                        }
                    }

                    return worstCost;
                }

                arma::uword Size() const
                {
                    return mSamples.size();
                }

                const arma::uvec &GetSamples() const
                {
                    return mSamples;
                }

                const arma::uword &At(arma::uword index) const
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

            using tCandidate = std::tuple<Part *, arma::uword, Part *, arma::uword>;

            template <typename tPartitioning>
            TripletImprover(const tPartitioning &partitioning, size_t seed = 42)
                : mRandom(seed)
                , mDistances(partitioning.GetDistances())
            {
                for (auto &part : partitioning.GetParts())
                {
                    mParts.emplace_back(part, mDistances);
                }
            }

            void Improve()
            {
                std::vector<Part *> parts(mParts.size());

                for (arma::uword i = 0; i < mParts.size(); ++i)
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
                auto [worstPart, worstSample, worstSampleGlob, worstCost] = WorstSample();

                std::shuffle(parts.begin(), parts.end(), mRandom);

                for (auto part1 : parts)
                {
                    if (part1 == worstPart)
                    {
                        continue;
                    }

                    for (auto &part2 : mParts)
                    {
                        for (arma::uword sample1 = 0, end1 = part1->Size(); sample1 < end1; ++sample1)
                        {
                            const arma::uword sample1Glob = part1->At(sample1);

                            for (arma::uword sample2 = 0, end2 = part2.Size(); sample2 < end2; ++sample2)
                            {
                                const arma::uword sample2Glob = part2.At(sample2);

                                if (sample2Glob == worstSampleGlob || sample2Glob == sample1Glob)
                                {
                                    continue;
                                }

                                const bool hasThreeParts = !(part1 == &part2 || &part2 == worstPart);

                                const double cost = hasThreeParts
                                                    ? std::max({ worstPart->WorstCostAfterReplace(worstSample, sample2Glob),
                                                                 part1->WorstCostAfterReplace(sample1, worstSampleGlob), part2.WorstCostAfterReplace(sample2, sample1Glob) })
                                                    : std::max(
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
            SyncLib::Util::LinearCongruentialRandom mRandom;

            arma::mat mDistances;
            std::vector<Part> mParts;

            std::tuple<Part *, arma::uword, arma::uword, double> WorstSample()
            {
                double worstCost = 0;
                arma::uword worstSample = 0;
                arma::uword worstSampleGlob = 0;
                Part *worstPart = nullptr;

                for (auto &part : mParts)
                {
                    auto slicer = part.GetSamples();
                    arma::uword i, j;
                    std::tie(i, j) = Util::ArgMax(mDistances(slicer, slicer), arma::SizeMat(slicer.size(), slicer.size()));
                    const arma::uword iGlob = slicer[i];
                    const arma::uword jGlob = slicer[j];
                    const double cost = std::max(mDistances(iGlob, jGlob), mDistances(jGlob, iGlob));

                    if (cost > worstCost)
                    {
                        worstCost = cost;
                        worstPart = &part;

                        const arma::uvec iSlice({ iGlob });
                        const arma::uvec jSlice({ jGlob });
                        const double iCost = (arma::mean(mDistances(iSlice, slicer), 1) + arma::mean(mDistances(slicer, iSlice)))[0];
                        const double jCost = (arma::mean(mDistances(jSlice, slicer), 1) + arma::mean(mDistances(slicer, jSlice)))[0];

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

                for (const arma::uword &i : part.GetSamples())
                {
                    for (const arma::uword &j : part.GetSamples())
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

        std::vector<std::vector<arma::uword>> MakeImprovedClusterInitialisedPartitioning(const arma::mat &distances,
                                                                                         arma::uword count = 0);
    } // namespace Partitioning
} // namespace SyncLib

#endif
