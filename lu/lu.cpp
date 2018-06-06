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
#define SYNCLIB_STRICT_ASSERT
#include "sync/sync.h"
#include "sync/util/random.h"
#include "sync/util/timer.h"
#include "sync/util/json.h"

#include "args/args.h"

#include <armadillo>

#include <numeric>
#include <tuple>

struct ProcessorMatrix
{
    size_t p;
    size_t M, N;

    ProcessorMatrix(const size_t iP, const size_t iM, const size_t iN)
        : p(iP),
          M(iM),
          N(iN)
    {
        SyncLibInternal::Assert(M * N == p, "Dimensions mismatch, {} * {} = {} != {}", M, N, M * N, p);
    }

    ProcessorMatrix(const size_t iP, const size_t iM)
        : ProcessorMatrix(iP, iM, iM)
    {}

    std::tuple<size_t, size_t>operator()(size_t rank) const
    {
        return std::make_tuple(rank / N, rank % N);
    }

    size_t operator()(size_t s, size_t t)const
    {
        return s * N + t;
    }

    template<typename tEnv>
    tEnv &Split0(tEnv &env, size_t s0, size_t s1) const
    {
        return env.Split(s1, s0);
    }

    template<typename tEnv>
    tEnv &Split1(tEnv &env, size_t s0, size_t s1) const
    {
        return env.Split(s0, s1);
    }

    template<typename tEnv>
    std::tuple<tEnv &, tEnv &> Split(tEnv &env, size_t s0, size_t s1) const
    {
        tEnv &env0 = Split0(env, s0, s1);
        return std::tuple<tEnv &, tEnv &>(env0, Split1(env, s0, s1));
    }
};

class MemoryPool
{
public:

    class Claim
    {
    public:

        arma::mat mat;

        Claim(size_t rows, size_t columns, arma::mat &memory, MemoryPool &pool)
            : mMemory(memory),
              mPool(pool),
              mat(&memory(0), rows, columns, false, true)
        {

        }

        ~Claim()
        {
            mPool.mAvailable.push_back(&mMemory);
        }

    private:

        arma::mat &mMemory;
        MemoryPool &mPool;
    };

    friend class ::MemoryPool::Claim;

    Claim Allocate(size_t rows, size_t columns)
    {
        arma::mat *memory = nullptr;
        size_t size = rows * columns;

        for (auto mat : mAvailable)
        {
            if (mat->n_elem > size)
            {
                memory = mat;
                break;
            }
        }

        if (memory == nullptr)
        {
            mAllocated.emplace_back(new arma::mat(rows, columns));
            memory = mAllocated.back();
        }

        return { rows, columns, *memory, *this };
    }

    ~MemoryPool()
    {
        for (auto mat : mAllocated)
        {
            delete mat;
        }
    }

private:

    std::vector<arma::mat *> mAvailable;
    std::vector<arma::mat *> mAllocated;
};

thread_local MemoryPool tmpMemPool;

using tEnv = SyncLib::Environments::SharedMemoryBSP;

struct LUCommunicator
{
    template<typename tEnv>
    static std::tuple<tEnv &, tEnv &> MaybeSplit(tEnv &env, const ProcessorMatrix &P, size_t s0, size_t s1)
    {
        return { env, env };
    }

    template<typename tEnv, typename... tEnvs>
    static void SyncAll(tEnv &env, tEnvs &...)
    {
        env.Sync();
    }

    template<typename tSendQueue>
    static void ColumnSendPivot(tSendQueue &sqPivot, size_t s0, size_t s1, size_t r, size_t kr0, double a, const ProcessorMatrix &P)
    {
        for (size_t t0 = 0; t0 < P.M; ++t0)
        {
            sqPivot.Send(P(t0, s1), a, s0, r);
        }
    }

    template<typename tSharedValue>
    static void ShareR(tSharedValue &rGlob, size_t s0, const ProcessorMatrix &P)
    {
        for (size_t t1 = 0; t1 < P.N; ++t1)
        {
            rGlob.Put(P(s0, t1));
        }
    }

    template<typename tSendQueue, typename tVec>
    static void SendSwap(tSendQueue &sqSwap, size_t t0, size_t s1, size_t k, const tVec &rowR, const ProcessorMatrix &P)
    {
        sqSwap.Send(P(t0, s1), k, rowR);

    }

    template<typename tSendQueue, typename tVec>
    static void SendColumnToRowMembers(tSendQueue &sqColK, size_t s0, size_t k, const ProcessorMatrix &P, const tVec &colK)
    {
        for (size_t t1 = 0; t1 < P.N; ++t1)
        {
            sqColK.Send(P(s0, t1), k, colK);
        }
    }

    template<typename tSendQueue, typename tVec>
    static void SendColumnToRemainingMembers(tSendQueue &sqColKRemaining, size_t t0, size_t k, const ProcessorMatrix &P,
                                             const tVec &colK)
    {
        for (size_t t1 = 0; t1 < P.N; ++t1)
        {
            sqColKRemaining.Send(P(t0, t1), k, colK);
        }
    }

    template<typename tSendQueue, typename tVec>
    static void SendRowToColumnMembers(tSendQueue &sqRowK, size_t s1, size_t k, const ProcessorMatrix &P, const tVec &rowK)
    {
        for (size_t t0 = 0; t0 < P.M; ++t0)
        {
            sqRowK.Send(P(t0, s1), k, rowK);
        }
    }

    template<typename tSendQueue, typename tVec>
    static void SendDelayedSwap(tSendQueue &sqDelayedSwap, size_t s1, size_t k, size_t c0, const ProcessorMatrix &P,
                                const tVec &rowR2)
    {
        sqDelayedSwap.Send(P(k % P.M, s1), k, c0, rowR2);
    }
};

template<typename tEnv, typename... tEnvs>
struct EnvSyncer
{
    static void SyncAll(tEnv &env, tEnvs &...envs)
    {
        env.Sync();
        EnvSyncer<tEnvs...>::SyncAll(envs...);
    }
};

template<typename tEnv>
struct EnvSyncer<tEnv>
{
    static void SyncAll(tEnv &env)
    {
        env.Sync();
    }
};

struct LUSplitCommunicator
{
    template<typename tEnv>
    static std::tuple<tEnv &, tEnv &> MaybeSplit(tEnv &env, const ProcessorMatrix &P, size_t s0, size_t s1)
    {
        return P.Split(env, s0, s1);
    }

    template<typename... tEnvs>
    static void SyncAll(tEnvs &... envs)
    {
        EnvSyncer<tEnvs...>::SyncAll(envs...);
    }

    template<typename tSendQueue>
    static void ColumnSendPivot(tSendQueue &sqPivot, size_t s0, size_t s1, size_t r, size_t kr0, double a, const ProcessorMatrix &P)
    {
        /*for (size_t t0 = 0; t0 < P.M; ++t0)
        {
            sqPivot.Send(P(t0, s1), a, s0, r + kr0);
        }*/
        sqPivot.Broadcast(a, s0, r);
    }

    template<typename tSharedValue>
    static void ShareR(tSharedValue &rGlob, size_t s0, const ProcessorMatrix &P)
    {
        /*for (size_t t1 = 0; t1 < P.N; ++t1)
        {
            rGlob.Put(P(s0, t1));
        }*/
        rGlob.Broadcast();
    }

    template<typename tSendQueue, typename tVec>
    static void SendSwap(tSendQueue &sqSwap, size_t t0, size_t s1, size_t k, const tVec &rowR, const ProcessorMatrix &P)
    {
        sqSwap.Send(t0, k, rowR);

    }

    template<typename tSendQueue, typename tVec>
    static void SendColumnToRowMembers(tSendQueue &sqColK, size_t s0, size_t k, const ProcessorMatrix &P, const tVec &colK)
    {
        /*for (size_t t1 = 0; t1 < P.N; ++t1)
        {
            sqColK.Send(P(s0, t1), k, colK);
        }*/
        sqColK.Broadcast(k, colK);
    }

    template<typename tSendQueue, typename tVec>
    static void SendColumnToRemainingMembers(tSendQueue &sqColKRemaining, size_t t0, size_t k, const ProcessorMatrix &P,
                                             const tVec &colK)
    {
        /*for (size_t t1 = 0; t1 < P.N; ++t1)
        {
            sqColKRemaining.Send(P(t0, t1), k, colK);
        }*/
        sqColKRemaining.Broadcast(k, colK);
    }

    template<typename tSendQueue, typename tVec>
    static void SendRowToColumnMembers(tSendQueue &sqRowK, size_t s1, size_t k, const ProcessorMatrix &P, const tVec &rowK)
    {
        /*for (size_t t0 = 0; t0 < P.M; ++t0)
        {
            sqRowK.Send(P(t0, s1), k, rowK);
        }*/
        sqRowK.Broadcast(k, rowK);
    }

    template<typename tSendQueue, typename tVec>
    static void SendDelayedSwap(tSendQueue &sqDelayedSwap, size_t s1, size_t k, size_t c0, const ProcessorMatrix &P,
                                const tVec &rowR2)
    {
        sqDelayedSwap.Send(k % P.M, k, c0, rowR2);
    }
};

template<typename tSubEnv, typename... tSubEnvs>
struct LUCoreImpl;

template<typename tCommunicator, typename tEnv, typename tSubEnv, typename... tSubEnvs>
struct LUAlgorithm
{
    template<typename... tValues>
    using SendQueue = typename tEnv::template SendQueue<tValues...>;

    template<typename tT>
    using SharedValue = typename tEnv::template SharedValue<tT>;

    using tCore = LUCoreImpl<tSubEnv, tSubEnvs..., SyncLib::Environments::NoBSP>;

    static size_t LocalCount(size_t p, size_t s, size_t n)
    {
        return (n - s + p - 1) / p;
    }

    static size_t GlobalRow(size_t k, const ProcessorMatrix &P, size_t s0)
    {
        return k * P.M + s0;
    }

    static size_t GlobalCol(size_t k, const ProcessorMatrix &P, size_t s1)
    {
        return k * P.N + s1;
    }

    static void FindLocalPivot(arma::mat &A, size_t k, size_t kr0, size_t nlr, size_t kc0, size_t s0, size_t s1,
                               const ProcessorMatrix &P, SendQueue<double, size_t, size_t> &sqPivot)
    {
        if (k % P.N == s1 && kr0 < nlr)
        {
            /* Superstep 0 (0):
            Compute the local pivot element
            */
            auto[r, a] = tCore::FindLocalPivot(A, kc0, kr0, nlr);
            r = GlobalRow(r, P, s0);

            /* Superstep 0 (1):
            Share the pivot element and its index with processors
            in the same processor column.
            */
            tCommunicator::ColumnSendPivot(sqPivot, s0, s1, r, kr0, a, P);
        }
    }

    static void FindGlobalPivotAndDivideColumnK(arma::mat &A, size_t k, size_t kr0, size_t nlr, size_t kc0, size_t s0, size_t s1,
                                                const ProcessorMatrix &P, SendQueue<double, size_t, size_t> &sqPivot, SharedValue<size_t> &rGlob)
    {
        /* Superstep 1 (2):
        Compute the global pivot element, its value and its source
        */
        if (k % P.N == s1)
        {
            double aMax = 0.0;
            size_t t0Max, rMax;

            for (auto[a, t0, r] : sqPivot)
            {
                if (std::abs(a) > aMax)
                {
                    aMax = a;
                    t0Max = t0;
                    rMax = r;
                }
            }

            /* Superstep 1 (2) continued:
            Divide column k of U by the pivot element
            */
            if (std::abs(aMax) <= std::numeric_limits<float>::epsilon())
            {
                throw std::runtime_error("LU is singular at stage " + std::to_string(k));
            }

            rGlob = rMax;

            /* Superstep 1 (3):
            Share the pivot row index with processors in the same processor row
            */
            tCommunicator::ShareR(rGlob, s0, P);

            if (kr0 < nlr)
            {
                tCore::DivideColumnK(A, rMax / P.M, kr0, nlr, kc0, aMax, t0Max, s0);
            }
        }
    }

    static void UpdatePermutation(size_t k, size_t k0, arma::uvec &pi, SharedValue<size_t> &rGlob,
                                  std::vector<std::tuple<size_t, size_t>> &swaps)
    {
        std::swap(pi[k], pi[rGlob]);

        swaps[k - k0] = std::make_tuple(k - k0, rGlob - k0);
    }

    static void ExchangeBlockRows(arma::mat &A, size_t k, size_t ck0, size_t ck1, size_t s0, size_t s1, const ProcessorMatrix &P,
                                  size_t rGlob, SendQueue<size_t, arma::vec> &sqSwap, SendQueue<size_t, arma::vec> &sqRowK)
    {
        if (k % P.M == s0 && ck0 < ck1)
        {
            size_t t0 = rGlob % P.M;
            arma::vec rowK(&A(ck0, k / P.M), ck1 - ck0, false, true);
            tCommunicator::SendSwap(sqSwap, t0, s1, rGlob / P.M, rowK, P);
        }

        if (rGlob % P.M == s0 && ck0 < ck1)
        {
            size_t t0 = k % P.M;
            arma::vec rowR(&A(ck0, rGlob / P.M), ck1 - ck0, false, true);
            tCommunicator::SendSwap(sqSwap, t0, s1, k / P.M, rowR, P);

            /*if (ck0 > k / P.N)
            {
            arma::vec subRowR(&rowR(0), ck0 - k / P.N, false, true);
            tCommunicator::SendRowToColumnMembers(sqRowK, s1, k, P, subRowR);
            }*/
        }
    }

    static void ShareColumnK(arma::mat &A, size_t k, size_t k0, size_t kr1, size_t nlr, size_t rk1, size_t kc0, size_t s0,
                             size_t s1, const ProcessorMatrix &P, SendQueue<size_t, arma::rowvec> &sqColK/*,
                                                                                    SendQueue<size_t, arma::rowvec> &sqColKRemaining*/)
    {
        if (k % P.N == s1 && kr1 < nlr)
        {
            // Share our column with processors in the same processor column
            auto colK = A(kc0, arma::span(kr1, nlr - 1));

            /*for (size_t t1 = 0; t1 < P.N; ++t1)
            {
            sqColK.Send(P(s0, t1), k, colK);
            }*/
            tCommunicator::SendColumnToRowMembers(sqColK, s0, k, P, colK);

            /*if (kr1 < rk1)
            {
            // Share the column inside the block with everyone else
            for (size_t t0 = 0; t0 < P.M; ++t0)
            {
            // skip t0 == s0, he already received it in the loop above
            if (t0 == s0) { continue; }

            / *for (size_t t1 = 0; t1 < P.N; ++t1)
            {
            sqColKRemaining.Send(P(t0, t1), GlobalRow(kr1, P, s0) - k0, A(kc0, arma::span(kr1, rk1 - 1)));
            }* /
            tCommunicator::SendColumnToRemainingMembers(sqColKRemaining, t0, GlobalRow(kr1, P, s0) - k0, P, A(kc0, arma::span(kr1, rk1 - 1)));
            }
            }*/
        }
    }

    static void ShareBlockRowK(arma::mat &A, size_t k, size_t kr0, size_t ck0, size_t ck1, size_t s0, size_t s1,
                               const ProcessorMatrix &P, SendQueue<size_t, arma::vec> &sqRowK)
    {
        if (k % P.M == s0 && ck0 < ck1)
        {
            // Share our row with processors in the same processor column
            auto rowK = A(arma::span(ck0, ck1 - 1), kr0);

            /*for (size_t t0 = 0; t0 < P.M; ++t0)
            {
            sqRowK.Send(P(t0, s1), k, rowK);
            }*/
            tCommunicator::SendRowToColumnMembers(sqRowK, s1, k, P, rowK);
        }
    }

    static void UpdateRemainingBlock(arma::mat &A, size_t k, size_t k0, size_t kr1, size_t rk1, size_t nlr, size_t ck0, size_t kc1,
                                     size_t ck1, size_t s0, size_t s1, arma::mat &L11, arma::mat &L21, arma::mat &tmpMem, const ProcessorMatrix &P,
                                     SendQueue<size_t, arma::rowvec> &sqColK, SendQueue<size_t, arma::vec> &sqRowK/*,
                                                                                     SendQueue<size_t, arma::rowvec> &sqColKRemaining*/)
    {
        for (auto[kj, colKj] : sqColK)
        {
            for (auto[ki, rowKi] : sqRowK)
            {
                if (kc1 < ck1)
                {
                    arma::vec rowKi1(&rowKi(kc1 - ck0), ck1 - kc1, false, true);
                    /*arma::mat Lkdiff(&tmpMem(0), rowKi1.n_rows, colKj.n_cols, false, true);

                    Lkdiff = rowKi1 * colKj;
                    A(arma::span(kc1, ck1 - 1), arma::span(kr1, nlr - 1)) -= Lkdiff;*/
                    tCore::UpdateA21(A, rowKi1, colKj, tmpMem, kc1, ck1, kr1, nlr);
                }

                if (ck0 < kc1 && k > k0)
                {
                    arma::rowvec rowKi0(&rowKi(0), kc1 - ck0, false, true);

                    for (size_t j = 0, jj = GlobalCol(ck0, P, s1) - k0; j < rowKi0.n_elem && jj < k - k0; ++j, jj += P.N)
                    {
                        L11(k - k0, jj) = rowKi0(j);
                    }
                }
            }

            if (rk1 < nlr)
            {
                arma::rowvec subColKj(&colKj(rk1 - kr1), nlr - rk1, false, true);
                L21(k - k0, arma::span(0, nlr - rk1 - 1)) = subColKj;
            }
        }

        /*for (auto[rt0, colKj] : sqColKRemaining)
        {
        for (size_t i = 0, ri = rt0; i < colKj.n_elem; ++i, ri += P.M)
        {
        L11(ri, k - k0) = colKj(i);
        }
        }*/
    }

    static void ReconstructSwaps(arma::uvec &oldRows, arma::uvec &newRows, size_t k0, size_t swapCount,
                                 std::vector<std::tuple<size_t, size_t>> &swaps)
    {
        std::map<size_t, size_t> swapMap;

        size_t memberCount = swapCount;

        for (size_t l = 0; l < swapCount; ++l)
        {
            auto &[x, y] = swaps[l];
            oldRows(l) = x;

            if (y >= swapCount)
            {
                auto it = swapMap.find(y);

                if (it == swapMap.end())
                {
                    swapMap.insert({ y, memberCount });
                    oldRows(memberCount) = y;
                    y = memberCount++;
                }
                else
                {
                    y = it->second;
                }
            }
        }

        oldRows.resize(memberCount);
        newRows.resize(memberCount);

        oldRows += k0;
        newRows = oldRows;

        for (auto &[x, y] : swaps)
        {
            std::swap(newRows(x), newRows(y));
        }
    }

    static void PerformDelayedSwaps(arma::mat &A, arma::uvec &oldRows, arma::uvec &newRows, size_t ck0, size_t ck1,
                                    size_t nlc, size_t s0, size_t s1, const ProcessorMatrix &P, SendQueue<size_t, size_t, arma::vec> &sqDelayedSwap)
    {
        for (size_t l = 0; l < newRows.n_rows; ++l)
        {
            size_t k = oldRows(l);
            size_t r = newRows(l);

            if (r % P.M == s0)
            {
                // If I am the owner of the new value for row k, send it to the old owner of row k
                arma::vec rowR(&A(0, r / P.M), nlc, false, true);

                if (ck0 > 0)
                {
                    arma::vec rowR0(&rowR(0), ck0, false, true);
                    // sqDelayedSwap.Send(P(k % P.M, s1), k / P.M, 0, rowR0);
                    tCommunicator::SendDelayedSwap(sqDelayedSwap, s1, k, 0, P, rowR0);
                }

                if (ck1 < nlc)
                {
                    arma::vec rowR2(&rowR(ck1), nlc - ck1, false, true);
                    //sqDelayedSwap.Send(P(k % P.M, s1), k / P.M, ck1, rowR2);
                    tCommunicator::SendDelayedSwap(sqDelayedSwap, s1, k, ck1, P, rowR2);
                }
            }
        }
    }

    static void FixL21Rows(arma::mat &A, size_t s0, size_t s1, size_t ck0, size_t ck1, size_t k1, arma::uvec &oldRows,
                           const ProcessorMatrix &P, SendQueue<size_t, arma::rowvec> &sqColK)
    {
        for (auto k : oldRows)
        {
            if (k % P.M == s0 && k >= k1)
            {
                // If I am in row k, and k > k1, reconstruct this row
                //arma::vec rowK(&A(ck0, k / P.M), ck1 - ck0, false, true);
                //auto rowK = A(arma::span(ck0, ck1 - 1), k / P.M).t();
                tCommunicator::SendColumnToRowMembers(sqColK, s0, k / P.M * P.N + s1, P, A(arma::span(ck0, ck1 - 1), k / P.M).t());
            }
        }
    }

    static void ShareRemainingRows(arma::mat &A, size_t k0, size_t k1, size_t ck1, size_t nlc, size_t s0, size_t s1,
                                   const ProcessorMatrix &P, SendQueue<size_t, arma::vec> &sqRowK)
    {
        for (size_t k = k0; k < k1; ++k)
        {
            if (k % P.M == s0 && ck1 < nlc)
            {
                // Share our column with processors in the same processor column
                auto rowK = A(arma::span(ck1, nlc - 1), k / P.M);

                /*for (size_t t0 = 0; t0 < P.M; ++t0)
                {
                sqRowK.Send(P(t0, s1), k, rowK);
                }*/
                tCommunicator::SendRowToColumnMembers(sqRowK, s1, k, P, rowK);
            }
        }
    }

    static void CompleteL11(arma::mat &L11, size_t k0, size_t ck0, size_t s0, size_t s1, const ProcessorMatrix &P,
                            SendQueue<size_t, arma::rowvec> &sqColK)
    {
        for (size_t j = GlobalCol(ck0, P, s1) - k0; j < L11.n_cols; j += P.N)
        {
            tCommunicator::SendColumnToRowMembers(sqColK, s0, j, P, L11.col(j).t());
        }
    }

    static void ReconstructU12(arma::mat &A, size_t b, size_t k0, size_t rk0, size_t rk1, size_t ck1, size_t nlc, size_t s0,
                               const ProcessorMatrix &P, arma::mat &L11, arma::mat &U12, SendQueue<size_t, arma::vec> &sqRowK)
    {
        for (auto[k, rowK] : sqRowK)
        {
            U12(k - k0, arma::span(0, rowK.n_elem - 1)) = rowK.t();
        }

        for (size_t i = 2; i < b; ++i)
        {
            for (size_t j = 0; j < i - 1; ++j)
            {
                L11(i, j) -= arma::as_scalar(L11(i, arma::span(j + 1, i - 1)) * L11(arma::span(j + 1, i - 1), j));
            }
        }

        U12 -= L11 * U12;

        // Update our part of A with the updated value
        if (ck1 < nlc)
        {
            for (size_t k = rk0, kr = GlobalRow(rk0, P, s0) - k0; k < rk1; ++k, kr += P.M)
            {
                A(arma::span(ck1, nlc - 1), k) = U12.row(kr).t();
            }
        }
    }

    template<typename tState>
    static void LUImpl(tSubEnv &subEnv, tEnv &env, arma::mat &A, arma::uvec &pi, const ProcessorMatrix &P, size_t n, size_t b,
                       tState &state)
    {
        size_t s0 = state.s0;
        size_t s1 = state.s1;
        size_t nlr = state.nlr;
        size_t nlc = state.nlc;
        auto &env0 = state.env0;
        auto &env1 = state.env1;

        for (size_t k0 = 0, block = 0; k0 < n; k0 += b, ++block)
        {
            const double foo = A.max();
            const size_t k1 = std::min(k0 + b, n);
            const size_t rk0 = LocalCount(P.M, s0, k0);
            const size_t ck0 = LocalCount(P.N, s1, k0);
            const size_t rk1 = std::min(nlr, LocalCount(P.M, s0, k1));
            const size_t ck1 = std::min(nlc, LocalCount(P.N, s1, k1));

            arma::mat L21(&state.L21mem(0), b, nlr - rk1, false, true);
            arma::mat U12(&state.U12mem(0), b, nlc - ck1, false, true);

            if (subEnv.Rank() == 0)
            {
                size_t swapCount = k1 - k0;
                state.swaps.resize(swapCount);

                for (size_t k = k0; k < k1; ++k)
                {
                    size_t kr0 = LocalCount(P.M, s0, k);
                    size_t kr1 = LocalCount(P.M, s0, k + 1);
                    size_t kc0 = LocalCount(P.N, s1, k);
                    size_t kc1 = LocalCount(P.N, s1, k + 1);

                    // fmt::print("{}: k={}\n", env.Rank(), k);
                    // fflush(stdout);
                    FindLocalPivot(A, k, kr0, nlr, kc0, s0, s1, P, state.sqPivot);
                    env0.Sync();

                    FindGlobalPivotAndDivideColumnK(A, k, kr0, nlr, kc0, s0, s1, P, state.sqPivot, state.rGlob);
                    env1.Sync();

                    // fmt::print("{}: r={}\n", env.Rank(), rGlob.Value());
                    // fflush(stdout);
                    UpdatePermutation(k, k0, pi, state.rGlob, state.swaps);

                    /* Superstep 2 (4):
                    Exchange rows k and r between processors, but only columns k0 <= k < k1
                    */
                    ExchangeBlockRows(A, k, ck0, ck1, s0, s1, P, state.rGlob, state.sqSwap, state.sqRowK);
                    env0.Sync();
                    // fmt::print("{}: rows exchanged\n", env.Rank());
                    // fflush(stdout);

                    /* Superstep 3 (5):
                    Receive the rows and update the permutation vector
                    */
                    for (auto[i, Ai] : state.sqSwap)
                    {
                        arma::vec rowI(&A(ck0, i), ck1 - ck0, false, true);
                        rowI = Ai;
                    }

                    /* Superstep 3 (6):
                    Broadcast our part of the column to processors in the same column
                    Broadcast our part of the row to processors in the same row.
                    */
                    ShareColumnK(A, k, k0, kr1, nlr, rk1, kc0, s0, s1, P, state.sqColK);
                    ShareBlockRowK(A, k, kr0, ck0, ck1, s0, s1, P, state.sqRowK);
                    tCommunicator::SyncAll(env0, env1);
                    // fmt::print("{}: rows&column duplicated\n", env.Rank());
                    // fflush(stdout);

                    // fmt::print("{}: updating A21\n", env.Rank());
                    // fflush(stdout);
                    /* Superstep 4 (7):
                    Receive the row and column parts, multiply them, subtract them.
                    */
                    UpdateRemainingBlock(A, k, k0, kr1, rk1, nlr, ck0, kc1, ck1, s0, s1, state.L11, L21, state.tmpMem, P, state.sqColK,
                                         state.sqRowK);
                    // fmt::print("{}: updated A22\n", env.Rank());
                    // fflush(stdout);
                }

                // fmt::print("{}: Reconstructing swaps\n", env.Rank());
                // fflush(stdout);
                arma::uvec oldRows(swapCount * 2);
                arma::uvec newRows(swapCount * 2);

                ReconstructSwaps(oldRows, newRows, k0, swapCount, state.swaps);

                // fmt::print("{}: Reconstructed swaps\n", env.Rank());
                // fflush(stdout);
                PerformDelayedSwaps(A, oldRows, newRows, ck0, ck1, nlc, s0, s1, P, state.sqDelayedSwap);

                // fmt::print("{}: Scheduling delayed swaps\n", env.Rank());
                // fflush(stdout);
                FixL21Rows(A, s0, s1, ck0, ck1, k1, oldRows, P, state.sqColK);

                // fmt::print("{}: Completing L11\n", env.Rank());
                // fflush(stdout);
                CompleteL11(state.L11, k0, ck0, s0, s1, P, state.sqColKL11);

                // fmt::print("{}: Completed L11\n", env.Rank());
                // fflush(stdout);
                tCommunicator::SyncAll(env0, env1);
                // fmt::print("{}: Prepared A12, L21\n", env.Rank());
                // fflush(stdout);

                for (auto[i, j, Ai] : state.sqDelayedSwap)
                {
                    arma::vec rowIPart(&A(j, i / P.M), Ai.n_elem, false, true);
                    rowIPart = Ai;
                }

                for (auto[x, rowK] : state.sqColK)
                {
                    size_t k = (x / P.N) * P.M + s0 - k1;
                    size_t t1 = x % P.N;

                    for (size_t j = 0, jj = t1; j < rowK.n_elem; ++j, jj += P.N)
                    {
                        L21(jj, k) = rowK(j);
                    }
                }

                for (auto[j, colJ] : state.sqColKL11)
                {
                    state.L11.col(j) = colJ.t();
                }

                /* Superstep 3 (6):
                Broadcast our part of the column to processors in the same column
                Broadcast our part of the row to processors in the same row.
                */
                ShareRemainingRows(A, k0, k1, ck1, nlc, s0, s1, P, state.sqRowK);
                tCommunicator::SyncAll(env0);
                // fmt::print("{}: Prepared U12\n", env.Rank());
                // fflush(stdout);
                //env0.Sync();

                /*fmt::print("{}: k0:{}, L11{}\n", env.Rank(), k0, json(L11).dump());

                for (size_t i = 0; i < std::min(b, n - k0 - 1); ++i)
                {
                for (size_t j = 0; j < std::min(b, n - k0 - 1); ++j)
                {
                double expected = j < i ? 0.5 : 0.0;
                SyncLibInternal::Assert(std::abs(L11(i, j) - expected) < 1.0e-10,
                "L11 in processor {} was incorrect at ({}, {}): was {}, but should be {}\n",
                env.Rank(), i, j, L11(i, j), expected);
                }
                }

                */


                ReconstructU12(A, b, k0, rk0, rk1, ck1, nlc, s0, P, state.L11, U12, state.sqRowK);

                /*SyncLibInternal::Assert(arma::norm(U12 - arma::ones(arma::size(U12))) < 1.0e-10, "U12 was incorrect");
                SyncLibInternal::Assert(arma::norm(L21 - arma::ones(arma::size(L21)) * 0.5) < 1.0e-10, "L21 was incorrect");*/
            }

            // fmt::print("{}: Reconstructed U12\n", env.Rank());
            // fflush(stdout);

            // Update A22
            subEnv.Barrier();

            if (ck1 < nlc && rk1 < nlr)
            {
                tCore::UpdateA22(subEnv, A, U12, L21, ck1, nlc, rk1, nlr, state.tmpMem);
            }

            subEnv.Barrier();


            // fmt::print("{}: Updated A22\n", env.Rank());
            // fflush(stdout);
        }
    }

    static void LU(tEnv &env, arma::mat &A, arma::uvec &pi, const ProcessorMatrix &P, size_t n, size_t b)
    {
        auto [s0, s1] = P(env.Rank());
        auto[env0, env1] = tCommunicator::MaybeSplit(env, P, s0, s1);

        struct LUState
        {
            size_t s0, s1;// ] = P(env.Rank());

            size_t nlr; //= A.n_cols;
            size_t nlc; //= A.n_rows;

            tEnv &env0, &env1;
            //auto[env0, env1] = tCommunicator::MaybeSplit(env, P, s0, s1);

            SharedValue<size_t> rGlob; //(env1);

            SendQueue<double, size_t, size_t> sqPivot; //(env0);
            SendQueue<size_t, arma::vec> sqSwap; //(env0);
            SendQueue<size_t, size_t, arma::vec> sqDelayedSwap; //(env0);

            SendQueue<size_t, arma::vec> sqRowK; //(env0);
            SendQueue<size_t, arma::rowvec> sqColK; //(env1);
            SendQueue<size_t, arma::rowvec> sqColKL11; //(env1);
            // SendQueue<size_t, arma::rowvec> sqColKRemaining; //(env);

            std::vector<std::tuple<size_t, size_t>> swaps; //(b);

            arma::mat L11; //(b, b, arma::fill::zeros);
            arma::mat L21mem; //(b, nlr);
            arma::mat U12mem; //(b, nlc);

            arma::mat tmpMem;  //(nlc - b / P.N, nlr - b / P.M);

            LUState(size_t iS0, size_t iS1, size_t iNlr, size_t iNlc, tEnv &iEnv0, tEnv &iEnv1, size_t iB, const ProcessorMatrix &P)
                : s0(iS0),
                  s1(iS1),
                  nlr(iNlr),
                  nlc(iNlc),
                  env0(iEnv0),
                  env1(iEnv1),
                  rGlob(env1),
                  sqPivot(env0),
                  sqSwap(env0),
                  sqDelayedSwap(env0),
                  sqRowK(env0),
                  sqColK(env1),
                  sqColKL11(env1),
                  swaps(iB),
                  L11(iB, iB, arma::fill::zeros),
                  L21mem(iB, iNlr),
                  U12mem(iB, iNlc),
                  tmpMem(iNlc - iB / P.N, iNlr - iB / P.M)
            {
            }
        } state(s0, s1, A.n_cols, A.n_rows, env0, env1, b, P);
        /*{
            s0, s1,
            A.n_cols, A.n_rows,
            &env0, &env1,
            env1,
            env0, env0, env0,
            env0, env1, env1,
            b,
            {b, b, arma::fill::zeros},
            {b, A.n_cols}, {b, A.n_rows},
            {A.n_rows - b / P.N, A.n_cols - b / P.N}
        };*/

        tSubEnv subEnv(2);
        subEnv.Run(LUImpl<decltype(state)>, env, A, pi, P, n, b, state);
    }

    static void InitializeTestA(arma::mat &A, size_t n, size_t s0, size_t s1, const ProcessorMatrix &P)
    {
        for (size_t i = 0; i < A.n_cols; ++i)
        {
            size_t iglob = GlobalRow(i, P, s0); /* Global row index in A */
            iglob = (iglob - 1 + n) % n;        /* Global row index in B */

            for (size_t j = 0; j < A.n_rows; ++j)
            {
                size_t jglob = GlobalCol(j, P, s1); /* Global column index in A and B */
                A(j, i) = (iglob <= jglob ? 0.5 * iglob + 1 : 0.5 * (jglob + 1));
            }
        }
    }

    static void CheckResultLU(arma::mat &A, arma::uvec &pi, size_t n, size_t s0, size_t s1, const ProcessorMatrix &P)
    {
        for (size_t i = 0; i < A.n_cols; ++i)
        {
            size_t iGlob = GlobalRow(i, P, s0);

            for (size_t j = 0; j < A.n_rows; ++j)
            {
                size_t jGlob = GlobalCol(j, P, s1);

                if (iGlob <= jGlob)
                {
                    SyncLibInternal::Assert(abs(A(j, i) - 1) <= 1.0e-10, "Entry ({}, {}) of U was incorrect, was {} but should be {}\n",
                                            iGlob, jGlob, A(j, i), 1);
                }
                else
                {
                    SyncLibInternal::Assert(abs(A(j, i) - 0.5) <= 1.0e-10, "Entry ({}, {}) of L was incorrect, was {} but should be {}\n",
                                            iGlob, jGlob, A(j, i), 0.5);
                }
            }
        }

        for (size_t i = 0; i < n; ++i)
        {
            SyncLibInternal::Assert(pi[i] == ((i + 1) % n),
                                    "Permutation vector was incorrect at index {}, was {} but should be {}", i, (i + 1) % n, pi[i]);
        }
    }

    static void LUTest(tEnv &env, const ProcessorMatrix &P, size_t n, size_t b)
    {
        auto[s0, s1] = P(env.Rank());
        arma::mat A(LocalCount(P.N, s1, n), LocalCount(P.M, s0, n));

        SyncLib::Util::Timer<> timer;

        {
            InitializeTestA(A, n, s0, s1, P);

            arma::uvec pi = arma::regspace<arma::uvec>(0, n - 1);

            env.Barrier();
            env.Barrier();
            timer.Tic();
            LU(env, A, pi, P, n, P.p * b);

            if (env.Rank() == 0)
            {
                fmt::print("LU factorisation of A: {} x {} with blocksize b={:>4} and processor count p={:>2} x {:>2}={} took {:.2f}s\n",
                           n, n, P.p * b, P.M, P.N, P.p, timer.Toc());
            }

            CheckResultLU(A, pi, n, s0, s1, P);
        }
    }
};

/*
template<typename tCommunicator, typename tEnv, typename tSubEnv, typename... tSubEnvs>
class LUAlgorithmClass
{
public:

    template<typename... tValues>
    using SendQueue = typename tEnv::template SendQueue<tValues...>;

    template<typename tT>
    using SharedValue = typename tEnv::template SharedValue<tT>;

    using tCore = LUCoreImpl<tSubEnv, tSubEnvs..., SyncLib::Environments::NoBSP>;

    static size_t LocalCount(size_t p, size_t s, size_t n)
    {
        return (n - s + p - 1) / p;
    }

    static size_t GlobalRow(size_t k, const ProcessorMatrix &P, size_t s0)
    {
        return k * P.M + s0;
    }

    static size_t GlobalCol(size_t k, const ProcessorMatrix &P, size_t s1)
    {
        return k * P.N + s1;
    }

    static void FindLocalPivot(arma::mat &A, size_t k, size_t kr0, size_t nlr, size_t kc0, size_t s0, size_t s1,
                               const ProcessorMatrix &P, SendQueue<double, size_t, size_t> &sqPivot)
    {
        if (k % P.N == s1 && kr0 < nlr)
        {
            / * Superstep 0 (0):
            Compute the local pivot element
            * /
            auto[r, a] = tCore::FindLocalPivot(A, kc0, kr0, nlr);
            r = GlobalRow(r, P, s0);

            / * Superstep 0 (1):
            Share the pivot element and its index with processors
            in the same processor column.
            * /
            tCommunicator::ColumnSendPivot(sqPivot, s0, s1, r, kr0, a, P);
        }
    }

    static void FindGlobalPivotAndDivideColumnK(arma::mat &A, size_t k, size_t kr0, size_t nlr, size_t kc0, size_t s0, size_t s1,
                                                const ProcessorMatrix &P, SendQueue<double, size_t, size_t> &sqPivot, SharedValue<size_t> &rGlob)
    {
        / * Superstep 1 (2):
        Compute the global pivot element, its value and its source
        * /
        if (k % P.N == s1)
        {
            double aMax = 0.0;
            size_t t0Max, rMax;

            for (auto[a, t0, r] : sqPivot)
            {
                if (std::abs(a) > aMax)
                {
                    aMax = a;
                    t0Max = t0;
                    rMax = r;
                }
            }

            / * Superstep 1 (2) continued:
            Divide column k of U by the pivot element
            * /
            if (std::abs(aMax) <= std::numeric_limits<float>::epsilon())
            {
                throw std::runtime_error("LU is singular at stage " + std::to_string(k));
            }

            rGlob = rMax;

            / * Superstep 1 (3):
            Share the pivot row index with processors in the same processor row
            * /
            tCommunicator::ShareR(rGlob, s0, P);

            if (kr0 < nlr)
            {
                tCore::DivideColumnK(A, rMax / P.M, kr0, nlr, kc0, aMax, t0Max, s0);
            }
        }
    }

    static void UpdatePermutation(size_t k, size_t k0, arma::uvec &pi, SharedValue<size_t> &rGlob,
                                  std::vector<std::tuple<size_t, size_t>> &swaps)
    {
        std::swap(pi[k], pi[rGlob]);

        swaps[k - k0] = std::make_tuple(k - k0, rGlob - k0);
    }

    static void ExchangeBlockRows(arma::mat &A, size_t k, size_t ck0, size_t ck1, size_t s0, size_t s1, const ProcessorMatrix &P,
                                  size_t rGlob, SendQueue<size_t, arma::vec> &sqSwap, SendQueue<size_t, arma::vec> &sqRowK)
    {
        if (k % P.M == s0 && ck0 < ck1)
        {
            size_t t0 = rGlob % P.M;
            arma::vec rowK(&A(ck0, k / P.M), ck1 - ck0, false, true);
            tCommunicator::SendSwap(sqSwap, t0, s1, rGlob / P.M, rowK, P);
        }

        if (rGlob % P.M == s0 && ck0 < ck1)
        {
            size_t t0 = k % P.M;
            arma::vec rowR(&A(ck0, rGlob / P.M), ck1 - ck0, false, true);
            tCommunicator::SendSwap(sqSwap, t0, s1, k / P.M, rowR, P);

            / *if (ck0 > k / P.N)
            {
            arma::vec subRowR(&rowR(0), ck0 - k / P.N, false, true);
            tCommunicator::SendRowToColumnMembers(sqRowK, s1, k, P, subRowR);
            }* /
        }
    }

    static void ShareColumnK(arma::mat &A, size_t k, size_t k0, size_t kr1, size_t nlr, size_t rk1, size_t kc0, size_t s0,
                             size_t s1, const ProcessorMatrix &P, SendQueue<size_t, arma::rowvec> &sqColK/ *,
                                                                                    SendQueue<size_t, arma::rowvec> &sqColKRemaining* /)
    {
        if (k % P.N == s1 && kr1 < nlr)
        {
            // Share our column with processors in the same processor column
            auto colK = A(kc0, arma::span(kr1, nlr - 1));

            / *for (size_t t1 = 0; t1 < P.N; ++t1)
            {
            sqColK.Send(P(s0, t1), k, colK);
            }* /
            tCommunicator::SendColumnToRowMembers(sqColK, s0, k, P, colK);

            / *if (kr1 < rk1)
            {
            // Share the column inside the block with everyone else
            for (size_t t0 = 0; t0 < P.M; ++t0)
            {
            // skip t0 == s0, he already received it in the loop above
            if (t0 == s0) { continue; }

            / *for (size_t t1 = 0; t1 < P.N; ++t1)
            {
            sqColKRemaining.Send(P(t0, t1), GlobalRow(kr1, P, s0) - k0, A(kc0, arma::span(kr1, rk1 - 1)));
            }* /
            tCommunicator::SendColumnToRemainingMembers(sqColKRemaining, t0, GlobalRow(kr1, P, s0) - k0, P, A(kc0, arma::span(kr1, rk1 - 1)));
            }
            }* /
        }
    }

    static void ShareBlockRowK(arma::mat &A, size_t k, size_t kr0, size_t ck0, size_t ck1, size_t s0, size_t s1,
                               const ProcessorMatrix &P, SendQueue<size_t, arma::vec> &sqRowK)
    {
        if (k % P.M == s0 && ck0 < ck1)
        {
            // Share our row with processors in the same processor column
            auto rowK = A(arma::span(ck0, ck1 - 1), kr0);

            / *for (size_t t0 = 0; t0 < P.M; ++t0)
            {
            sqRowK.Send(P(t0, s1), k, rowK);
            }* /
            tCommunicator::SendRowToColumnMembers(sqRowK, s1, k, P, rowK);
        }
    }

    static void UpdateRemainingBlock(arma::mat &A, size_t k, size_t k0, size_t kr1, size_t rk1, size_t nlr, size_t ck0, size_t kc1,
                                     size_t ck1, size_t s0, size_t s1, arma::mat &L11, arma::mat &L21, arma::mat &tmpMem, const ProcessorMatrix &P,
                                     SendQueue<size_t, arma::rowvec> &sqColK, SendQueue<size_t, arma::vec> &sqRowK/ *,
                                                                                     SendQueue<size_t, arma::rowvec> &sqColKRemaining* /)
    {
        for (auto[kj, colKj] : sqColK)
        {
            for (auto[ki, rowKi] : sqRowK)
            {
                if (kc1 < ck1)
                {
                    arma::vec rowKi1(&rowKi(kc1 - ck0), ck1 - kc1, false, true);
                    / *arma::mat Lkdiff(&tmpMem(0), rowKi1.n_rows, colKj.n_cols, false, true);

                    Lkdiff = rowKi1 * colKj;
                    A(arma::span(kc1, ck1 - 1), arma::span(kr1, nlr - 1)) -= Lkdiff;* /
                    tCore::UpdateA21(A, rowKi1, colKj, tmpMem, kc1, ck1, kr1, nlr);
                }

                if (ck0 < kc1 && k > k0)
                {
                    arma::rowvec rowKi0(&rowKi(0), kc1 - ck0, false, true);

                    for (size_t j = 0, jj = GlobalCol(ck0, P, s1) - k0; j < rowKi0.n_elem && jj < k - k0; ++j, jj += P.N)
                    {
                        L11(k - k0, jj) = rowKi0(j);
                    }
                }
            }

            if (rk1 < nlr)
            {
                arma::rowvec subColKj(&colKj(rk1 - kr1), nlr - rk1, false, true);
                L21(k - k0, arma::span(0, nlr - rk1 - 1)) = subColKj;
            }
        }

        / *for (auto[rt0, colKj] : sqColKRemaining)
        {
        for (size_t i = 0, ri = rt0; i < colKj.n_elem; ++i, ri += P.M)
        {
        L11(ri, k - k0) = colKj(i);
        }
        }* /
    }

    static void ReconstructSwaps(arma::uvec &oldRows, arma::uvec &newRows, size_t k0, size_t swapCount,
                                 std::vector<std::tuple<size_t, size_t>> &swaps)
    {
        std::map<size_t, size_t> swapMap;

        size_t memberCount = swapCount;

        for (size_t l = 0; l < swapCount; ++l)
        {
            auto &[x, y] = swaps[l];
            oldRows(l) = x;

            if (y >= swapCount)
            {
                auto it = swapMap.find(y);

                if (it == swapMap.end())
                {
                    swapMap.insert({ y, memberCount });
                    oldRows(memberCount) = y;
                    y = memberCount++;
                }
                else
                {
                    y = it->second;
                }
            }
        }

        oldRows.resize(memberCount);
        newRows.resize(memberCount);

        oldRows += k0;
        newRows = oldRows;

        for (auto &[x, y] : swaps)
        {
            std::swap(newRows(x), newRows(y));
        }
    }

    static void PerformDelayedSwaps(arma::mat &A, arma::uvec &oldRows, arma::uvec &newRows, size_t ck0, size_t ck1,
                                    size_t nlc, size_t s0, size_t s1, const ProcessorMatrix &P, SendQueue<size_t, size_t, arma::vec> &sqDelayedSwap)
    {
        for (size_t l = 0; l < newRows.n_rows; ++l)
        {
            size_t k = oldRows(l);
            size_t r = newRows(l);

            if (r % P.M == s0)
            {
                // If I am the owner of the new value for row k, send it to the old owner of row k
                arma::vec rowR(&A(0, r / P.M), nlc, false, true);

                if (ck0 > 0)
                {
                    arma::vec rowR0(&rowR(0), ck0, false, true);
                    // sqDelayedSwap.Send(P(k % P.M, s1), k / P.M, 0, rowR0);
                    tCommunicator::SendDelayedSwap(sqDelayedSwap, s1, k, 0, P, rowR0);
                }

                if (ck1 < nlc)
                {
                    arma::vec rowR2(&rowR(ck1), nlc - ck1, false, true);
                    //sqDelayedSwap.Send(P(k % P.M, s1), k / P.M, ck1, rowR2);
                    tCommunicator::SendDelayedSwap(sqDelayedSwap, s1, k, ck1, P, rowR2);
                }
            }
        }
    }

    static void FixL21Rows(arma::mat &A, size_t s0, size_t s1, size_t ck0, size_t ck1, size_t k1, arma::uvec &oldRows,
                           const ProcessorMatrix &P, SendQueue<size_t, arma::rowvec> &sqColK)
    {
        for (auto k : oldRows)
        {
            if (k % P.M == s0 && k >= k1)
            {
                // If I am in row k, and k > k1, reconstruct this row
                //arma::vec rowK(&A(ck0, k / P.M), ck1 - ck0, false, true);
                //auto rowK = A(arma::span(ck0, ck1 - 1), k / P.M).t();
                tCommunicator::SendColumnToRowMembers(sqColK, s0, k / P.M * P.N + s1, P, A(arma::span(ck0, ck1 - 1), k / P.M).t());
            }
        }
    }

    static void ShareRemainingRows(arma::mat &A, size_t k0, size_t k1, size_t ck1, size_t nlc, size_t s0, size_t s1,
                                   const ProcessorMatrix &P, SendQueue<size_t, arma::vec> &sqRowK)
    {
        for (size_t k = k0; k < k1; ++k)
        {
            if (k % P.M == s0 && ck1 < nlc)
            {
                // Share our column with processors in the same processor column
                auto rowK = A(arma::span(ck1, nlc - 1), k / P.M);

                / *for (size_t t0 = 0; t0 < P.M; ++t0)
                {
                sqRowK.Send(P(t0, s1), k, rowK);
                }* /
                tCommunicator::SendRowToColumnMembers(sqRowK, s1, k, P, rowK);
            }
        }
    }

    static void CompleteL11(arma::mat &L11, size_t k0, size_t ck0, size_t s0, size_t s1, const ProcessorMatrix &P,
                            SendQueue<size_t, arma::rowvec> &sqColK)
    {
        for (size_t j = GlobalCol(ck0, P, s1) - k0; j < L11.n_cols; j += P.N)
        {
            tCommunicator::SendColumnToRowMembers(sqColK, s0, j, P, L11.col(j).t());
        }
    }

    static void ReconstructU12(arma::mat &A, size_t b, size_t k0, size_t rk0, size_t rk1, size_t ck1, size_t nlc, size_t s0,
                               const ProcessorMatrix &P, arma::mat &L11, arma::mat &U12, SendQueue<size_t, arma::vec> &sqRowK)
    {
        for (auto[k, rowK] : sqRowK)
        {
            U12(k - k0, arma::span(0, rowK.n_elem - 1)) = rowK.t();
        }

        for (size_t i = 2; i < b; ++i)
        {
            for (size_t j = 0; j < i - 1; ++j)
            {
                L11(i, j) -= arma::as_scalar(L11(i, arma::span(j + 1, i - 1)) * L11(arma::span(j + 1, i - 1), j));
            }
        }

        U12 -= L11 * U12;

        // Update our part of A with the updated value
        if (ck1 < nlc)
        {
            for (size_t k = rk0, kr = GlobalRow(rk0, P, s0) - k0; k < rk1; ++k, kr += P.M)
            {
                A(arma::span(ck1, nlc - 1), k) = U12.row(kr).t();
            }
        }
    }

    void LU(tEnv &env, arma::mat &A, arma::uvec &pi, const ProcessorMatrix &P, size_t n, size_t b)
    {
        SharedValue<size_t> rGlob(env1);

        for (size_t k0 = 0, block = 0; k0 < n; k0 += b, ++block)
        {
            const double foo = A.max();
            const size_t k1 = std::min(k0 + b, n);
            const size_t rk0 = LocalCount(P.M, s0, k0);
            const size_t ck0 = LocalCount(P.N, s1, k0);
            const size_t rk1 = std::min(nlr, LocalCount(P.M, s0, k1));
            const size_t ck1 = std::min(nlc, LocalCount(P.N, s1, k1));

            arma::mat L21(&L21mem(0), b, nlr - rk1, false, true);
            arma::mat U12(&U12mem(0), b, nlc - ck1, false, true);

            size_t swapCount = k1 - k0;
            swaps.resize(swapCount);

            for (size_t k = k0; k < k1; ++k)
            {
                size_t kr0 = LocalCount(P.M, s0, k);
                size_t kr1 = LocalCount(P.M, s0, k + 1);
                size_t kc0 = LocalCount(P.N, s1, k);
                size_t kc1 = LocalCount(P.N, s1, k + 1);

                // fmt::print("{}: k={}\n", env.Rank(), k);
                // fflush(stdout);
                FindLocalPivot(A, k, kr0, nlr, kc0, s0, s1, P, sqPivot);
                env0.Sync();

                FindGlobalPivotAndDivideColumnK(A, k, kr0, nlr, kc0, s0, s1, P, sqPivot, rGlob);
                env1.Sync();

                // fmt::print("{}: r={}\n", env.Rank(), rGlob.Value());
                // fflush(stdout);
                UpdatePermutation(k, k0, pi, rGlob, swaps);

                / * Superstep 2 (4):
                Exchange rows k and r between processors, but only columns k0 <= k < k1
                * /
                ExchangeBlockRows(A, k, ck0, ck1, s0, s1, P, rGlob, sqSwap, sqRowK);
                env0.Sync();
                // fmt::print("{}: rows exchanged\n", env.Rank());
                // fflush(stdout);

                / * Superstep 3 (5):
                Receive the rows and update the permutation vector
                * /
                for (auto[i, Ai] : sqSwap)
                {
                    arma::vec rowI(&A(ck0, i), ck1 - ck0, false, true);
                    rowI = Ai;
                }

                / * Superstep 3 (6):
                Broadcast our part of the column to processors in the same column
                Broadcast our part of the row to processors in the same row.
                * /
                ShareColumnK(A, k, k0, kr1, nlr, rk1, kc0, s0, s1, P, sqColK);
                ShareBlockRowK(A, k, kr0, ck0, ck1, s0, s1, P, sqRowK);
                tCommunicator::SyncAll(env0, env1);
                // fmt::print("{}: rows&column duplicated\n", env.Rank());
                // fflush(stdout);

                // fmt::print("{}: updating A21\n", env.Rank());
                // fflush(stdout);
                / * Superstep 4 (7):
                Receive the row and column parts, multiply them, subtract them.
                * /
                UpdateRemainingBlock(A, k, k0, kr1, rk1, nlr, ck0, kc1, ck1, s0, s1, L11, L21, tmpMem, P, sqColK, sqRowK);
                // fmt::print("{}: updated A22\n", env.Rank());
                // fflush(stdout);
            }

            // fmt::print("{}: Reconstructing swaps\n", env.Rank());
            // fflush(stdout);
            arma::uvec oldRows(swapCount * 2);
            arma::uvec newRows(swapCount * 2);

            ReconstructSwaps(oldRows, newRows, k0, swapCount, swaps);

            // fmt::print("{}: Reconstructed swaps\n", env.Rank());
            // fflush(stdout);
            PerformDelayedSwaps(A, oldRows, newRows, ck0, ck1, nlc, s0, s1, P, sqDelayedSwap);

            // fmt::print("{}: Scheduling delayed swaps\n", env.Rank());
            // fflush(stdout);
            FixL21Rows(A, s0, s1, ck0, ck1, k1, oldRows, P, sqColK);

            // fmt::print("{}: Completing L11\n", env.Rank());
            // fflush(stdout);
            CompleteL11(L11, k0, ck0, s0, s1, P, sqColKL11);

            // fmt::print("{}: Completed L11\n", env.Rank());
            // fflush(stdout);
            tCommunicator::SyncAll(env0, env1);
            // fmt::print("{}: Prepared A12, L21\n", env.Rank());
            // fflush(stdout);

            for (auto[i, j, Ai] : sqDelayedSwap)
            {
                arma::vec rowIPart(&A(j, i / P.M), Ai.n_elem, false, true);
                rowIPart = Ai;
            }

            for (auto[x, rowK] : sqColK)
            {
                size_t k = (x / P.N) * P.M + s0 - k1;
                size_t t1 = x % P.N;

                for (size_t j = 0, jj = t1; j < rowK.n_elem; ++j, jj += P.N)
                {
                    L21(jj, k) = rowK(j);
                }
            }

            for (auto[j, colJ] : sqColKL11)
            {
                L11.col(j) = colJ.t();
            }

            / * Superstep 3 (6):
            Broadcast our part of the column to processors in the same column
            Broadcast our part of the row to processors in the same row.
            * /
            ShareRemainingRows(A, k0, k1, ck1, nlc, s0, s1, P, sqRowK);
            tCommunicator::SyncAll(env0);
            // fmt::print("{}: Prepared U12\n", env.Rank());
            // fflush(stdout);
            //env0.Sync();

            / *fmt::print("{}: k0:{}, L11{}\n", env.Rank(), k0, json(L11).dump());

            for (size_t i = 0; i < std::min(b, n - k0 - 1); ++i)
            {
            for (size_t j = 0; j < std::min(b, n - k0 - 1); ++j)
            {
            double expected = j < i ? 0.5 : 0.0;
            SyncLibInternal::Assert(std::abs(L11(i, j) - expected) < 1.0e-10,
            "L11 in processor {} was incorrect at ({}, {}): was {}, but should be {}\n",
            env.Rank(), i, j, L11(i, j), expected);
            }
            }

            * /


            ReconstructU12(A, b, k0, rk0, rk1, ck1, nlc, s0, P, L11, U12, sqRowK);
            // fmt::print("{}: Reconstructed U12\n", env.Rank());
            // fflush(stdout);

            / *SyncLibInternal::Assert(arma::norm(U12 - arma::ones(arma::size(U12))) < 1.0e-10, "U12 was incorrect");
            SyncLibInternal::Assert(arma::norm(L21 - arma::ones(arma::size(L21)) * 0.5) < 1.0e-10, "L21 was incorrect");* /

            // Update A22
            if (ck1 < nlc && rk1 < nlr)
            {
                tCore::UpdateA22(A, U12, L21, ck1, nlc, rk1, nlr, tmpMem);
            }

            // fmt::print("{}: Updated A22\n", env.Rank());
            // fflush(stdout);
        }
    }

    static void InitializeTestA(arma::mat &A, size_t n, size_t s0, size_t s1, const ProcessorMatrix &P)
    {
        for (size_t i = 0; i < A.n_cols; ++i)
        {
            size_t iglob = GlobalRow(i, P, s0); / * Global row index in A * /
            iglob = (iglob - 1 + n) % n;        / * Global row index in B * /

            for (size_t j = 0; j < A.n_rows; ++j)
            {
                size_t jglob = GlobalCol(j, P, s1); / * Global column index in A and B * /
                A(j, i) = (iglob <= jglob ? 0.5 * iglob + 1 : 0.5 * (jglob + 1));
            }
        }
    }

    static void CheckResultLU(arma::mat &A, arma::uvec &pi, size_t n, size_t s0, size_t s1, const ProcessorMatrix &P)
    {
        for (size_t i = 0; i < A.n_cols; ++i)
        {
            size_t iGlob = GlobalRow(i, P, s0);

            for (size_t j = 0; j < A.n_rows; ++j)
            {
                size_t jGlob = GlobalCol(j, P, s1);

                if (iGlob <= jGlob)
                {
                    SyncLibInternal::Assert(abs(A(j, i) - 1) <= 1.0e-10, "Entry ({}, {}) of U was incorrect, was {} but should be {}\n",
                                            iGlob, jGlob, A(j, i), 1);
                }
                else
                {
                    SyncLibInternal::Assert(abs(A(j, i) - 0.5) <= 1.0e-10, "Entry ({}, {}) of L was incorrect, was {} but should be {}\n",
                                            iGlob, jGlob, A(j, i), 0.5);
                }
            }
        }

        for (size_t i = 0; i < n; ++i)
        {
            SyncLibInternal::Assert(pi[i] == ((i + 1) % n),
                                    "Permutation vector was incorrect at index {}, was {} but should be {}", i, (i + 1) % n, pi[i]);
        }
    }

    static void LUTest(tEnv &env, const ProcessorMatrix &P, size_t n, size_t b)
    {
        auto[s0, s1] = P(env.Rank());
        arma::mat A(LocalCount(P.N, s1, n), LocalCount(P.M, s0, n));

        SyncLib::Util::Timer<> timer;

        {
            InitializeTestA(A, n, s0, s1, P);

            arma::uvec pi = arma::regspace<arma::uvec>(0, n - 1);

            env.Barrier();
            env.Barrier();
            timer.Tic();
            LU(env, A, pi, P, n, P.p * b);

            if (env.Rank() == 0)
            {
                fmt::print("LU factorisation of A: {}x{} with blocksize b={:>4} and processor count p={} took {:.2f}s\n",
                           n, n, P.p * b, P.p, timer.Toc());
            }

            CheckResultLU(A, pi, n, s0, s1, P);
        }
    }

    LUAlgorithmClass(tEnv &baseEnv, const ProcessorMatrix &iP, size_t iS0, size_t iS1, arma::mat &iA, size_t iB)
        : env(baseEnv),
          env0(iP.Split0(baseEnv, iS0, iS1)),
          env1(iP.Split1(baseEnv, iS0, iS1)),
          P(iP),
          s0(iS0),
          s1(iS1),
          S(env, env0, env1),
          state( { iB, {iB, iB, arma::fill::zeros}, {iB, iA.n_cols}, {iB, iA.n_rows}, {iA.n_rows - b / iP.N, iA.n_cols - b / P.M} }),
    data({ iA, iA.n_cols, A.n_rows, iB })
    {}

private:

    tEnv &env;
    const ProcessorMatrix &P;
    size_t s0, size_t s1;

    struct
    {
        std::vector<std::tuple<size_t, size_t>> swaps;

        arma::mat L11;
        arma::mat L21mem;
        arma::mat U12mem;

        arma::mat tmpMem;
    } state;

    struct
    {
        arma::mat &A;
        size_t nlr;
        size_t nlc;
        size_t b;
    } data;

    struct LUSendQueues
    {
        SendQueue<double, size_t, size_t> pivot;
        SendQueue<size_t, arma::vec> swap;
        SendQueue<size_t, size_t, arma::vec> delayedSwap;

        SendQueue<size_t, arma::vec> rowK;
        SendQueue<size_t, arma::rowvec> colK;
        SendQueue<size_t, arma::rowvec> colKL11;

        LUSendQueues(tEnv &env, tEnv &env0, tEnv &env1)
            : pivot(env0),
              swap(env0),
              delayedSwap(env0),
              rowK(env0),
              colK(env1),
              colKL11(env1)
        {}
    } S;
};*/

template<typename... tSubEnvs>
struct LUCoreImpl<SyncLib::Environments::NoBSP, tSubEnvs...>
{
    static std::tuple<size_t, double> FindLocalPivot(arma::mat &A, size_t kc0, size_t kr0, size_t nlr)
    {
        auto colK = A(kc0, arma::span(kr0, nlr - 1));
        auto absColK = arma::abs(colK);

        size_t r = arma::index_max(absColK);

        return { r + kr0, colK(r) };
    }

    static void DivideColumnK(arma::mat &A, size_t r, size_t kr0, size_t nlr, size_t kc0, double a, size_t t0, size_t s0)
    {
        auto colK = A(kc0, arma::span(kr0, nlr - 1));
        colK /= a;

        if (t0 == s0)
        {
            colK(r - kr0) = a;
        }
    }

    static void UpdateA22(SyncLib::Environments::NoBSP &, arma::mat &A, arma::mat &U12, arma::mat &L21, size_t ck1, size_t nlc,
                          size_t rk1, size_t nlr,
                          arma::mat &tmpMem)
    {
        arma::mat A22diff(&tmpMem(0), U12.n_cols, L21.n_cols, false, true);
        A22diff = U12.t() * L21;

        A(arma::span(ck1, nlc - 1), arma::span(rk1, nlr - 1)) -= A22diff;
    }

    static void UpdateA21(arma::mat &A, arma::vec &rowK, arma::rowvec &colK, arma::mat &tmpMem, size_t kc1, size_t ck1, size_t kr1,
                          size_t nlr)
    {
        arma::mat Lkdiff(&tmpMem(0), rowK.n_rows, colK.n_cols, false, true);

        Lkdiff = rowK * colK;
        A(arma::span(kc1, ck1 - 1), arma::span(kr1, nlr - 1)) -= Lkdiff;
    }
};

template<typename... tSubEnvs>
struct LUCoreImpl<SyncLib::Environments::SharedMemoryBSP, tSubEnvs...>
    : public LUCoreImpl<tSubEnvs...>
{
    using tEnv = SyncLib::Environments::SharedMemoryBSP;
    using tParent = LUCoreImpl<tSubEnvs...>;

    static void UpdateA22(tEnv &env, arma::mat &A, arma::mat &U12, arma::mat &L21, size_t ck1, size_t nlc, size_t rk1, size_t nlr,
                          arma::mat &tmpMem)
    {
        size_t s = env.Rank();
        size_t p = env.Size();

        size_t count = nlr - rk1;
        size_t localCount = (count + p - 1) / p;
        size_t start = rk1 + s * localCount;
        size_t end = std::min(start + localCount, nlr);

        if (start < end)
        {
            arma::mat L21loc(&L21(0, start - rk1), L21.n_rows, end - start, false, true);
            auto claim = tmpMemPool.Allocate(U12.n_cols, end - start);
            arma::mat &A22diff = claim.mat;
            A22diff = U12.t() * L21loc;

            A(arma::span(ck1, nlc - 1), arma::span(start, end - 1)) -= A22diff;
        }
    }

    static void UpdateA21Impl(tEnv &env, arma::mat &A, arma::vec &rowK, arma::rowvec &colK, arma::mat &tmpMem, size_t kc1, size_t ck1,
                              size_t kr1,
                              size_t nlr)
    {
        size_t s = env.Rank();
        size_t p = env.Size();

        size_t count = nlr - kr1;
        size_t localCount = (count + p - 1) / p;
        size_t start = kr1 + s * localCount;
        size_t end = std::min(start + localCount, nlr);

        if (start < end)
        {
            arma::rowvec colKloc(&colK(start - kr1), end - start, false, true);
            auto claim = tmpMemPool.Allocate(rowK.n_elem, colKloc.n_elem);
            arma::mat &A21diff = claim.mat;
            A21diff = rowK * colKloc;

            A(arma::span(kc1, ck1 - 1), arma::span(start, end - 1)) -= A21diff;
        }
    }

    /*static void UpdateA22(arma::mat &A, arma::mat &U12, arma::mat &L21, size_t ck1, size_t nlc, size_t rk1, size_t nlr,
                          arma::mat &tmpMem)
    {
        tEnv subEnv(2);
        subEnv.Run(UpdateA22Impl, A, U12, L21, ck1, nlc, rk1, nlr, tmpMem);
        return;



        arma::mat A22diff(&tmpMem(0), U12.n_cols, L21.n_cols, false, true);
        A22diff = U12.t() * L21;

        A(arma::span(ck1, nlc - 1), arma::span(rk1, nlr - 1)) -= A22diff;
    }*/

    static void UpdateA21(arma::mat &A, arma::vec &rowK, arma::rowvec &colK, arma::mat &tmpMem, size_t kc1, size_t ck1, size_t kr1,
                          size_t nlr)
    {
        /*tEnv subEnv(2);
        subEnv.Run(UpdateA21Impl, A, rowK, colK, tmpMem, kc1, ck1, kr1, nlr);
        return;*/
        arma::mat Lkdiff(&tmpMem(0), rowK.n_rows, colK.n_cols, false, true);

        Lkdiff = rowK * colK;
        A(arma::span(kc1, ck1 - 1), arma::span(kr1, nlr - 1)) -= Lkdiff;
    }
};

template<typename tEnv>
void RunLUTest(tEnv &env, std::vector<uint32_t> &M, std::vector<uint32_t> &N, bool split, uint32_t n, uint32_t b,
               uint32_t iterations)
{
    ProcessorMatrix P(env.Size(), M[0], N[0]);

    for (size_t i = 0; i < iterations; ++i)
    {
        // fmt::print("{}: {}, {}\n", env.Rank(), M[0], N[0]);

        if (M.size() > 1 || N.size() > 1)
        {
            using tSubEnv = SyncLib::Environments::SharedMemoryBSP;

            size_t pSub = 1;

            if (M.size() > 1)
            {
                pSub *= M[1];
            }

            if (N.size() > 1)
            {
                pSub *= N[1];
            }

            if (split)
            {
                using tAlgorithm = LUAlgorithm<LUSplitCommunicator, tEnv, tSubEnv>;
                env.Run(tAlgorithm::LUTest, P, n, b);
            }
            else
            {
                using tAlgorithm = LUAlgorithm<LUCommunicator, tEnv, tSubEnv>;
                env.Run(tAlgorithm::LUTest, P, n, b);
            }
        }
        else
        {
            using tSubEnv = SyncLib::Environments::NoBSP;

            if (split)
            {
                using tAlgorithm = LUAlgorithm<LUSplitCommunicator, tEnv, tSubEnv>;
                env.Run(tAlgorithm::LUTest, P, n, b);
            }
            else
            {
                using tAlgorithm = LUAlgorithm<LUCommunicator, tEnv, tSubEnv>;
                env.Run(tAlgorithm::LUTest, P, n, b);
            }
        }
    }
}

int main(int argc, char **argv)
{
    //     fmt::print("wef\n");
    //     fflush(stdout);
    //     SyncLib::MPI::Comm comm(argc, argv);
    //     fmt::print("foo\n");
    //     fflush(stdout);
    //     std::vector<int> foo(comm.Size(), 0);
    //     std::vector<int> bar(comm.Size(), 0);
    //     SyncLib::MPI::NonBlockingRequest req;
    //     //int size = 1;
    //     comm.AllToAllNonBlocking(&foo[0], &bar[0], 1, req);
    //     //MPI_Ialltoall(&foo[0], 1, MPI_INT, &bar[0], 1, MPI_INT, MPI_COMM_WORLD, &req.request);
    //     fmt::print("bar\n");
    //     fflush(stdout);
    //     req.Wait();
    //     fmt::print("{}: {}\n", comm.Rank(), json(arma::Col<int>(bar)).dump());
    //     fflush(stdout);
    //     return 0;

    //     MPI_Init(&argc, &argv);
    //     MPI_Comm comm;
    //     int size;
    //     MPI_Comm_size(MPI_COMM_WORLD, &size);
    //     std::vector<int> sizes(size * (size - 1) / 2, 2);
    //     std::vector<int> periods(size * (size - 1) / 2, 0);
    //     MPI_Cart_create(MPI_COMM_WORLD, 1, sizes.data(), periods.data(), 0, &comm);
    //     fmt::print("foo\n");
    //     fflush(stdout);
    //     int p, s;
    //     MPI_Comm_rank(MPI_COMM_WORLD, &s);
    //     p = size;
    //
    //     if (s == 0)
    //     {
    //         system("pause");
    //     }
    //
    //     fmt::print("{}, {}\n", p, s);
    //     fflush(stdout);
    //
    //     MPI_Barrier(MPI_COMM_WORLD);
    //     std::vector<double> foo(p, 1.0 * s);
    //     std::vector<double> bar(p, -1.0);
    //     std::vector<int> counts(p, 1);
    //     std::vector<MPI_Aint> sdispl(p);
    //     std::vector<MPI_Aint> rdispl(p);
    //     MPI_Aint sbegin, rbegin;
    //     MPI_Get_address(&foo[0], &sbegin);
    //     MPI_Get_address(&bar[0], &rbegin);
    //
    //     for (int t = 0; t < p; ++t)
    //     {
    //         MPI_Aint saddr, raddr;
    //         MPI_Get_address(&foo[t], &saddr);
    //         MPI_Get_address(&bar[t], &raddr);
    //         sdispl[t] = MPI_Aint_diff(saddr, sbegin);
    //         rdispl[t] = MPI_Aint_diff(raddr, rbegin);
    //     }
    //
    //     std::vector<MPI_Datatype> types(p, MPI_DOUBLE);
    //     SyncLib::MPI::NonBlockingRequest req;
    //     MPI_Ineighbor_alltoallw(&foo[0], counts.data(), sdispl.data(), types.data(), &bar[0], counts.data(), rdispl.data(), types.data(),
    //                             comm, &req.request);
    //     req.Wait();
    //     fmt::print("{}\n", json(bar).dump());
    //     fflush(stdout);
    //     MPI_Barrier(MPI_COMM_WORLD);
    //     return 0;


    /*SyncLib::Environments::DistributedBSP eenv(argc, argv);
    fmt::print("{}\n", eenv.Rank());
    Args args("High-performance LU decomposition");
    args.AddOptions(
    {
        { "M", "The number of rows in the processor matrix", Option::U32List(), "1", "1", },
        { "N", "The number of columns in the processor matrix", Option::U32List(), "1", "1" },
        { { "s", "split" }, "Whether to split the environment into subset-environments", Option::Boolean(), "true", "true" },
        { "n", "The matrix size", Option::U32(), "1024", "1024" },
        { "b", "The block size multiplier, the real block size will be b*M*N", Option::U32(), "1", "1" },
        { "i", "The number of repetitions", Option::U32(), "1", "1" }
    });

    args.Parse(argc, argv);

    std::vector<uint32_t> argsM = args.GetOption("M").Get<std::vector<uint32_t>>();
    std::vector<uint32_t> argsN = args.GetOption("N").Get<std::vector<uint32_t>>();

    size_t M = argsM[0];
    size_t N = argsN[0];
    RunLUTest(eenv, argsM, argsN, args.GetOption("split"), args.GetOption("n"), args.GetOption("b"), args.GetOption("i"));
    return 0;*/

    //     /*
    //     using tEnv = SyncLib::Environments::DistributedBSP;
    //     using tSubEnv = SyncLib::Environments::SharedMemoryBSP;
    //     // using tSubEnv = SyncLib::Environments::NoBSP;
    //     /*/
    //     using tEnv = SyncLib::Environments::SharedMemoryBSP;
    //     using tSubEnv = SyncLib::Environments::SharedMemoryBSP;
    //     /**/
    //
    //     tEnv env(2);
    //     ProcessorMatrix P(env.Size(), env.Size() / 1, 1);
    //
    //     arma::mat A(4096, 4096);
    //
    //     //*
    //     using tAlgorithm = LUAlgorithm<LUSplitCommunicator, tEnv, tSubEnv>;
    //     env.Run(tAlgorithm::LUTest, P, A.n_rows);
    //     //env.Run(tAlgorithm::LUTest<tAlgorithm::MatMatBenchSplit>, P, A, B);
    //     /*/
    //     env.Run(tAlgorithm::MatMatTest<tAlgorithm::MatMatBench>, P, A, B);
    //     env.Run(tAlgorithm::MatMatTest<tAlgorithm::MatMatBenchSplit>, P, A, B);
    //     /**/
    //
    //     /*if (env.Rank() == 0)
    //     {
    //         SyncLib::Util::LinearCongruentialRandom random(42);
    //
    //         RandomizeMatrix(A, random);
    //         RandomizeMatrix(B, random);
    //
    //         SyncLib::Util::Timer<>timer;
    //         timer.Tic();
    //
    //         for (size_t it = 0; it < 4 * 1 * 1; ++it)
    //         {
    //             C = A * B.t();
    //         }
    //
    //         fmt::print("Single-Threaded multiplication took {}s\n", timer.Toc() / 4 * 1 * 1);
    //         fmt::print("Sum C sequential: {:9.4f}\n", arma::as_scalar(arma::sum(arma::sum(arma::abs(C), 0), 1)));
    //         fmt::print("Number of zeros (A, B, C): ({}, {}, {})\n",
    //                    A.n_elem - arma::nonzeros(A).eval().n_elem,
    //                    B.n_elem - arma::nonzeros(B).eval().n_elem,
    //                    C.n_elem - arma::nonzeros(C).eval().n_elem);
    //     }
    //     */
    //
    //     //system("pause");
    SyncLib::Environments::DistributedBSP mpiEnv(argc, argv);

    /*if (mpiEnv.Rank() == 0)
    {
        system("pause");
    }

    mpiEnv.Barrier();*/

    Args args("High-performance LU decomposition");
    args.AddOptions(
    {
        {"M", "The number of rows in the processor matrix", Option::U32List(), "1", "1",},
        {"N", "The number of columns in the processor matrix", Option::U32List(), "1", "1"},
        {{"s", "split"}, "Whether to split the environment into subset-environments", Option::Boolean(), "true", "true"},
        {"n", "The matrix size", Option::U32(), "1024", "1024"},
        {"b", "The block size multiplier, the real block size will be b*M*N", Option::U32(), "1", "1"},
        {"i", "The number of repetitions", Option::U32(), "1", "1"}
    });

    args.Parse(argc, argv);

    std::vector<uint32_t> argsM = args.GetOption("M").Get<std::vector<uint32_t>>();
    std::vector<uint32_t> argsN = args.GetOption("N").Get<std::vector<uint32_t>>();

    size_t M = argsM[0];
    size_t N = argsN[0];

    if (mpiEnv.Size() > 1)
    {
        RunLUTest(mpiEnv, argsM, argsN, args.GetOption("split"), args.GetOption("n"), args.GetOption("b"), args.GetOption("i"));
    }
    else
    {
        SyncLib::Environments::SharedMemoryBSP env(M * N);
        fmt::print("{}, {}, {}\n", M, N, M * N);
        RunLUTest(env, argsM, argsN, args.GetOption("split"), args.GetOption("n"), args.GetOption("b"), args.GetOption("i"));
    }
}
