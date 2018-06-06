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
#include "sync/sync.h"
#include "sync/util/random.h"
#include "sync/util/timer.h"

#include "args/args.h"

#include <armadillo>

#include <numeric>
#include <tuple>

template<typename tMat, typename tRandom>
void RandomizeMatrix(tMat &mat, tRandom &random)
{
    for (auto &entry : mat)
    {
        entry = random.NextDouble();
    }
}

struct ProcessorCube
{
    size_t p;
    size_t q0, q1, q2;

    ProcessorCube(const size_t iP, const size_t iQ0, const size_t iQ1, const size_t iQ2)
        : p(iP),
          q0(iQ0),
          q1(iQ1),
          q2(iQ2)
    {
        SyncLibInternal::Assert(q0 * q1 * q2 == p, "Dimensions mismatch, {} * {} * {} = {} != {}", q0, q1, q2, q0 * q1 * q2, p);
    }

    ProcessorCube(const size_t iP, const size_t iQ)
        : ProcessorCube(iP, iQ, iQ, iQ)
    {}

    std::tuple<size_t, size_t, size_t>operator()(size_t rank) const
    {
        return std::make_tuple(rank / (q1 * q2), (rank % (q1 * q2)) / q2, rank % q2);
    }

    size_t operator()(size_t s, size_t t, size_t u)const
    {
        return s * q1 * q2 + t * q2 + u;
    }

    template<typename tEnv>
    tEnv &Split0(tEnv &env, size_t s0, size_t s1, size_t s2) const
    {
        return env.Split(s2 * q1 + s1, s0);
    }

    template<typename tEnv>
    tEnv &Split1(tEnv &env, size_t s0, size_t s1, size_t s2) const
    {
        return env.Split(s0 * q2 + s2, s1);
    }

    template<typename tEnv>
    tEnv &Split2(tEnv &env, size_t s0, size_t s1, size_t s2) const
    {
        return env.Split(s0 * q1 + s1, s2);
    }

    template<typename tEnv>
    std::tuple<tEnv &, tEnv &, tEnv &> Split(tEnv &env, size_t s0, size_t s1, size_t s2) const
    {
        return std::tuple<tEnv &, tEnv &, tEnv &>(Split0(env, s0, s1, s2), Split1(env, s0, s1, s2), Split2(env, s0, s1, s2));
    }
};

template<typename tEnv>
struct MatMatHelper
{
    static size_t NumLocalBalanced(size_t n, size_t q, size_t s)
    {
        return (n - s + q - 1) / q;
    }

    static size_t OffsetGlobalBalanced(size_t n, size_t q, size_t s)
    {
        size_t offset = 0;

        for (size_t t = 0; t < s; ++t)
        {
            offset += NumLocalBalanced(n, q, t);
        }

        return offset;
    }

    static size_t NumLocal(size_t n, size_t q)
    {
        return (n + q - 1) / q;
    }

    template<typename tMat>
    static auto GetSubViewBalanced(tMat &M,
                                   size_t s0, size_t q0, size_t n0, size_t nl0,
                                   size_t s1, size_t q1, size_t n1, size_t nl1)
    {
        const arma::uword iOffset = OffsetGlobalBalanced(n0, q0, s0);
        const arma::uword jOffset = OffsetGlobalBalanced(n1, q1, s1);

        return M(arma::span(iOffset, iOffset + nl0 - 1), arma::span(jOffset, jOffset + nl1 - 1));
    }

    struct DimInfoContiguous
    {
        size_t n;
        size_t k;
        size_t m;
        size_t nl0;
        size_t nl1;
        size_t nl2;
        size_t nCol02;
        size_t nCol21;
        size_t nCol01;
        size_t nl02;
        size_t nl21;
        size_t nl01;

        static DimInfoContiguous Create(const size_t n, const size_t k, const size_t m, const ProcessorCube &P, const size_t s0,
                                        const size_t s1, const size_t s2)
        {
            const size_t nl0 = NumLocalBalanced(n, P.q0, s0);
            const size_t nl1 = NumLocalBalanced(m, P.q1, s1);
            const size_t nl2 = NumLocalBalanced(k, P.q2, s2);

            const size_t nCol02 = NumLocal(nl2, P.q1);
            const size_t nCol21 = NumLocal(nl1, P.q0);
            const size_t nCol01 = NumLocal(nl1, P.q2);

            const size_t nl02 = nl0 * nCol02;
            const size_t nl21 = nl2 * nCol21;
            const size_t nl01 = nl0 * nCol01;

            return { n, k, m, nl0, nl1, nl2, nCol02, nCol21, nCol01, nl02, nl21, nl01 };
        }
    };

    template<typename tT, typename tSendQueue>
    static void FanOutMatrixContiguous(arma::Mat<tT> &M, tSendQueue &sq, size_t q, size_t v0, size_t vStart, size_t vSkip,
                                       size_t sizeV, size_t sizeTotal)
    {
        const arma::uword start = v0 * sizeV;
        const arma::uword size = std::min<arma::uword>(start + sizeV, sizeTotal) - start;

        if (size == 0)
        {
            return;
        }

        arma::Col<tT>tmp(&M(start), size, false, true);

        for (size_t v = 0, rank1 = vStart; v < q; ++v, rank1 += vSkip)
        {
            if (v == v0)
            {
                continue;
            }

            sq.Send(rank1, start, tmp);
        }
    }

    template<typename tT, typename tSendQueue>
    static void FanInMatrixContiguous(arma::Mat<tT> &M, tSendQueue &sq)
    {
        for (auto [i, mVec] : sq)
        {
            arma::Col<tT>subM(&M(i), mVec.n_elem, false, true);
            subM = mVec;
        }
    }

    template<typename tT, typename tSendQueue>
    static void ReduceCContiuous(arma::Mat<tT> &C_st, tSendQueue &sq, const size_t s2, const size_t startT, const size_t q2,
                                 const DimInfoContiguous &dimInfo, tEnv &env)
    {
        {
            for (size_t t2 = 0, t = startT; t2 < q2; ++t2, ++t)
            {
                if (t2 == s2)
                {
                    continue;
                }

                const arma::uword start = t2 * dimInfo.nl01;
                const arma::uword size = std::min<arma::uword>(start + dimInfo.nl01, dimInfo.nl0 * dimInfo.nl1) - start;
                arma::Col<tT>subC(&C_st(start), size, false, true);

                sq.Send(t, start, subC);
            }
        }

        env.Sync();

        for (auto[i, cVec] : sq)
        {
            arma::Col<tT>subC(&C_st(i), cVec.n_elem, false, true);
            subC += cVec;
        }
    }
};

template<typename tSubEnv, typename... tSubEnvs>
struct MatMatCoreImpl;


template<typename tEnv, typename... tSubEnvs>
struct MatMatAlgorithm
{
    using tHelper = MatMatHelper<tEnv>;

    template<typename tDimInfo>
    static void MatMat(tEnv &env, arma::mat &A_su, arma::mat &B_ut, arma::mat &C_st,
                       tDimInfo &dimInfo, const ProcessorCube &P,
                       typename tEnv::template SendQueue<arma::uword, arma::vec> &sqA, typename tEnv::template SendQueue<arma::uword, arma::vec> &sqB,
                       typename tEnv::template SendQueue<arma::uword, arma::vec> &sqC)
    {
        auto[s0, s1, s2] = P(env.Rank());

        const size_t q0 = P.q0, q1 = P.q1, q2 = P.q2;

        const size_t nl0 = dimInfo.nl0, nl1 = dimInfo.nl1, nl2 = dimInfo.nl2,
                     nl02 = dimInfo.nl02, nl21 = dimInfo.nl21;

        tHelper::FanOutMatrixContiguous(A_su, sqA, q1, s1, s0 * q1 * q2 + s2, q2, nl02, nl0 * nl2);
        tHelper::FanOutMatrixContiguous(B_ut, sqB, q0, s0, s2 + s1 * q2, q1 * q2, nl21, nl2 * nl1);

        env.Sync();

        tHelper::FanInMatrixContiguous(A_su, sqA);
        tHelper::FanInMatrixContiguous(B_ut, sqB);

        MatMatCoreImpl<tSubEnvs..., SyncLib::Environments::NoBSP>::Compute(env, A_su, B_ut, C_st, dimInfo, P, sqC);
    }

    template<typename tDimInfo>
    static void MatMatSplit(tEnv &env, tEnv &env02, tEnv &env21, tEnv &env01,
                            arma::mat &A_su, arma::mat &B_ut, arma::mat &C_st,
                            tDimInfo &dimInfo, const ProcessorCube &P,
                            typename tEnv::template SendQueue<arma::uword, arma::vec> &sqA, typename tEnv::template SendQueue<arma::uword, arma::vec> &sqB,
                            typename tEnv::template SendQueue<arma::uword, arma::vec> &sqC)
    {
        auto[s0, s1, s2] = P(env.Rank());

        tHelper::FanOutMatrixContiguous(A_su, sqA, P.q1, s1, 0, 1, dimInfo.nl02, dimInfo.nl0 * dimInfo.nl2);
        env02.Sync();
        tHelper::FanInMatrixContiguous(A_su, sqA);

        tHelper::FanOutMatrixContiguous(B_ut, sqB, P.q0, s0, 0, 1, dimInfo.nl21, dimInfo.nl2 * dimInfo.nl1);
        env21.Sync();
        tHelper::FanInMatrixContiguous(B_ut, sqB);

        MatMatCoreImpl<tSubEnvs..., SyncLib::Environments::NoBSP>::Compute(env01, A_su, B_ut, C_st, dimInfo, P, sqC);
    }

    struct MatMatBench
    {
        template<typename tDimInfo>
        static void Run(size_t repetitions, SyncLib::Util::Timer<> &timer, tEnv &env, const ProcessorCube &P,
                        tDimInfo &dimInfo, arma::mat &A_su, arma::mat &B_ut, arma::mat &C_st)
        {
            typename tEnv::template SendQueue<arma::uword, arma::vec>sqA(env), sqB(env), sqC(env);

            for (size_t it = 0; it < repetitions; ++it)
            {
                env.Barrier();
                timer.Tic();
                MatMat(env, A_su, B_ut, C_st, dimInfo, P, sqA, sqB, sqC);

                if (env.Rank() == 0)
                {
                    fmt::print("Matrix multiplication with (n, k, m) = ({}, {}, {}) with processor count p={:>2} x {:>2} x {:>2}={} took {:.3f}s\n",
                               dimInfo.n, dimInfo.k, dimInfo.m, P.q0, P.q1, P.q2, P.p, timer.Toc());
                }
            }
        }
    };

    struct MatMatBenchSplit
    {
        template<typename tDimInfo>
        static void Run(size_t repetitions, SyncLib::Util::Timer<> &timer, tEnv &env, const ProcessorCube &P,
                        tDimInfo &dimInfo, arma::mat &A_su, arma::mat &B_ut, arma::mat &C_st)
        {
            const size_t rank = env.Rank();
            const auto[s0, s1, s2] = P(rank);

            auto [env21, env02, env01] = P.Split(env, s0, s1, s2);

            typename tEnv::template SendQueue<arma::uword, arma::vec> sqA(env02), sqB(env21), sqC(env01);

            for (size_t it = 0; it < repetitions; ++it)
            {
                env.Barrier();
                timer.Tic();
                MatMatSplit(env, env02, env21, env01, A_su, B_ut, C_st, dimInfo, P, sqA, sqB, sqC);

                if (env.Rank() == 0)
                {
                    fmt::print("Matrix multiplication with (n, k, m) = ({}, {}, {}) with processor count p={:>2} x {:>2} x {:>2}={} took {:.3f}s\n",
                               dimInfo.n, dimInfo.k, dimInfo.m, P.q0, P.q1, P.q2, P.p, timer.Toc());
                }
            }
        }
    };

    template<typename tBench>
    static void MatMatTest(tEnv &env, ProcessorCube &P, arma::mat &A, arma::mat &B, size_t repetitions)
    {
        const size_t rank = env.Rank();
        const auto[s0, s1, s2] = P(rank);

        size_t n = A.n_rows, m = A.n_cols, k = B.n_rows;
        const auto dimInfo = tHelper::DimInfoContiguous::Create(n, k, m, P, s0, s1, s2);

        arma::mat A_su(dimInfo.nl0, dimInfo.nl2, arma::fill::zeros);
        arma::mat B_ut(dimInfo.nl1, dimInfo.nl2, arma::fill::zeros);
        arma::mat C_st(dimInfo.nl0, dimInfo.nl1, arma::fill::zeros);

        SyncLib::Util::LinearCongruentialRandom random(rank);
        RandomizeMatrix(A_su, random);
        RandomizeMatrix(B_ut, random);

        SyncLib::Util::Timer<> timer;

        tBench::Run(repetitions, timer, env, P, dimInfo, A_su, B_ut, C_st);
        env.Barrier();

        typename tEnv::template SendQueue<double>sumC(env);
        {
            const arma::uword start = s2 * dimInfo.nl01;
            const arma::uword size = std::min<arma::uword>(start + dimInfo.nl01, dimInfo.nl0 * dimInfo.nl1) - start;
            const arma::vec subC(&C_st(start), size, false, true);
            sumC.Broadcast(arma::as_scalar(arma::sum(arma::abs(subC))));
        }
        env.Sync();
        // fmt::print("Sum C on s={:<5}: {:9.4f}\n", rank, std::accumulate(sumC.begin(), sumC.end(), 0.0));
        // fflush(stdout);
        env.Barrier();

        tHelper::GetSubViewBalanced(A, s0, P.q0, n, dimInfo.nl0, s2, P.q2, k, dimInfo.nl2) = A_su;
        tHelper::GetSubViewBalanced(B, s1, P.q1, m, dimInfo.nl1, s2, P.q2, k, dimInfo.nl2) = B_ut;
    }
};

template<typename... tSubEnvs>
struct MatMatCoreImpl<SyncLib::Environments::NoBSP, tSubEnvs...>
{
    template<typename tEnv, typename tMat>
    static void Compute(tEnv &env, const tMat &A_su, const tMat &B_ut, arma::mat &C_st,
                        const typename MatMatHelper<tEnv>::DimInfoContiguous &dimInfo, const ProcessorCube &P,
                        typename tEnv::template SendQueue<arma::uword, arma::vec> &sqC)
    {
        auto[s0, s1, s2] = P(env.Rank());

        C_st = A_su * B_ut.t();

        MatMatHelper<tEnv>::ReduceCContiuous(C_st, sqC, s2, s0 * P.q1 * P.q2 + s1 * P.q2, P.q2, dimInfo, env);
    }
};

template<typename... tSubEnvs>
struct MatMatCoreImpl<SyncLib::Environments::SharedMemoryBSP, tSubEnvs...>
{
    using tSubEnv = SyncLib::Environments::SharedMemoryBSP;
    using tHelper = MatMatHelper<tSubEnv>;

    static void MatMatShared(tSubEnv &env, const arma::mat &A, const arma::mat &B, arma::mat &C, const ProcessorCube &P)
    {
        auto[s0, s1, s2] = P(env.Rank());
        tSubEnv &env01 = P.Split2(env, s0, s1, s2);
        const tHelper::DimInfoContiguous dimInfo = tHelper::DimInfoContiguous::Create(A.n_rows, A.n_cols, B.n_rows, P, s0, s1, s2);

        arma::mat A_su = tHelper::GetSubViewBalanced(A, s0, P.q0, dimInfo.n, dimInfo.nl0, s2, P.q2, dimInfo.k, dimInfo.nl2);
        arma::mat B_ut = tHelper::GetSubViewBalanced(B, s1, P.q1, dimInfo.m, dimInfo.nl1, s2, P.q2, dimInfo.k, dimInfo.nl2);
        arma::mat C_st(A_su.n_rows, B_ut.n_rows);

        tSubEnv::SendQueue<arma::uword, arma::vec> sqC(env01);
        MatMatCoreImpl<tSubEnvs...>::Compute(env01, A_su, B_ut, C_st, dimInfo, P, sqC);

        auto sizeC = arma::size(C_st);

        const arma::uword iOffsetC = tHelper::OffsetGlobalBalanced(dimInfo.n, P.q0, s0);
        const arma::uword jOffsetC = tHelper::OffsetGlobalBalanced(dimInfo.m, P.q1, s1);

        for (arma::uword col = s2 * dimInfo.nCol01, colEnd = std::min<arma::uword>(col + dimInfo.nCol01, dimInfo.nl1);
             col < colEnd; ++col)
        {
            arma::vec subC(&C(iOffsetC, jOffsetC + col), dimInfo.nl0, false, true);
            arma::vec colC(&C_st(0, col), C_st.n_rows, false, true);
            subC = colC;
        }
    }

    template<typename tEnv, typename tMat>
    static void Compute(tEnv &env, const tMat &A_su, const tMat &B_ut, arma::mat &C_st,
                        const typename MatMatHelper<tEnv>::DimInfoContiguous &dimInfo, const ProcessorCube &P,
                        typename tEnv::template SendQueue<arma::uword, arma::vec> &sqC)
    {
        auto[s0, s1, s2] = P(env.Rank());

        using tSubEnv = SyncLib::Environments::SharedMemoryBSP;
        tSubEnv subEnv(2);
        ProcessorCube Psub(subEnv.Size(), 2, 1, 1);
        subEnv.Run(MatMatShared, A_su, B_ut, C_st, Psub);

        MatMatHelper<tEnv>::ReduceCContiuous(C_st, sqC, s2, s0 * P.q1 * P.q2 + s1 * P.q2, P.q2, dimInfo, env);
    }
};

#ifdef IS_WINDOWS
void Pause(bool pause = true)
{
    if (pause)
    {
        system("pause");
    }
}
#else
void Pause(bool)
{
}
#endif

template<typename tEnv>
void RunMatMatTest(tEnv &env, size_t n, size_t m, size_t k, size_t N, size_t M, size_t K, bool split, uint32_t iterations,
                   bool onlyMKL)
{
    ProcessorCube P(env.Size(), N, M, K);
    using tAlgorithm = MatMatAlgorithm<tEnv>;
    arma::mat A(n, m), B(k, m), C(n, k);

    if (!onlyMKL)
    {
        using tSubEnv = SyncLib::Environments::NoBSP;

        if (split)
        {
            env.Run(tAlgorithm::template MatMatTest<typename tAlgorithm::MatMatBenchSplit>, P, A, B, iterations);
        }
        else
        {
            env.Run(tAlgorithm::template MatMatTest<typename tAlgorithm::MatMatBench>, P, A, B, iterations);
        }
    }

    if (onlyMKL && env.Rank() == 0)
    {
        SyncLib::Util::LinearCongruentialRandom random(42);

        RandomizeMatrix(A, random);
        RandomizeMatrix(B, random);

        SyncLib::Util::Timer<>timer;

        for (size_t it = 0; it < iterations; ++it)
        {
            timer.Tic();
            C = A * B.t();
            fmt::print("Single-Threaded multiplication took {}s\n", timer.Toc());
        }

        fmt::print("Sum C sequential: {:9.4f}\n", arma::as_scalar(arma::sum(arma::sum(arma::abs(C), 0), 1)));
        fmt::print("Number of zeros (A, B, C): ({}, {}, {})\n",
                   A.n_elem - arma::nonzeros(A).eval().n_elem,
                   B.n_elem - arma::nonzeros(B).eval().n_elem,
                   C.n_elem - arma::nonzeros(C).eval().n_elem);
    }
}

int main(int argc, char **argv)
{
    SyncLib::Environments::DistributedBSP mpiEnv(argc, argv);

    /*if (mpiEnv.Rank() == 0)
    {
    system("pause");
    }

    mpiEnv.Barrier();*/

    Args args("Matrix-Matrix multiplication: C=AB");
    args.AddOptions(
    {
        {"n", "The number of rows in A, C", Option::U32(), "4096", "4096"},
        {"k", "The number of columns in A, rows in B", Option::U32(), "4096", "4096"},
        {"m", "The number of columns in B, C", Option::U32(), "4096", "4096"},
        {"N", "The divisor of n", Option::U32(), "2", "2",},
        {"K", "The divisor of k", Option::U32(), "2", "2",},
        {"M", "The divisor of m", Option::U32(), "2", "2",},
        {{"o", "only-mkl"}, "Run only with MKL", Option::Boolean(), "true", "false",},
        {{"s", "split"}, "Whether to split the environment into subset-environments", Option::Boolean(), "true", "true"},
        {"i", "The number of repetitions", Option::U32(), "1", "1"}
    });

    args.Parse(argc, argv);

    uint32_t n = args.GetOption("n");
    uint32_t m = args.GetOption("m");
    uint32_t k = args.GetOption("k");

    uint32_t N = args.GetOption("N");
    uint32_t M = args.GetOption("M");
    uint32_t K = args.GetOption("K");

    if (mpiEnv.Size() > 1)
    {
        RunMatMatTest(mpiEnv, n, m, k, N, M, K, args.GetOption("split"), args.GetOption("i"), args.GetOption("only-mkl"));
    }
    else
    {
        SyncLib::Environments::SharedMemoryBSP env(N * M * K);
        RunMatMatTest(env, n, m, k, N, M, K, args.GetOption("split"), args.GetOption("i"), args.GetOption("only-mkl"));
    }

    return EXIT_SUCCESS;
}
