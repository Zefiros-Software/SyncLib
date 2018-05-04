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

#include "nlohmann/json.hpp"

#include <armadillo>
#include <tuple>

using tEnv = SyncLib::Environments::SharedMemoryBSP;

// ReSharper disable CppInconsistentNaming
namespace nlohmann
{
    template<>
    struct adl_serializer<arma::vec>
    {
        static ::arma::vec from_json(const json &j)
        {
            const size_t size = j.size();
            ::arma::vec vect = ::arma::zeros(size);

            std::copy(j.begin(), j.end(), vect.begin());

            return vect;
        }

        static void to_json(json &j, const ::arma::vec &vect)
        {
            for (auto &entry : vect)
            {
                j.push_back(entry);
            }
        }
    };
    template<>
    struct adl_serializer<arma::rowvec>
    {
        static ::arma::rowvec from_json(const json &j)
        {
            const size_t size = j.size();
            ::arma::rowvec vect = ::arma::zeros(size);

            std::copy(j.begin(), j.end(), vect.begin());

            return vect;
        }

        static void to_json(json &j, const ::arma::rowvec &vect)
        {
            for (auto &entry : vect)
            {
                j.push_back(entry);
            }
        }
    };

    template<>
    struct adl_serializer<arma::mat>
    {
        static ::arma::mat from_json(const json &j)
        {
            const size_t rows = j.size();
            ::arma::mat matrix = ::arma::zeros(j.size(), j[0].size());

            for (size_t row = 0; row < rows; ++row)
            {
                ::arma::rowvec matRow = matrix.row(row);
                std::copy(j[row].begin(), j[row].end(), matRow.begin());
            }

            return matrix;
        }

        static void to_json(json &j, const ::arma::mat &matrix)
        {
            for (size_t row = 0, rows = matrix.n_rows; row < rows; ++row)
            {
                j.push_back(::arma::rowvec(matrix.row(row)));
            }
        }
    };
} // namespace nlohmann
// ReSharper restore CppInconsistentNaming

template<typename tMat, typename tRandom>
void RandomizeMatrix(tMat &mat, tRandom &random)
{
    for (auto &entry : mat)
    {
        entry = random.NextDouble();
    }
}

template<typename tEnv>
struct MatMatAlgoBase
{
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
    };

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
        size_t nl02;
        size_t nl21;
        size_t nl01;

        static DimInfoContiguous Create(const size_t n, const size_t k, const size_t m, const ProcessorCube &P, const size_t s0,
                                        const size_t s1, const size_t s2)
        {
            const size_t nl0 = NumLocalBalanced(n, P.q0, s0);
            const size_t nl1 = NumLocalBalanced(m, P.q1, s1);
            const size_t nl2 = NumLocalBalanced(k, P.q2, s2);

            const size_t nl02 = NumLocal(nl0 * nl2, P.q1);
            const size_t nl21 = NumLocal(nl2 * nl1, P.q0);
            const size_t nl01 = NumLocal(nl0 * nl1, P.q2);

            return { n, k, m, nl0, nl1, nl2, nl02, nl21, nl01 };
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

template<typename tEnv>
struct MatMatAlgorithm;

template<>
struct MatMatAlgorithm<SyncLib::Environments::SharedMemoryBSP>
    : public MatMatAlgoBase<SyncLib::Environments::SharedMemoryBSP>
{
    using tEnv = SyncLib::Environments::SharedMemoryBSP;
    using tAlgo = MatMatAlgoBase<SyncLib::Environments::SharedMemoryBSP>;

    using DimInfoContiguous = tAlgo::DimInfoContiguous;
    using ProcessorCube = tAlgo::ProcessorCube;

    template<typename tMat>
    static void MatMatCore(tEnv &env, const tMat &A_su, const tMat &B_ut, arma::mat &C_st,
                           const DimInfoContiguous &dimInfo, const ProcessorCube &P,
                           tEnv::SendQueue<arma::uword, arma::vec> &sqC)
    {
        auto [s0, s1, s2] = P(env.Rank());

        C_st = A_su * B_ut.t();

        tAlgo::ReduceCContiuous(C_st, sqC, s2, s0 * P.q1 * P.q2 + s1 * P.q2, P.q2, dimInfo, env);
    }

    static void MatMat(tEnv &env, arma::mat &A_su, arma::mat &B_ut, arma::mat &C_st,
                       const DimInfoContiguous &dimInfo, const ProcessorCube &P,
                       tEnv::SendQueue<arma::uword, arma::vec> &sqA, tEnv::SendQueue<arma::uword, arma::vec> &sqB,
                       tEnv::SendQueue<arma::uword, arma::vec> &sqC)
    {
        auto [s0, s1, s2] = P(env.Rank());

        const size_t q0 = P.q0, q1 = P.q1, q2 = P.q2;

        const size_t nl0 = dimInfo.nl0, nl1 = dimInfo.nl1, nl2 = dimInfo.nl2,
                     nl02 = dimInfo.nl02, nl21 = dimInfo.nl21;

        tAlgo::FanOutMatrixContiguous(A_su, sqA, q1, s1, s0 * q1 * q2 + s2, q2, nl02, nl0 * nl2);
        tAlgo::FanOutMatrixContiguous(B_ut, sqB, q0, s0, s2 + s1 * q2, q1 * q2, nl21, nl2 * nl1);

        env.Sync();

        tAlgo::FanInMatrixContiguous(A_su, sqA);
        tAlgo::FanInMatrixContiguous(B_ut, sqB);

        MatMatCore(env, A_su, B_ut, C_st, dimInfo, P, sqC);
    }

    /*static void MatMatInitWhole(tEnv &env, size_t n, size_t k, size_t m, tAlgo::ProcessorCube &P, arma::mat &A, arma::mat &B)
    {
        const size_t rank = env.Rank();
        auto [s0, s1, s2] = P(rank);

        const DimInfoContiguous dimInfo = tAlgo::DimInfoContiguous::Create(n, k, m, P, s0, s1, s2);

        arma::mat A_su(dimInfo.nl0, dimInfo.nl2, arma::fill::zeros);
        arma::mat B_ut(dimInfo.nl1, dimInfo.nl2, arma::fill::zeros);
        arma::mat C_st(dimInfo.nl0, dimInfo.nl1, arma::fill::zeros);

        SyncLib::Util::LinearCongruentialRandom random(env.Rank());

        RandomizeMatrix(A_su, random);
        RandomizeMatrix(B_ut, random);

        SyncLib::Util::Timer<>timer;
        tEnv::SendQueue<arma::uword, arma::vec>sqA(env), sqB(env), sqC(env);
        env.Barrier();
        timer.Tic();

        for (size_t it = 0; it < 4 * 1; ++it)
        {
            MatMat(env, A_su, B_ut, C_st, dimInfo, P, sqA, sqB, sqC);
        }

        fmt::print("Multiplication on s={} took {}s\n", rank, timer.TocTic() / 4 * 1);

        tEnv::SendQueue<double>sumC(env);
        {
            const arma::uword start = s2 * dimInfo.nl01;
            const arma::uword size = std::min<arma::uword>(start + dimInfo.nl01, dimInfo.nl0 * dimInfo.nl1) - start;
            const arma::vec subC(&C_st(start), size, false, true);
            sumC.Broadcast(arma::as_scalar(arma::sum(arma::abs(subC))));
        }
        env.Sync();
        fmt::print("Sum C on s={:<5}: {:9.4f}\n", rank, std::accumulate(sumC.begin(), sumC.end(), 0.0));

        GetSubViewBalanced(A, s0, P.q0, n, dimInfo.nl0, s2, P.q2, k, dimInfo.nl2) = A_su;
        GetSubViewBalanced(B, s1, P.q1, m, dimInfo.nl1, s2, P.q2, k, dimInfo.nl2) = B_ut;
    }*/

    static void MatMatSplit(tEnv &env, tEnv &env02, tEnv &env21, tEnv &env01,
                            arma::mat &A_su, arma::mat &B_ut, arma::mat &C_st,
                            const DimInfoContiguous &dimInfo, const ProcessorCube &P,
                            tEnv::SendQueue<arma::uword, arma::vec> &sqA, tEnv::SendQueue<arma::uword, arma::vec> &sqB,
                            tEnv::SendQueue<arma::uword, arma::vec> &sqC)
    {
        auto[s0, s1, s2] = P(env.Rank());

        tAlgo::FanOutMatrixContiguous(A_su, sqA, P.q1, s1, 0, 1, dimInfo.nl02, dimInfo.nl0 * dimInfo.nl2);
        env02.Sync();
        tAlgo::FanInMatrixContiguous(A_su, sqA);

        tAlgo::FanOutMatrixContiguous(B_ut, sqB, P.q0, s0, 0, 1, dimInfo.nl21, dimInfo.nl2 * dimInfo.nl1);
        env21.Sync();
        tAlgo::FanInMatrixContiguous(B_ut, sqB);

        MatMatCore(env01, A_su, B_ut, C_st, dimInfo, P, sqC);
    }

    struct MatMatBench
    {
        static void Run(size_t repetitions, SyncLib::Util::Timer<> &timer, tEnv &env, const tAlgo::ProcessorCube &P,
                        const tAlgo::DimInfoContiguous &dimInfo, arma::mat &A_su, arma::mat &B_ut, arma::mat &C_st)
        {
            tEnv::SendQueue<arma::uword, arma::vec>sqA(env), sqB(env), sqC(env);
            env.Barrier();
            timer.Tic();

            for (size_t it = 0; it < repetitions; ++it)
            {
                MatMat(env, A_su, B_ut, C_st, dimInfo, P, sqA, sqB, sqC);
            }
        }
    };

    struct MatMatBenchSplit
    {
        static void Run(size_t repetitions, SyncLib::Util::Timer<> &timer, tEnv &env, const tAlgo::ProcessorCube &P,
                        const tAlgo::DimInfoContiguous &dimInfo, arma::mat &A_su, arma::mat &B_ut, arma::mat &C_st)
        {
            const size_t rank = env.Rank();
            const auto[s0, s1, s2] = P(rank);

            tEnv &env02 = env.Split(s0 * P.q2 + s2, s1);
            tEnv &env21 = env.Split(s2 * P.q1 + s1, s0);
            tEnv &env01 = env.Split(s0 * P.q1 + s1, s2);

            tEnv::SendQueue<arma::uword, arma::vec> sqA(env02), sqB(env21), sqC(env01);
            env.Barrier();
            timer.Tic();

            for (size_t it = 0; it < repetitions; ++it)
            {
                MatMatSplit(env, env02, env21, env01, A_su, B_ut, C_st, dimInfo, P, sqA, sqB, sqC);
            }
        }
    };

    template<typename tBench>
    static void MatMatTest(tEnv &env, tAlgo::ProcessorCube &P, arma::mat &A, arma::mat &B)
    {
        const size_t rank = env.Rank();
        const auto[s0, s1, s2] = P(rank);

        size_t n = A.n_rows, k = A.n_cols, m = B.n_rows;
        const DimInfoContiguous dimInfo = tAlgo::DimInfoContiguous::Create(n, k, m, P, s0, s1, s2);

        arma::mat A_su(dimInfo.nl0, dimInfo.nl2, arma::fill::zeros);
        arma::mat B_ut(dimInfo.nl1, dimInfo.nl2, arma::fill::zeros);
        arma::mat C_st(dimInfo.nl0, dimInfo.nl1, arma::fill::zeros);

        SyncLib::Util::LinearCongruentialRandom random(rank);
        RandomizeMatrix(A_su, random);
        RandomizeMatrix(B_ut, random);

        SyncLib::Util::Timer<> timer;

        size_t repetitions = 4;
        tBench::Run(repetitions, timer, env, P, dimInfo, A_su, B_ut, C_st);

        fmt::print("Multiplication on s={} took {}s\n", rank, timer.TocTic() / repetitions);

        tEnv::SendQueue<double>sumC(env);
        {
            const arma::uword start = s2 * dimInfo.nl01;
            const arma::uword size = std::min<arma::uword>(start + dimInfo.nl01, dimInfo.nl0 * dimInfo.nl1) - start;
            const arma::vec subC(&C_st(start), size, false, true);
            sumC.Broadcast(arma::as_scalar(arma::sum(arma::abs(subC))));
        }
        env.Sync();
        fmt::print("Sum C on s={:<5}: {:9.4f}\n", rank, std::accumulate(sumC.begin(), sumC.end(), 0.0));

        GetSubViewBalanced(A, s0, P.q0, n, dimInfo.nl0, s2, P.q2, k, dimInfo.nl2) = A_su;
        GetSubViewBalanced(B, s1, P.q1, m, dimInfo.nl1, s2, P.q2, k, dimInfo.nl2) = B_ut;
    }

    /*static void MatMatInitSplit(tEnv &env, size_t n, size_t k, size_t m, tAlgo::ProcessorCube &P, arma::mat &A,
                                arma::mat &B)
    {
        const size_t rank = env.Rank();
        auto [s0, s1, s2] = P(rank);

        const DimInfoContiguous dimInfo = tAlgo::DimInfoContiguous::Create(n, k, m, P, s0, s1, s2);

        arma::mat A_su(dimInfo.nl0, dimInfo.nl2, arma::fill::zeros);
        arma::mat B_ut(dimInfo.nl1, dimInfo.nl2, arma::fill::zeros);
        arma::mat C_st(dimInfo.nl0, dimInfo.nl1, arma::fill::zeros);

        SyncLib::Util::LinearCongruentialRandom random(env.Rank());

        RandomizeMatrix(A_su, random);
        RandomizeMatrix(B_ut, random);

        SyncLib::Util::Timer<>timer;

        tEnv &env02 = env.Split(s0 * P.q2 + s2, s1);
        tEnv &env21 = env.Split(s2 * P.q1 + s1, s0);
        tEnv &env01 = env.Split(s0 * P.q1 + s1, s2);

        tEnv::SendQueue<arma::uword, arma::vec> sqA(env02), sqB(env21), sqC(env01);
        env.Barrier();
        timer.Tic();

        for (size_t it = 0; it < 4 * 1; ++it)
        {
            MatMatSplit(env, env02, env21, env01, A_su, B_ut, C_st, dimInfo, P, sqA, sqB, sqC);
        }

        fmt::print("Multiplication on s={} took {}s\n", rank, timer.TocTic() / 4 * 1);

        tEnv::SendQueue<double>sumC(env);
        {
            const arma::uword start = s2 * dimInfo.nl01;
            const arma::uword size = std::min<arma::uword>(start + dimInfo.nl01, dimInfo.nl0 * dimInfo.nl1) - start;
            const arma::vec subC(&C_st(start), size, false, true);
            sumC.Broadcast(arma::as_scalar(arma::sum(arma::abs(subC))));
        }
        env.Sync();
        fmt::print("Sum C on s={:<5}: {:9.4f}\n", rank, std::accumulate(sumC.begin(), sumC.end(), 0.0));

        GetSubViewBalanced(A, s0, P.q0, n, dimInfo.nl0, s2, P.q2, k, dimInfo.nl2) = A_su;
        GetSubViewBalanced(B, s1, P.q1, m, dimInfo.nl1, s2, P.q2, k, dimInfo.nl2) = B_ut;
    }*/
};


int main(int /*argc*/, char ** /*argv*/)
{
    using tAlgorithm = MatMatAlgorithm<tEnv>;

    tEnv env(8);
    tAlgorithm::ProcessorCube P(env.Size(), 4, 1, 2);

    arma::mat A(6144 * 4 - 1, 4095), B(2047, 4095), C(6144 * 4 - 1, 2047);
    /*env.Run(tAlgorithm::MatMatTest<typename tAlgorithm::MatMatBench>, P, A, B);
    env.Run(tAlgorithm::MatMatTest<typename tAlgorithm::MatMatBenchSplit>, P, A, B);*/

    SyncLib::Util::LinearCongruentialRandom random(42);

    RandomizeMatrix(A, random);
    RandomizeMatrix(B, random);

    SyncLib::Util::Timer<>timer;
    timer.Tic();

    for (size_t it = 0; it < 4 * 1 * 1; ++it)
    {
        C = A * B.t();
    }

    fmt::print("Single-Threaded multiplication took {}s\n", timer.Toc() / 4 * 1 * 1);
    fmt::print("Sum C sequential: {:9.4f}\n", arma::as_scalar(arma::sum(arma::sum(arma::abs(C), 0), 1)));
    fmt::print("Number of zeros (A, B, C): ({}, {}, {})",
               A.n_elem - arma::nonzeros(A).eval().n_elem,
               B.n_elem - arma::nonzeros(B).eval().n_elem,
               C.n_elem - arma::nonzeros(C).eval().n_elem);

    //system("pause");
}
