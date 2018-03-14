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
#include "sync/bench/mpiPingpong.h"
#include "sync/util/algorithm.h"

#include "bench.h"

#include "args/args.h"

#include "nlohmann/json.hpp"

#include <armadillo>

#include <experimental/filesystem>
#include <algorithm>
#include <numeric>

#define NITERS 5000
#define MAXN 1024
#define MAXH 256
#define MEGA 1000000.0

auto LeastSquares(size_t h0, size_t h1, const std::vector<double> &t)
{
    /* This function computes the parameters g and l of the
    linear function T(h)= g*h+l that best fits
    the data points (h,t[h]) with h0 <= h <= h1. */

    /* Compute sums:
    sumt  =  sum of t[h] over h0 <= h <= h1
    sumth =         t[h]*h
    nh    =         1
    sumh  =         h
    sumhh =         h*h */

    arma::vec subT = arma::vec(t).rows(h0, h1 + 1);
    double sumt = arma::sum(subT);
    double sumth = arma::sum(subT % SyncLib::Util::ArmaRange(h0, h1 + 1));

    size_t nh = h1 - h0 + 1;
    size_t sumh = (h1 * (h1 + 1) - (h0 - 1) * h0) / 2;
    size_t sumhh = (h1 * (h1 + 1) * (2 * h1 + 1) - (h0 - 1) * h0 * (2 * h0 - 1)) / 6;

    /* Solve    nh*l +  sumh*g =  sumt
    sumh*l + sumhh*g =  sumth */

    double a = nh / (double)sumh;    // nh <= sumh

    /* subtract a times second eqn from first eqn to obtain g */
    double g = (sumt - a * sumth) / (sumh - a * sumhh);

    /* use second eqn to obtain l */
    double l = (sumth - sumhh * g) / sumh;

    return std::make_tuple(g, l);
}

template<typename tTimer>
SYNCLIB_NOINLINE double MeasureR(tTimer &timer, size_t n, std::vector<double> &y, double alpha, const std::vector<double> &x,
                                 std::vector<double> &z, double beta, size_t repetitions)
{
    /* Measure time of 2*repetitions DAXPY operations of length n */
    timer.Tic();

    constexpr size_t rep = 50;

    for (size_t iter = 0; iter < rep * repetitions; ++iter)
    {
        for (size_t i = 0; i < n; ++i)
        {
            y[i] += alpha * x[i];
        }

        for (size_t i = 0; i < n; ++i)
        {
            z[i] -= beta * x[i];
        }
    }

    return timer.Toc() / rep;
}

struct BenchReporter
{
    static void ReportR(size_t n, double minR, double maxR, double avR, double fool)
    {
        fmt::print("n= {:>5} min= {:>8.3f} max= {:>8.3f} av= {:>8.3f} Mflop/s fool={:>7.1f}\n", n, minR, maxR, avR, fool);
    }

    static void ReportR0()
    {
        fmt::print("minimum time is 0\n");
    }

    static void ReportRelationH(size_t h, double time, double flops)
    {
        fmt::print("Time of {:>5}-relation = {:>7.2f} microsec = {:>8.0f} flops\n", h, time, flops);
    }

    template<typename tT>
    static void ReportSize()
    {
        fmt::print("size of {} = {} bytes\n", typeid(tT).name(), sizeof(tT));
    }

    template<typename tH0, typename tH1>
    static void ReportGL(tH0 h0, tH1 h1, double g, double l)
    {
        fmt::print("Range h={:>4} to {:>5}: g= {:>7.1f}, l= {:>7.1f}\n", h0, h1, g, l);
    }

    static void ReportBottomLine(size_t p, double rDivMega, double g, double l)
    {
        fmt::print("The bottom line for this BSP computer is:\n");
        fmt::print("p= {}, r= {:.3f} Mflop/s, g= {:.1f}, l= {:.1f}\n", p, rDivMega, g, l);
    }

    static size_t RequestP(size_t maxP, bool restrictP = true)
    {
        fmt::print("How many processors do you want to use?\n");
        size_t p = 4;
        //std::cin >> p;

        if (restrictP && p > maxP)
        {
            fmt::print("Sorry, your requested {}, but only {} processors available.\n", p, maxP);
            exit(EXIT_FAILURE);
        }

        return p;
    }
};

template<typename tEnv>
void BspBench(tEnv &env, size_t maxH, size_t maxN, size_t repetitions)
{

    SyncLib::Util::Timer<> timer;
    /**** Determine p ****/
    size_t p = env.Size();
    size_t s = env.Rank();

    auto Time = env.template ShareArray<double>(p);
    auto dest = env.template ShareArray<double>(2 * maxH + p);
    std::vector<double> hTime;

    /**** Determine r ****/
    double r = 0.0;

    std::vector<double> x(maxN), y(maxN), z(maxN);

    for (size_t n = 1; n <= maxN; n *= 2)
    {
        /* Initialize scalars and vectors */
        double alpha = 1.0 / 3.0;
        double beta = 4.0 / 9.0;

        for (size_t i : SyncLib::Ranges::Range(n))
        {
            double xyz = static_cast<double>(i);
            z[i] = xyz;
            y[i] = xyz;
            x[i] = xyz;
        }

        double time = MeasureR(timer, n, y, alpha, x, z, beta, repetitions);

        Time.PutValue(0, time, s);
        env.Sync();

        /* Processor 0 determines minimum, maximum, average computing rate */
        if (s == 0)
        {
            auto [mintime, maxtime] = SyncLib::Util::MinMax(Time.begin(), Time.end());

            if (mintime > 0.0)
            {
                /* Compute r = average computing rate in flop/s */
                size_t nflops = 4 * repetitions * n;
                mintime = nflops / (mintime * std::mega::num);
                maxtime = nflops / (maxtime * std::mega::num);

                r = arma::mean(static_cast<double>(nflops) / arma::vec(Time.Value()));

                BenchReporter::ReportR(n, mintime, maxtime, r / std::mega::num, y[n - 1] + z[n - 1]);
            }
            else
            {
                BenchReporter::ReportR0();
            }
        }
    }

    /* r is taken as the value at length maxN */

    /**** Determine g and l ****/
    std::vector<size_t> destproc(maxH), destindex(maxH);
    std::vector<double> src(maxH), t(maxH + 1);

    for (size_t h = 0; h <= maxH; ++h)
    {
        /* Initialize communication pattern */
        for (size_t i : SyncLib::Ranges::Range(h))
        {
            src[i] = (double)i;

            if (p == 1)
            {
                destproc[i] = 0;
                destindex[i] = i;
            }
            else
            {
                /* destination processor is one of the p-1 others */
                destproc[i] = (s + 1 + i % (p - 1)) % p;
                /* destination index is in my own part of dest */
                destindex[i] = s + (i / (p - 1)) * p;
            }
        }

        /* Measure time of repetitions h-relations */
        env.Sync();
        timer.Tic();

        for (size_t iter = 0; iter < repetitions; iter++)
        {
            for (size_t i = 0; i < h; ++i)
            {
                dest.PutValue(destproc[i], src[i], destindex[i]);
            }

            env.Sync();
        }

        double time = timer.Toc();
        hTime.push_back(time * std::mega::num / repetitions);

        /* Compute time of one h-relation */
        if (s == 0)
        {
            // time in flop units
            t[h] = (time * r) / repetitions;
            BenchReporter::ReportRelationH(h, (time * std::mega::num) / repetitions, t[h]);
        }
    }

    if (s == 0)
    {
        const auto [g0, l0] = LeastSquares(0, p, t);
        const auto [g,   l] = LeastSquares(p, maxH, t);

        BenchReporter::ReportSize<double>();
        BenchReporter::ReportGL(0, "p", g0, l0);
        BenchReporter::ReportGL("p", maxH, g, l);
        BenchReporter::ReportBottomLine(p, r / std::mega::num, g, l);
    }
}



int main(int argc, char **argv)
{
    using tEnv = SyncLib::Environments::SharedMemoryBSP;
    tEnv env;
    size_t p;

    {
        Args args("bench", "Edupack benchmark for SyncLib");
        args.AddOptions(
        {
            {{"p", "processors"}, "The number of processors", Option::U32(), fmt::format("{}", env.MaxSize()), fmt::format("{}", env.MaxSize()) }
        });
        args.Parse(argc, argv);
        uint32_t aP = args.GetOption("processors");
        p = aP;
    }


    env.Run(p, BspBench<tEnv>, MAXH, MAXN, NITERS);

#ifdef _WIN32
    system("pause");
#endif

    exit(EXIT_SUCCESS);
}
