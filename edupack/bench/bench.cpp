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
#include "sync/bench/partitioning.h"
#include "sync/sync.h"
#include "sync/util/algorithm.h"

#include "bench.h"

#include "args/args.h"
#include "preproc/preproc.h"

#include "nlohmann/json.hpp"

#include <armadillo>

#include <numeric>

auto LeastSquares(const size_t h0, const size_t h1, const size_t multiplier, const std::vector<double> &t)
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

    const arma::vec subT = arma::vec(t).rows(h0, h1);
    const double sumt = arma::sum(subT);
    const double sumth = arma::sum(subT % SyncLib::Util::ArmaRange(h0, h1)) * multiplier;

    const size_t nh = h1 - h0 + 1;
    const double sumh = multiplier * (h1 * (h1 + 1.0) - (h0 - 1.0) * h0) / 2;
    const double sumhh = multiplier * multiplier * (h1 * (h1 + 1) * (2.0 * h1 + 1) - (h0 - 1) * h0 * (2.0 * h0 - 1)) / 6;

    /* Solve    nh*l +  sumh*g =  sumt
    sumh*l + sumhh*g =  sumth */

    const double a = nh / sumh; // nh<= sumh

    /* subtract a times second eqn from first eqn to obtain g */
    double g = (sumt - a * sumth) / (sumh - a * sumhh);

    /* use second eqn to obtain l */
    double l = (sumth - sumhh * g) / sumh;

    return std::make_tuple(g, l);
}

template <typename tTimer>
NOINLINE double MeasureR(tTimer &timer, size_t n, arma::vec &y, double alpha, const arma::vec &x, arma::vec &z, double beta,
                         size_t repetitions)
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
    static void ReportR(const size_t n, const double minR, const double maxR, const double avR, const double fool)
    {
        fmt::print("n= {:>5} min= {:>8.3f} max= {:>8.3f} av= {:>8.3f} Mflop/s "
                   "fool={:>7.1f}\n",
                   n, minR, maxR, avR, fool);
    }

    static void ReportR0()
    {
        fmt::print("minimum time is 0\n");
    }

    static void ReportRelationH(const size_t h, const double time, const double flops, const bool printH)
    {
        if (printH)
        {
            fmt::print("Time of {:>5}-relation = {:>7.2f} microsec = {:>8.0f} flops\n", h, time, flops);
        }
    }

    template <typename tT>
    static void ReportSize()
    {
        fmt::print("size of {} = {} bytes\n", typeid(tT).name(), sizeof(tT));
    }

    template <typename tH0, typename tH1>
    static void ReportGL(tH0 h0, tH1 h1, double g, double l)
    {
        fmt::print("Range h={:>4} to {:>5}: g= {:>7.1f}, l= {:>7.1f}\n", h0, h1, g, l);
    }

    static void ReportBottomLine(const size_t p, const double rDivMega, const double g, const double l)
    {
        fmt::print("The bottom line for this BSP computer is:\n");
        fmt::print("p= {}, r= {:.3f} Mflop/s, g= {:.1f}, l= {:.1f}\n", p, rDivMega, g, l);
    }

    static size_t RequestP(const size_t maxP, const bool restrictP = true)
    {
        fmt::print("How many processors do you want to use?\n");
        const size_t p = 4;
        // std::cin>>p;

        if (restrictP && p > maxP)
        {
            fmt::print("Sorry, your requested {}, but only {} processors available.\n", p, maxP);
            exit(EXIT_FAILURE);
        }

        return p;
    }
};

template <typename tEnv>
void BspBench(tEnv &env, const size_t maxH, const size_t maxN, const size_t repetitions, const size_t batchSize,
              const std::string &output, const bool printH)
{

    SyncLib::Util::Timer<> timer;
    /**** Determine p ****/
    const size_t p = env.Size();
    const size_t s = env.Rank();

    typename tEnv::template SendQueue<double> timeQueue(env);
    typename tEnv::template SharedArray<double> dest(env, 2 * maxH + p);
    std::vector<double> hTime;

    /**** Determine r ****/
    double r = 0.0;

    arma::vec x(maxN), y(maxN), z(maxN);

    for (size_t n = 1; n <= maxN; n *= 2)
    {
        /* Initialize scalars and vectors */
        const double alpha = 1.0 / 3.0;
        const double beta = 4.0 / 9.0;

        for (size_t i = 0; i < n; ++i)
        {
            const auto xyz = static_cast<double>(i);
            z[i] = xyz;
            y[i] = xyz;
            x[i] = xyz;
        }

        double time = MeasureR(timer, n, y, alpha, x, z, beta, repetitions);

        timeQueue.Send(0, time);
        env.Sync();

        /* Processor 0 determines minimum, maximum, average computing rate */
        if (s == 0)
        {
            std::vector<double> timeCopy = timeQueue.ToVector();

            auto [mintime, maxtime] = SyncLib::Util::MinMax(timeCopy.begin(), timeCopy.end());

            if (mintime > 0.0)
            {
                /* Compute r = average computing rate in flop/s */
                const size_t nflops = 4 * repetitions * n;
                mintime = nflops / (mintime * std::mega::num);
                maxtime = nflops / (maxtime * std::mega::num);

                r = static_cast<double>(nflops) / arma::mean(arma::vec(timeCopy));

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

    typename tEnv::template SendQueue<size_t, double> queue(env);

    for (size_t h = 0; h <= maxH; ++h)
    {
        /* Initialize communication pattern */
        for (size_t i = 0; i < h; ++i)
        {
            src[i] = static_cast<double>(i);

            if (p == 1)
            {
                destproc[i] = 0;
                destindex[i] = i;
            }
            else
            {
                /* destination processor is one of the p-1 others */
                destproc[i] = (s + 1 + (i % (p - 1))) % p;
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
                for (size_t l = 0; l < batchSize; ++l)
                {
                    // dest.PutValue(destproc[i], src[i], destindex[i]);
                    queue.Send(destproc[i], destindex[i], src[i]);
                    // env.Print("Send: {}, {}, {}\n", destproc[i], destindex[i], src[i]);
                }
            }

            env.Sync();

            for (auto [index, val] : queue)
            {
                dest.Value()[index] = val;
                // env.Print("Recv: {}, {}, {}\n", s, index, val);
            }
        }

        const double time = timer.Toc();
        hTime.push_back(time * std::mega::num / repetitions);

        /* Compute time of one h-relation */
        if (s == 0)
        {
            // time in flop units
            t[h] = (time * r) / repetitions;
            BenchReporter::ReportRelationH(h, (time * std::mega::num) / repetitions, t[h], printH || h == maxH);
        }
    }

    if (s == 0)
    {
        const auto [g0, l0] = LeastSquares(0, p, batchSize, t);
        const auto [g, l] = LeastSquares(p, maxH, batchSize, t);

        const double flops = maxH * batchSize * g + l;
        BenchReporter::ReportRelationH(maxH, flops * std::mega::num / r, flops, true);
        BenchReporter::ReportSize<double>();
        BenchReporter::ReportGL(0, "p", g0, l0);
        BenchReporter::ReportGL("p", maxH, g, l);
        BenchReporter::ReportBottomLine(p, r / std::mega::num, g, l);

        if (!output.empty())
        {
            std::ofstream benchResults(output);
            nlohmann::json dump;
            dump["timings"] = arma::vec(t);
            dump["r"] = r;
            dump["g"] = g;
            dump["l"] = l;
            benchResults << dump;
            benchResults.flush();
        }
    }
}

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

template <typename tEnv, typename... tArgs>
void RunBenchmark(Args &parser, bool pause, tArgs &&... args)
{
    Pause(pause && parser.GetOption("start-paused"));
    uint32_t maxH = parser.GetOption("h");
    uint32_t maxN = parser.GetOption("n");
    uint32_t niters = parser.GetOption("i");
    uint32_t batchSize = parser.GetOption("b");
    auto outputOption = parser.GetOption("o");
    std::string output;

    if (outputOption.Count() > 0)
    {
        output = outputOption.Get<std::string>();
    }

    auto printOptions = parser.GetOption("print");
    bool printH = false;

    if (printOptions.Count() > 0)
    {
        std::vector<std::string> prints = printOptions.Get<std::vector<std::string>>();

        for (auto &o : prints)
        {
            if (o == "h" || o == "h-relation")
            {
                printH = true;
            }
        }
    }

    tEnv env(std::forward<tArgs>(args)...);
    env.Run(BspBench<tEnv>, maxH, maxN, niters, batchSize, output, printH);
    Pause(pause && parser.GetOption("exit-paused"));
}

#if defined(_DEBUG)
constexpr bool IsDebugBuild = true;
#elif defined(NDEBUG)
constexpr bool IsDebugBuild = false;
#else
constexpr bool IsDebugBuild = true;
#endif

template <typename tEnv>
double InnerProductPut(tEnv &env, std::vector<double> &x, std::vector<double> &y)
{
    size_t p = env.Size();
    size_t s = env.Rank();

    // using shared array with put for partial inner products
    typename tEnv::template SharedArray<double> partialInnerProducts(env, p);

    double alpha = 0.0;

    for (size_t i = 0, iEnd = x.size(); i < iEnd; ++i)
    {
        alpha += x[i] * y[i];
    }

    for (size_t t = 0; t < p; ++t)
    {
        partialInnerProducts.PutValue(t, alpha, s);
    }

    env.Sync();

    // Sum the partial inner products
    return std::accumulate(partialInnerProducts.begin(), partialInnerProducts.end(), 0.0);
}

template <typename tEnv>
double InnerProduct(tEnv &env, std::vector<double> &x, std::vector<double> &y)
{
    const size_t p = env.Size();

    // using shared array with put for partial inner products
    typename tEnv::template SendQueue<double> partialInnerProducts(env);

    double alpha = 0.0;

    for (size_t i = 0, iEnd = x.size(); i < iEnd; ++i)
    {
        alpha += x[i] * y[i];
    }

    for (size_t t = 0; t < p; ++t)
    {
        partialInnerProducts.Send(t, alpha);
    }

    env.Sync();

    // Sum the partial inner products
    return std::accumulate(partialInnerProducts.begin(), partialInnerProducts.end(), 0.0);
}

template <typename tEnv>
void BaselProblem(tEnv &env, size_t n)
{
    const size_t p = env.Size();
    const size_t s = env.Rank();

    const size_t nl = (n + p - s - 1) / p;
    std::vector<double> x(nl);

    for (size_t i = 0; i < nl; ++i)
    {
        const size_t iGlob = i * p + s;
        x[i] = 1.0 / (iGlob + 1);
    }

    SyncLib::Util::Timer<std::chrono::microseconds> timer;

    // Let everyone start at the same time
    env.Barrier();
    timer.Tic();

    double alpha = 0.0;

    for (size_t j = 0; j < 1000; ++j)
    {
        alpha = InnerProduct(env, x, x);
    }

    // Measure the time when everyone is done
    env.Barrier();
    const double elapsed = timer.Toc() / 1000;

    printf("Processor %zd: solution to the Basel problem "
           "with reciprocals up to 1/%zd^2 is %.6f\n",
           s, n, alpha);

    if (s == 0)
    {
        printf("This took only %.6lf microseconds.\n", elapsed);
    }
}

// int main(int argc, char **argv)
// {
//     using tEnv = SyncLib::Environments::SharedMemoryBSP;
//     tEnv env(4);
//
//     // n can be requested from the user using your favorite method.
//     // You can also request if for s=0 inside the function.
//     size_t n = 1000000;
//
//     env.Run(BaselProblem<tEnv>, n);
//
//     return 0;
// }

int main(int argc, char **argv)
{
    Args args("bench", "Edupack benchmark for SyncLib");

    std::string iterations(IsDebugBuild ? "100" : "1000");

    args.AddOptions(
    {
        {
            "start-paused",
            "Whether we want to pause before the benchmark starts",
            Option::Boolean(),
            "true",
            "false",
        },
        {
            "exit-paused",
            "Whether we want to pause before the benchmark exits",
            Option::Boolean(),
            "true",
            "false",
        },
        {
            { "i", "iterations" },
            "The number of iterations for each h-relation",
            Option::U32(),
            iterations,
            iterations,
        },
        {
            { "h", "maximum-h" },
            "The maximum h-relation",
            Option::U32(),
            "256",
            "256",
        },
        {
            "print",
            "Fields to print",
            Option::StringList(),
            {},
            {},
        },
        {
            { "n", "maximum-n" },
            "The maximum number of DAXPYs for measuring r",
            Option::U32(),
            "1024",
            "1024",
        },
        {
            { "o", "output-file" },
            "The file to store the output in",
            Option::String(),
        },
        {
            "parts",
            "The number of parts to split the environment in",
            Option::U32(),
            "1",
            "1",
        },
    });

    SyncLib::MPI::Comm comm(argc, argv);

    {
        std::string batchSize = comm.Size() > 1 ? "8" : "1";

        args.AddOptions(
        {
            {
                { "b", "batch-size" },
                "Size of each communication packet",
                Option::U32(),
                batchSize,
                batchSize,
            },
        });
    }

    if (comm.Size() > 1)
    {
        char parsed = false;

        if (comm.Rank() == 0)
        {
            args.Parse(argc, argv);
            parsed = true;
        }

        comm.Broadcast(parsed, 0);

        if (parsed && comm.Rank() != 0)
        {
            args.Parse(argc, argv);
        }

        // SyncLib::Internal::PinThread(comm.Rank());

        const uint32_t parts = args.GetOption("parts");
        //         Pause(comm.Rank() == 0);
        //         comm.Barrier();
        using tEnv = SyncLib::Environments::DistributedBSP;
        tEnv env(argc, argv);
        env.Split(0, env.Size() - env.Rank());

        if (parts > 1)
        {
            const size_t partSize = (comm.Size() + parts - 1) / parts;
            auto comm2 = comm.Split(comm.Rank() / partSize, comm.Rank() % partSize);
            RunBenchmark<tEnv>(args, comm2.Rank() == 0, comm2);
        }
        else
        {
            RunBenchmark<tEnv>(args, comm.Rank() == 0, comm);
        }
    }
    else
    {
        using tEnv = SyncLib::Environments::SharedMemoryBSP;
        {
            std::string p = std::to_string(tEnv::MaxSize());

            args.AddOptions(
            {
                {
                    { "p", "processors" },
                    "The number of processors",
                    Option::U32(),
                    p,
                    p,
                },
            });
        }

        args.Parse(argc, argv);
        uint32_t p = args.GetOption("processors");

        RunBenchmark<tEnv>(args, true, p);
    }

    // fflush(stdout);
    comm.Barrier();

    return 0;
}
