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
#include "sync/bindings/mpi.h"

#include "sync/util/algorithm.h"
#include "sync/util/timer.h"

#include "fmt/format.h"

#include <iostream>

int main(int argc, char *argv[])
{
    SyncLib::MPI::Comm comm(argc, argv);

    SyncLib::Util::Timer<> timer;

    size_t s = comm.Rank();
    size_t p = comm.Size();
    size_t p2 = SyncLib::Util::NextPowerOfTwo(p);
    size_t receive;

    std::vector<double> timings(p, 0.0);
    constexpr size_t repetitions = 50000;
    constexpr size_t rotations = 20;

    comm.Barrier();


    for (size_t i = 0; i < rotations; ++i)
    {
        for (size_t mask = 1; mask < p2; ++mask)
        {
            size_t t = s ^ mask;

            if (t < p)
            {
                if (s < t)
                {
                    fmt::print("Performing ping-pong measurement between {} and {}\n", s, t);
                    fflush(stdout);
                }

                timer.Tic();

                for (size_t j = 0; j < repetitions; ++j)
                {
                    comm.SendReceive(t, &s, 1, &receive, 1);
                }

                double elapsed = timer.Toc() * 1000;

                timings[t] += elapsed;
            }

            comm.Barrier();
        }
    }

    for (size_t s2 = 0; s2 < p; ++s2)
    {
        if (s != s2)
        {
            comm.Barrier();
            continue;
        }

        for (size_t mask = 1; mask < p2; ++mask)
        {
            size_t t = s ^ mask;

            if (t < p)
            {
                fmt::print("Average communication time between {} and {}: {:>8.4f}us\n", s, t, timings[t] * 1000 / repetitions / rotations);
            }
        }

        fmt::print("\n");

        comm.Barrier();
    }
}