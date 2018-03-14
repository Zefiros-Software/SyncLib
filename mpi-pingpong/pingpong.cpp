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

#include "sync/util/time.h"

#include "args/args.h"

#include "nlohmann/json.hpp"
#include "fmt/format.h"

#include <armadillo>

#include <experimental/filesystem>
#include <iomanip>
#include <ctime>

namespace fs = std::experimental::filesystem;

int main(int argc, char *argv[])
{
    SyncLib::MPI::Comm comm(argc, argv);

    uint32_t q = 0;
    std::string outputFile = "run";

    if (comm.Rank() == 0)
    {
        Args args("bench", "MPI pingpong");
        std::string timestamp = SyncLib::Util::GetTimeString();
        args.AddOptions(
        {
            { { "q", "parts" }, "Number of parts for partitioning.", Option::U32(), "0", "0" },
            { { "o", "output"}, "Output file for timings.", Option::String(), timestamp, timestamp}
        });
        args.Parse(argc, argv);

        q = args.GetOption("parts");
        outputFile = args.GetOption("output");
    }

    comm.Broadcast(q, 0);

    SyncLib::Bench::MPIPingPongBenchmark bench(comm);
    bench.PingPong();

    {
        std::ofstream out;

        if (comm.Rank() == 0)
        {
            fs::path results("./results");

            if (!fs::exists(results))
            {
                fs::create_directory(results);
            }

            out.open(results / fmt::format("{}.json", outputFile));
        }

        bench.Serialise(out);
    }

    auto [G, L] = bench.ComputePairwise();

    //     using json = nlohmann::json;
    //     Args args("bench", "Edupack benchmark");
    //     args.AddOptions({ { "timings", "File with timings", Option::String() } });
    //     args.Parse(argc, argv);
    //
    //     nlohmann::json data;
    //     {
    //         namespace fs = std::experimental::filesystem;
    //         fs::path timings(fs::absolute(args.GetOption("timings").Get<std::string>()));
    //         std::ifstream(timings) >> data;
    //     }
    //     arma::mat distances, L;
    //     std::tie(distances, L) = SyncLib::Bench::MPIPingPongBenchmark::ComputePairwise(data);
    auto parts = SyncLib::Partitioning::MakeImprovedClusterInitialisedPartitioning(G, q);

    if (comm.Rank() == 0)
    {
        fmt::print("Original max: {}\n", G.max());

        for (auto &part : parts)
        {
            arma::uvec slicer(part);
            fmt::print("Max in part of size {}: {}\n", slicer.size(), G(slicer, slicer).max());
        }
    }

    //system("pause");
    comm.Barrier();
    return EXIT_SUCCESS;
}