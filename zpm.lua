
-- [[
-- @cond ___LICENSE___
--
-- Copyright (c) 2016-2018 Zefiros Software.
--
-- Permission is hereby granted, free of charge, to any person obtaining a copy
-- of this software and associated documentation files (the "Software"), to deal
-- in the Software without restriction, including without limitation the rights
-- to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
-- copies of the Software, and to permit persons to whom the Software is
-- furnished to do so, subject to the following conditions:
--
-- The above copyright notice and this permission notice shall be included in
-- all copies or substantial portions of the Software.
--
-- THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
-- IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
-- FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
-- AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
-- LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
-- OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
-- THE SOFTWARE.
--
-- @endcond
-- ]]

workspace "SyncLib"
    
    zefiros.setDefaults( "sync")

    cppdialect "C++17"

    defines "SYNCLIB_ENABLE_MPI"

    filter "system:windows"
        defines {
            "_SCL_SECURE_NO_WARNINGS",
            "NOMINMAX"
        }

    filter {}

    project "sync"
        kind "StaticLib"
        cppdialect "C++17"

        mpi "On"

        zpm.uses {
            "Zefiros-Software/Armadillo",
            "Zefiros-Software/Fmt",
            "Zefiros-Software/Args",
            "Zefiros-Software/Json"
        }

    project "sync-test"
        zpm.uses "Zefiros-Software/Armadillo"

    project "mpi-pingpong"
        location("mpi-pingpong")
        kind "ConsoleApp"

        mpi "On"

        links "sync"

        zpm.uses {
            "Zefiros-Software/Armadillo",
            "Zefiros-Software/Fmt",
            "Zefiros-Software/Args",
            "Zefiros-Software/Json"
        }

        includedirs {
            "mpi-pingpong",
            "sync/include"
        }

        files {
            "mpi-pingpong/**/*.cpp",
            "mpi-pingpong/*.cpp",
            "mpi-pingpong/**/*.h",
            "mpi-pingpong/*.h"
        }

    project "edupack-bench"
        location "edupack"
        kind "ConsoleApp"

        links "sync"

        mpi "On"

        debugargs {"--timings", "../../slurm-4045658.out"}

        zpm.uses {
            "Zefiros-Software/Armadillo",
            "Zefiros-Software/PlotLib",
            "Zefiros-Software/Json",
            "Zefiros-Software/Args",
            "Zefiros-Software/Fmt"
        }

        includedirs {
            "edupack/bench",
            "sync/include"
        }

        files {
            "edupack/bench/**/*.cpp",
            "edupack/bench/*.cpp",
            "edupack/bench/**/*.h",
            "edupack/bench/*.h"
        }

        filter "system:not windows"
            links "pthread"
        
        filter {}
