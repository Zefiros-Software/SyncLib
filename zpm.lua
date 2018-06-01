
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

local function useSyncLibraries()
    zpm.uses {
        "Zefiros-Software/Armadillo",
        "Zefiros-Software/Args",
        "Zefiros-Software/Json",
        "Zefiros-Software/Fmt",
        "Zefiros-Software/Fs",
        "Zefiros-Software/hwloc"
    }
end

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
        mpimt "On"
        vectorextensions "avx2"

        useSyncLibraries()

    project "sync-test"
        useSyncLibraries()

        mpi "On"
        mpimt "On"
        vectorextensions "avx2"

    project "mpi-pingpong"
        location("mpi-pingpong")
        kind "ConsoleApp"

        mpi "On"
        mpimt "On"
        vectorextensions "avx2"

        links "sync"

        useSyncLibraries()

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

    project "mat-mat"
        location "matmat"
        kind "ConsoleApp"

        links "sync"

        mpi "On"
        mpimt "On"
        vectorextensions "avx2"

        useSyncLibraries()

        includedirs {
            "matmat",
            "sync/include"
        }

        files {
            "matmat/**/*.cpp",
            "matmat/*.cpp",
            "matmat/**/*.h",
            "matmat/*.h"
        }

        filter "system:not windows"
            links "pthread"
        
        filter {}

    project "edupack-bench"
        location "edupack"
        kind "ConsoleApp"

        links "sync"

        -- debugargs { "--exit-paused" }

        mpi "On"
        mpimt "On"
        vectorextensions "avx2"

        useSyncLibraries()

        zpm.uses {
            "Zefiros-Software/PlotLib"
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
