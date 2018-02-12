
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
local gcc = premake.tools.gcc
local api = premake.api

api.register {
    name = "gccwrapper",
    scope = "config",
    kind = "string",
    allowed = {
        "mpi"
    }
}

gcc.wrappers = {
    mpi = {
        cc = "mpicc",
        cxx = "mpicxx"
    }
}

premake.override(gcc, "gettoolname", function(base, cfg, tool)
    local wrapper = cfg.gccwrapper
    if wrapper and gcc.wrappers[wrapper] and gcc.wrappers[wrapper][tool] then
        return gcc.wrappers[wrapper][tool]
    else
        return base(cfg, tool)
    end
end)

local function linkMPI()
    if os.istarget("windows") then
        local mpi = os.getenv("I_MPI_ROOT")

        if mpi == nil then
            mpi = os.findlib("impi")

            if mpi ~= nil then
                mpi = mpi .. "../.."
            end
        end

        if mpi ~= nil then
            libdirs { mpi .. "/intel64/lib" }
            links "impi"
            includedirs { mpi .. "/intel64/include"}
        end
    else
        gccwrapper "mpi"
    end
end

workspace "SyncLib"
    
    zefiros.setDefaults( "sync", {
        mayLink = false
    } )
    
    cppdialect "C++17"

    filter "system:windows"
        defines {
            "_SCL_SECURE_NO_WARNINGS",
            "NOMINMAX"
        }

    filter {}

    project "sync"
        kind "StaticLib"
        cppdialect "C++17"

    project "mpi-pingpong"
        location("mpi-pingpong")
        kind "ConsoleApp"

        linkMPI()

        zpm.uses {
            "Zefiros-Software/Fmt"
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
        location("edupack")
        kind "ConsoleApp"

        links "sync"

        filter "system:not windows"
            links "pthread"

        zpm.uses {
            "Zefiros-Software/Armadillo",
            "Zefiros-Software/PlotLib",
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
