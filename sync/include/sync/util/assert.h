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
#pragma once
#ifndef __SYNCLIB_ASSERT_H__
#define __SYNCLIB_ASSERT_H__

#include "fmt/format.h"

#include "preproc/preproc.h"

#include <cassert>
#include <iostream>
#include <string>

namespace SyncLibInternal
{

#if defined(_DEBUG) || !defined(NDEBUG) || defined(SYNCLIB_STRICT_ASSERT)
    template <typename tT, typename... tArgs>
    FORCEINLINE void Assert(const tT &condition, tArgs &&... args)
    {
        if (!condition)
        {
            auto msg = fmt::format(std::forward<tArgs>(args)...);
            std::cout << msg << std::endl;

#if !defined(_DEBUG) || defined(NDEBUG)
            throw std::runtime_error(msg);
#endif
        }

        assert(condition);
    }
#else
    template <typename tT, typename... tArgs>
    FORCEINLINE void Assert(const tT &, tArgs &&...)
    {
    }
#endif

} // namespace SyncLibInternal

#endif
