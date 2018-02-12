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
#ifndef __DEFINES_H__
#define __DEFINES_H__

// From boost
#if !defined(SYNCLIB_FORCEINLINE)
#  if defined(_MSC_VER)
#    define SYNCLIB_FORCEINLINE __forceinline
#  elif defined(__GNUC__) && __GNUC__ > 3
// Clang also defines __GNUC__ (as 4)
#    define SYNCLIB_FORCEINLINE inline __attribute__ ((__always_inline__))
#  else
#    define SYNCLIB_FORCEINLINE inline
#  endif
#endif

// From boost
#if !defined(SYNCLIB_NOINLINE)
#  if defined(_MSC_VER)
#    define SYNCLIB_NOINLINE __declspec(noinline)
#  elif defined(__GNUC__) && __GNUC__ > 3
// Clang also defines __GNUC__ (as 4)
#    if defined(__CUDACC__)
// nvcc doesn't always parse __noinline__,
#      define SYNCLIB_NOINLINE __attribute__ ((noinline))
#    else
#      define SYNCLIB_NOINLINE __attribute__ ((__noinline__))
#    endif
#  else
#    define SYNCLIB_NOINLINE
#  endif
#endif

#endif