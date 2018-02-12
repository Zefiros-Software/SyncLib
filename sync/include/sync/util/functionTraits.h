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
#ifndef __SYNCLIB_FUNCTIONTRAITS_H__
#define __SYNCLIB_FUNCTIONTRAITS_H__

namespace SyncLib
{
    namespace Internal
    {
        template <typename tFunc>
        struct FunctionTraits : public FunctionTraits<decltype(&tFunc::operator())> {};

        template <typename tClass, typename tReturn, typename... tArgs>
        struct FunctionTraits<tReturn(tClass::*)(tArgs...) const>
        {
            struct result
            {
                using type = tReturn;
            };

            template <size_t tIndex>
            struct arg
            {
                using type = typename std::tuple_element<tIndex, std::tuple<tArgs...>>::type;
            };
        };
    }
}

#endif