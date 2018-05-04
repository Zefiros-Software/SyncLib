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
#ifndef __SYNCLIB_SPLITFOR_H__
#define __SYNCLIB_SPLITFOR_H__

#include "preproc/preproc.h"

#include <armadillo>

namespace SyncLib
{
    namespace Util
    {
        template <typename tT, typename tFunc>
        FORCEINLINE void SplitFor(const tT &start, const tT &split, const tT &end, const tFunc &body)
        {
            for (tT i = split; i < end; ++i)
            {
                body(i);
            }

            for (tT i = start; i < split; ++i)
            {
                body(i);
            }
        }

        template <typename tT>
        tT NextPowerOfTwo(tT v)
        {
            --v;
            v |= v >> 1;
            v |= v >> 2;
            v |= v >> 4;
            v |= v >> 8;
            v |= v >> 16;
            return ++v;
        }

        template <typename tT>
        arma::Col<tT> ArmaRange(const tT &start, const tT &stop, const tT &delta)
        {
            return arma::regspace<arma::Col<tT>>(start, delta, stop);
        }

        template <typename tT>
        arma::Col<tT> ArmaRange(const tT &start, const tT &stop)
        {
            return arma::regspace<arma::Col<tT>>(start, stop);
        }

        template <typename tIterator>
        auto MinMax(tIterator begin, tIterator end)
        {
            auto minmaxtime = std::minmax_element(begin, end);
            return std::make_tuple(*minmaxtime.second, *minmaxtime.first);
        }

        template <typename tT>
        auto ArgMax(const arma::Mat<tT> &data)
        {
            auto argMax = arma::ind2sub(arma::SizeMat(data.n_rows, data.n_cols), data.index_max());
            return std::make_tuple(argMax[0], argMax[1]);
        }

        template <typename tMat>
        auto ArgMax(const tMat &data, const arma::SizeMat &shape)
        {
            auto argMax = arma::ind2sub(shape, data.index_max());
            return std::make_tuple(argMax[0], argMax[1]);
        }

        template <typename tT>
        auto ArgMin(const arma::Mat<tT> &data)
        {
            auto argMin = arma::ind2sub(arma::SizeMat(data.n_rows, data.n_cols), data.index_min());
            return std::make_tuple(argMin[0], argMin[1]);
        }

        template <typename tMat>
        auto ArgMin(const tMat &data, const arma::SizeMat &shape)
        {
            auto argMin = arma::ind2sub(shape, data.index_min());
            return std::make_tuple(argMin[0], argMin[1]);
        }

        template <typename tDelim>
        inline void Split(std::string_view s, tDelim delimiter, std::vector<std::string_view> &words)
        {
            for (size_t offset = s.find_first_not_of(delimiter), next = s.find_first_of(delimiter, offset), size = s.size(); offset < size;
                 offset = s.find_first_not_of(delimiter, next), next = s.find_first_of(delimiter, offset))
            {
                words.push_back(s.substr(offset, next - offset));
            }
        }

        template <typename tDelim>
        inline std::vector<std::string_view> Split(std::string_view s, tDelim delimiter)
        {
            std::vector<std::string_view> words;
            Split(s, delimiter, words);
            return words;
        }

        inline std::string_view Trim(std::string_view s)
        {
            const std::string_view delimiter(" \t\r\n");
            return s.substr(s.find_first_not_of(delimiter), s.find_last_not_of(delimiter) + 1);
        }
    } // namespace Util
} // namespace SyncLib

#endif
