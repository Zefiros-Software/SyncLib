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
#ifndef __SYNCLIB_JSON_H__
#define __SYNCLIB_JSON_H__

#include "nlohmann/json.hpp"

#include <armadillo>

// ReSharper disable CppInconsistentNaming
namespace nlohmann
{
    template<typename tVec, typename tT>
    struct VecSerialiser
    {
        static tVec from_json(const json &j)
        {
            const size_t size = j.size();
            tVec vec(size, arma::fill::zeros);
            std::copy_n(j.begin(), size, vec.begin());
            return vec;
        }

        static void to_json(json &j, const tVec &vec)
        {
            for (auto &x : vec)
            {
                j.push_back(x);
            }
        }
    };

    template<typename tT>
    struct adl_serializer<arma::Row<tT>, void> : VecSerialiser<arma::Row<tT>, tT> {  };

    template<typename tT>
    struct adl_serializer<arma::Col<tT>, void> : VecSerialiser<arma::Col<tT>, tT> {  };

    template<typename tMat, typename tT>
    struct MatSerialiser
    {
        static tMat from_json(const json &j)
        {
            const size_t rows = j.size();
            tMat matrix(j.size(), j[0].size(), arma::fill::zeros);

            arma::Row<tT> matRow(matrix.n_cols);

            for (size_t row = 0; row < rows; ++row)
            {
                std::copy(j[row].begin(), j[row].end(), matRow.begin());
                matrix.row(row) = matRow;
            }

            return matrix;
        }

        static void to_json(json &j, const tMat &matrix)
        {
            for (size_t row = 0, rows = matrix.n_rows; row < rows; ++row)
            {
                j.push_back(arma::Row<tT>(matrix.row(row)));
            }
        }
    };

    template<typename tT>
    struct adl_serializer<arma::Mat<tT>, void> : MatSerialiser<arma::Mat<tT>, tT> {};
} // namespace nlohmann
// ReSharper restore CppInconsistentNaming

using json = nlohmann::json;

#endif