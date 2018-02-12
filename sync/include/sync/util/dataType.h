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
#ifndef __SYNCLIB__DATATYPE_H__
#define __SYNCLIB__DATATYPE_H__

namespace SyncLib
{
    namespace Internal
    {
        enum class DataType : uint8_t
        {
            Serialisable = 0x0,
            U8 = 0x1,
            U16 = 0x2,
            U32 = 0x3,
            U64 = 0x4,
            S8 = 0x1 | 0x40,
            S16 = 0x2 | 0x40,
            S32 = 0x3 | 0x40,
            S64 = 0x4 | 0x40,
            String = 0x5,
            Float = 0x6,
            Double = 0x7,
            Array = 0x80
        };

        template<typename tT>
        struct DataTypeEnum
        {
            static constexpr DataType value = DataType::Serialisable;
        };

#define SYNCLIB_DATATYPE_MAPPING(tT, dataType)  \
template<>                                      \
struct DataTypeEnum<tT>                         \
{                                               \
    static constexpr DataType value = dataType; \
};

        SYNCLIB_DATATYPE_MAPPING(uint8_t,  DataType::U8)
        SYNCLIB_DATATYPE_MAPPING(uint16_t, DataType::U16)
        SYNCLIB_DATATYPE_MAPPING(uint32_t, DataType::U32)
        SYNCLIB_DATATYPE_MAPPING(uint64_t, DataType::U64)
        SYNCLIB_DATATYPE_MAPPING(int8_t,   DataType::S8)
        SYNCLIB_DATATYPE_MAPPING(int16_t,  DataType::S16)
        SYNCLIB_DATATYPE_MAPPING(int32_t,  DataType::S32)
        SYNCLIB_DATATYPE_MAPPING(int64_t,  DataType::S64)

        SYNCLIB_DATATYPE_MAPPING(float, DataType::Float)
        SYNCLIB_DATATYPE_MAPPING(double, DataType::Double)
        SYNCLIB_DATATYPE_MAPPING(std::string, DataType::String)

#undef SYNCLIB_DATATYPE_MAPPING

        template<typename tT>
        struct DataTypeEnum<std::vector<tT>>
        {
            static constexpr DataType value = DataType::Array | DataTypeEnum<tT>::value;
        };

        constexpr DataType operator~(const DataType &dataType)
        {
            return static_cast<DataType>(~static_cast<uint8_t>(dataType));
        }

        constexpr DataType operator&(const DataType &first, const DataType &second)
        {
            return static_cast<DataType>(static_cast<uint8_t>(first) & static_cast<uint8_t>(second));
        }

        constexpr DataType operator|(const DataType &first, const DataType &second)
        {
            return static_cast<DataType>(static_cast<uint8_t>(first) | static_cast<uint8_t>(second));
        }

        constexpr DataType operator^(const DataType &first, const DataType &second)
        {
            return static_cast<DataType>(static_cast<uint8_t>(first) ^ static_cast<uint8_t>(second));
        }
    }
}

#endif