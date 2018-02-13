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
#ifndef __SYNCLIB_RANGE_H__
#define __SYNCLIB_RANGE_H__

#include <limits>
#include <tuple>

namespace SyncLib
{
    namespace Ranges
    {
        namespace Internal
        {

            template<typename tT>
            class SteppedRange;

            template<typename tT>
            class ReverseSteppedRange
            {
            public:

                class ReverseSteppedRangeIterator
                {
                public:

                    ReverseSteppedRangeIterator(tT current, tT step)
                        : mCurrent(current),
                          mStep(step)
                    {}

                    bool operator!=(const ReverseSteppedRangeIterator &other)
                    {
                        return other.mCurrent != mCurrent;
                    }

                    ReverseSteppedRangeIterator &operator++()
                    {
                        mCurrent -= mStep;
                        return *this;
                    }

                    const tT &operator*() const
                    {
                        return mCurrent;
                    }

                private:

                    tT mCurrent, mStep;
                };

                using iterator = ReverseSteppedRangeIterator;

                ReverseSteppedRange(tT from, tT until, tT step)
                    : mStart(from),
                      mStop(until),
                      mStep(step)
                {
                    static_assert(std::numeric_limits<tT>::is_integer, "Only integer Ranges are supported through this interface");
                }

                iterator begin() const
                {
                    tT endDiff = (mStop - mStart) % mStep;
                    return mStop - (endDiff > 0 ? endDiff : mStep);
                }

                iterator end() const
                {
                    return mStart - mStep;
                }

                SteppedRange<tT> Reverse()
                {
                    return SteppedRange<tT>(mStart, mStop, mStep);
                }

            private:

                tT mStart, mStop, mStep;
            };

            template<typename tT>
            class SteppedRange
            {
            public:

                class SteppedRangeIterator
                {
                public:

                    SteppedRangeIterator(tT current, tT step)
                        : mCurrent(current),
                          mStep(step)
                    {}

                    bool operator!=(const SteppedRangeIterator &other)
                    {
                        return other.mCurrent != mCurrent;
                    }

                    SteppedRangeIterator &operator++()
                    {
                        mCurrent += mStep;
                        return *this;
                    }

                    const tT &operator*() const
                    {
                        return mCurrent;
                    }

                private:

                    tT mCurrent, mStep;
                };

                using iterator = SteppedRangeIterator;

                SteppedRange(tT from, tT until, tT step)
                    : mStart(from),
                      mStop(until + step - 1 - (until + step - 1) % step),
                      mStep(step)
                {
                    static_assert(std::numeric_limits<tT>::is_integer, "Only integer Ranges are supported through this interface");
                }

                iterator begin() const
                {
                    return iterator(mStart, mStep);
                }

                iterator end() const
                {
                    return iterator(mStop, mStep);
                }

                ReverseSteppedRange<tT> Reverse()
                {
                    return ReverseSteppedRange<tT>(mStart, mStop, mStep);
                }

            private:

                tT mStart, mStop, mStep;
            };

            template<typename tT>
            class Range;

            template<typename tT>
            class ReverseRange
            {
            public:

                class ReverseRangeIterator
                {
                public:

                    ReverseRangeIterator(tT current)
                        : mCurrent(current)
                    {
                    }

                    ReverseRangeIterator &operator++()
                    {
                        --mCurrent;
                        return *this;
                    }

                    bool operator!=(const ReverseRangeIterator &other)
                    {
                        return other.mCurrent != mCurrent;
                    }

                    const tT &operator*() const
                    {
                        return mCurrent;
                    }

                private:

                    tT mCurrent;
                };

                using iterator = ReverseRangeIterator;

                ReverseRange(tT from, tT until)
                    : mStart(from),
                      mStop(until)
                {
                    static_assert(std::numeric_limits<tT>::is_integer, "Only integer Ranges are supported through this interface");
                }

                iterator begin() const
                {
                    return mStop - 1;
                }

                iterator end() const
                {
                    return mStart - 1;
                }

                Range<tT> Reverse()
                {
                    return Range<tT>(mStart, mStop);
                }

            private:

                tT mStart, mStop;
            };

            template<typename tT>
            class Range
            {
            public:

                class RangeIterator
                {
                public:

                    using value_type = tT;

                    RangeIterator(tT current)
                        : mCurrent(current)
                    {
                        static_assert(std::numeric_limits<tT>::is_integer, "Only integer Ranges are supported through this interface");
                    }

                    bool operator!=(const RangeIterator &other)
                    {
                        return other.mCurrent != mCurrent;
                    }

                    RangeIterator &operator++()
                    {
                        ++mCurrent;
                        return *this;
                    }

                    const tT &operator*() const
                    {
                        return mCurrent;
                    }

                private:

                    tT mCurrent;
                };

                using iterator = RangeIterator;
                using value_type = tT;

                Range(tT from, tT until)
                    : mStart(from),
                      mStop(until)
                {
                    static_assert(std::numeric_limits<tT>::is_integer, "Only integer Ranges are supported through this interface");
                }

                iterator begin() const
                {
                    return mStart;
                }

                iterator end() const
                {
                    return mStop;
                }

                ReverseRange<tT> Reverse()
                {
                    return ReverseRange<tT>(mStart, mStop);
                }

            private:

                tT mStart, mStop;
            };

            template<typename tIter>
            class IteratorRange
            {
            public:

                using iterator = tIter;

                IteratorRange(const tIter &begin, const tIter &end)
                    : mBegin(begin),
                      mEnd(end)
                {
                }

                tIter begin() const
                {
                    return mBegin;
                }

                tIter end() const
                {
                    return mEnd;
                }

            private:

                tIter mBegin, mEnd;
            };

            template<typename tIter1, typename tIter2>
            class ZipIterators
            {
            public:

                using tValue1 = typename tIter1::value_type;
                using tValue2 = typename tIter2::value_type;

                class ZipIteratorsIterator
                {
                public:

                    ZipIteratorsIterator(const tIter1 &current, const tIter2 &zip)
                        : mCurrent(current),
                          mZip(zip)
                    {}

                    bool operator!=(const ZipIteratorsIterator &other)const
                    {
                        return mCurrent != other.mCurrent;
                    }

                    ZipIteratorsIterator &operator++()
                    {
                        ++mCurrent;
                        ++mZip;
                        return *this;
                    }

                    auto operator*()
                    {
                        return std::tuple<tValue1 &, tValue2 &>(*mCurrent, *mZip);
                    }

                private:

                    tIter1 mCurrent;
                    tIter2 mZip;
                };

                using iterator = ZipIteratorsIterator;

                ZipIterators(const tIter1 &begin, const tIter1 &end, const tIter2 &zip)
                    : mBegin(begin),
                      mEnd(end),
                      mZip(zip)
                {}

                iterator begin() const
                {
                    return ZipIteratorsIterator(mBegin, mZip);
                }

                iterator end() const
                {
                    return ZipIteratorsIterator(mEnd, mZip);
                }

            private:

                tIter1 mBegin, mEnd;
                tIter2 mZip;
            };
        }

        template<typename tT>
        Internal::Range<tT> Range(tT until)
        {
            return Internal::Range<tT>(0, until);
        }

        template<typename tT>
        Internal::Range<tT> Range(tT from, tT until)
        {
            return Internal::Range<tT>(from, until);
        }

        template<typename tT>
        Internal::SteppedRange<tT> Range(tT from, tT until, tT step)
        {
            return Internal::SteppedRange<tT>(from, until, step);
        }

        template<typename tIter>
        Internal::IteratorRange<tIter> IteratorRange(const tIter &begin, const tIter &end)
        {
            return Internal::IteratorRange<tIter>(begin, end);
        }

        template<typename tIter1, typename tIter2>
        Internal::ZipIterators<tIter1, tIter2> ZipIterators(const tIter1 &begin, const tIter1 &end, const tIter2 &zip)
        {
            return Internal::ZipIterators<tIter1, tIter2>(begin, end, zip);
        }

        template<typename tIterable1, typename tIterable2>
        auto Zip(tIterable1 &first, tIterable2 &second)
        {
            return Internal::ZipIterators<tIterable1, tIterable2>(first.begin(), first.end(), second.begin());
        }

        template<typename tIterable1, typename tIterable2>
        auto Zip(const tIterable1 &first, const tIterable2 &second)
        {
            return Internal::ZipIterators<tIterable1, tIterable2>(first.begin(), first.end(), second.begin());
        }
    }
}

#endif