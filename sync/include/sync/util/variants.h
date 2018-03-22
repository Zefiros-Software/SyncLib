#pragma once
#ifndef __SYNCLIB_VARIANTS_H__
#define __SYNCLIB_VARIANTS_H__
#include <variant>

template<typename tT, typename tVariant>
tT &GetVariantReference(tVariant &x)
{
    if (std::holds_alternative<tT>(x))
    {
        return std::get<tT>(x);
    }
    else
    {
        return *std::get<tT *>(x);
    }
}

template<typename tT, typename tVariant>
const tT &GetVariantReference(const tVariant &x)
{
    if (std::holds_alternative<tT>(x))
    {
        return std::get<tT>(x);
    }
    else
    {
        return *std::get<tT *>(x);
    }
}

template<typename tT, typename tVariant>
constexpr bool HoldsAlternative(const tVariant &x)
{
    return std::holds_alternative<tT>(x) || std::holds_alternative<tT *>(x);
}

template<typename tT, typename tS, typename... tArgs>
struct AnyType
{
    constexpr static bool value = std::is_same_v<tT, tS> || AnyType<tT, tArgs...>::value;
};

template<typename tT, typename tS>
struct AnyType<tT, tS>
{
    constexpr static bool value = std::is_same_v<tT, tS>;
};

template<typename... tArgs>
struct FilterHelper
{
    template<typename tT>
    constexpr static bool Contains = AnyType<tT, tArgs...>::value;

    template<typename tT>
    using Extend = typename std::conditional_t<Contains<tT>, FilterHelper<tArgs...>, FilterHelper<tArgs..., tT>>;

    template<template<typename...> typename tTemplate>
    using Templated = tTemplate<tArgs...>;
};

template<typename tFilterHelper, typename tS, typename... tArgs>
struct Filter
{
    using type = typename Filter<typename tFilterHelper::template Extend<tS>, tArgs...>::type;
};

template<typename tFilterHelper, typename tS>
struct Filter<tFilterHelper, tS>
{
    using type = typename tFilterHelper::template Extend<tS>;
};

template<typename tT, typename... tArgs>
struct Filtered
{
    using type = typename Filter<FilterHelper<tT>, tArgs...>::type;
};

template<typename tT>
struct Filtered<tT>
{
    using type = FilterHelper<tT>;
};

template<typename... tArgs>
using FilteredType = typename Filtered<tArgs...>::type;

template<typename... tArgs>
using ValueOrPointer = std::variant<tArgs..., tArgs *...>;

template<typename... tArgs>
using SingleAndVectorVariant = ValueOrPointer<tArgs..., std::vector<tArgs>...>;

template<typename... tArgs>
using SingleVariant = ValueOrPointer<tArgs...>;

template<typename... tArgs>
using VectorVariant = ValueOrPointer<std::vector<tArgs>...>;

template<template<typename...> typename tVariant, typename... tArgs>
using NumericVariant = typename FilteredType<tArgs..., std::make_signed_t<tArgs>..., char, float, double, long double>
                       ::template Templated<tVariant>;

template<template<typename...> typename tVariant>
using AllNumericVariant =
    NumericVariant<tVariant, uint8_t, uint16_t, uint32_t, size_t, unsigned long int, uint64_t, unsigned long long int>;

using ValueVariants = AllNumericVariant<SingleVariant>;
using VectorVariants = AllNumericVariant<VectorVariant>;
using AllVariants = AllNumericVariant<SingleAndVectorVariant>;

#endif