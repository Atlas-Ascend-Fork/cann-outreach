#ifndef OPS_MATH_DEV_TESTS_UT_COMMON_ANY_VALUE_H
#define OPS_MATH_DEV_TESTS_UT_COMMON_ANY_VALUE_H

#include <memory>
#include <cstdint>
#include <string>
#include <vector>

namespace Ops {
namespace Math {
class AnyValue {
public:
    enum ValueType
    {
        VT_STRING = 1,
        VT_FLOAT = 2,
        VT_BOOL = 3,
        VT_INT = 4,
        VT_LIST_LIST_INT = 10,
        VT_LIST_BASE = 1000,

        VT_LIST_FLOAT = static_cast<int32_t>(VT_LIST_BASE) + static_cast<int32_t>(VT_FLOAT),
        VT_LIST_BOOL = static_cast<int32_t>(VT_LIST_BASE) + static_cast<int32_t>(VT_BOOL),
        VT_LIST_INT = static_cast<int32_t>(VT_LIST_BASE) + static_cast<int32_t>(VT_INT),
    };

    AnyValue(ValueType type, const std::shared_ptr<void>& valuePtr) : type_(type), valuePtr_(valuePtr)
    {}
    ~AnyValue() = default;
    AnyValue(const AnyValue& anyValue) : type_(anyValue.type_), valuePtr_(anyValue.valuePtr_)
    {}

    template<typename T>
    static inline AnyValue CreateFrom(const T& value);

    ValueType type_;
    std::shared_ptr<void> valuePtr_;
};

template <>
inline AnyValue AnyValue::CreateFrom<std::string>(const std::string& value)
{
    return AnyValue(VT_STRING, std::make_shared<std::string>(value));
}

template <>
inline AnyValue AnyValue::CreateFrom<float>(const float& value)
{
    return AnyValue(VT_FLOAT, std::make_shared<float>(value));
}

template <>
inline AnyValue AnyValue::CreateFrom<bool>(const bool& value)
{
    return AnyValue(VT_BOOL, std::make_shared<bool>(value));
}

template <>
inline AnyValue AnyValue::CreateFrom<int64_t>(const int64_t& value)
{
    return AnyValue(VT_INT, std::make_shared<int64_t>(value));
}

template <>
inline AnyValue AnyValue::CreateFrom<std::vector<float>>(const std::vector<float>& value)
{
    return AnyValue(VT_LIST_FLOAT, std::make_shared<std::vector<float>>(value));
}

template <>
inline AnyValue AnyValue::CreateFrom<std::vector<bool>>(const std::vector<bool>& value)
{
    return AnyValue(VT_LIST_BOOL, std::make_shared<std::vector<bool>>(value));
}

template <>
inline AnyValue AnyValue::CreateFrom<std::vector<int64_t>>(const std::vector<int64_t>& value)
{
    return AnyValue(VT_LIST_INT, std::make_shared<std::vector<int64_t>>(value));
}

template <>
inline AnyValue AnyValue::CreateFrom<std::vector<std::vector<int64_t>>>(const std::vector<std::vector<int64_t>>& value)
{
    return AnyValue(VT_LIST_LIST_INT, std::make_shared<std::vector<std::vector<int64_t>>>(value));
}
} // namespace Math
} // namespace Ops

#endif // OPS_MATH_DEV_TESTS_UT_COMMON_ANY_VALUE_H
