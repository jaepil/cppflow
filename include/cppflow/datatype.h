//
// Created by serizba on 12/7/20.
//

#ifndef CPPFLOW2_DATATYPE_H
#define CPPFLOW2_DATATYPE_H

#include <tensorflow/c/tf_datatype.h>

#include <type_traits>
#include <string>
#include <typeinfo>
#include <ostream>
#include <stdexcept>

namespace cppflow {

    using datatype = TF_DataType;

    /**
     * @return A string representing dt
     *
     */
    inline std::string_view to_string_view(datatype dt) {
        switch (dt) {
            case TF_FLOAT:
                return "TF_FLOAT";
            case TF_DOUBLE:
                return "TF_DOUBLE";
            case TF_INT32:
                return "TF_INT32";
            case TF_UINT8:
                return "TF_UINT8";
            case TF_INT16:
                return "TF_INT16";
            case TF_INT8:
                return "TF_INT8";
            case TF_STRING:
                return "TF_STRING";
            case TF_COMPLEX64:
                return "TF_COMPLEX64";
            case TF_INT64:
                return "TF_INT64";
            case TF_BOOL:
                return "TF_BOOL";
            case TF_QINT8:
                return "TF_QINT8";
            case TF_QUINT8:
                return "TF_QUINT8";
            case TF_QINT32:
                return "TF_QINT32";
            case TF_BFLOAT16:
                return "TF_BFLOAT16";
            case TF_QINT16:
                return "TF_QINT16";
            case TF_QUINT16:
                return "TF_QUINT16";
            case TF_UINT16:
                return "TF_UINT16";
            case TF_COMPLEX128:
                return "TF_COMPLEX128";
            case TF_HALF:
                return "TF_HALF";
            case TF_RESOURCE:
                return "TF_RESOURCE";
            case TF_VARIANT:
                return "TF_VARIANT";
            case TF_UINT32:
                return "TF_UINT32";
            case TF_UINT64:
                return "TF_UINT64";
            default:
                return "DATATYPE_NOT_KNOWN";
        }
    }
    inline std::string to_string(datatype dt) {
        return std::string {to_string_view(dt)};
    }

    /**
     *
     * @tparam T
     * @return The TensorFlow type of T
     */
    template<typename T>
    inline constexpr TF_DataType deduce_tf_type() {
        if constexpr (std::is_same_v<T, float>)
            return TF_FLOAT;
        if constexpr (std::is_same_v<T, double>)
            return TF_DOUBLE;
        if constexpr (std::is_same_v<T, int32_t>)
            return TF_INT32;
        if constexpr (std::is_same_v<T, uint8_t>)
            return TF_UINT8;
        if constexpr (std::is_same_v<T, int16_t>)
            return TF_INT16;
        if constexpr (std::is_same_v<T, int8_t>)
            return TF_INT8;
        if constexpr (std::is_same_v<T, int64_t>)
            return TF_INT64;
        if constexpr (std::is_same_v<T, unsigned char>
                    || std::is_same_v<T, bool>)
            return TF_BOOL;
        if constexpr (std::is_same_v<T, uint16_t>)
            return TF_UINT16;
        if constexpr (std::is_same_v<T, uint32_t>)
            return TF_UINT32;
        if constexpr (std::is_same_v<T, uint64_t>)
            return TF_UINT64;
        if constexpr (std::is_same_v<T, std::string> 
                    || std::is_same_v<T, std::string_view>)
            return TF_STRING;

        // decode with `c++filt --type $output` for gcc
        throw std::runtime_error {"Could not deduce type! type_name: "
                                + std::string {typeid(T).name()}};
    }

    /**
     * @return  The stream os after inserting the string representation of dt
     *
     */
    inline std::ostream& operator<<(std::ostream& os, datatype dt) {
        os << to_string_view(dt);
        return os;
    }

}
#endif //CPPFLOW2_DATATYPE_H
