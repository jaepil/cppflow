//
// Created by serizba on 12/7/20.
//

#ifndef CPPFLOW2_DATATYPE_H
#define CPPFLOW2_DATATYPE_H

#include <tensorflow/c/tf_datatype.h>

#include <ostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <typeinfo>


namespace cppflow {

using datatype = TF_DataType;

inline std::string_view to_string_view(datatype dt) {
    switch (dt) {
        case TF_FLOAT:
            return "float";
        case TF_DOUBLE:
            return "double";
        case TF_INT32:
            return "int32";
        case TF_UINT8:
            return "uint8";
        case TF_INT16:
            return "int16";
        case TF_INT8:
            return "int8";
        case TF_STRING:
            return "string";
        case TF_COMPLEX64:
            return "complex64";
        case TF_INT64:
            return "int64";
        case TF_BOOL:
            return "bool";
        case TF_QINT8:
            return "qint8";
        case TF_QUINT8:
            return "quint8";
        case TF_QINT32:
            return "qint32";
        case TF_BFLOAT16:
            return "bfloat16";
        case TF_QINT16:
            return "qint16";
        case TF_QUINT16:
            return "quint16";
        case TF_UINT16:
            return "uint16";
        case TF_COMPLEX128:
            return "complex128";
        case TF_HALF:
            return "half";
        case TF_RESOURCE:
            return "resource";
        case TF_VARIANT:
            return "variant";
        case TF_UINT32:
            return "uint32";
        case TF_UINT64:
            return "uint64";
        default:
            return "unknown";
    }
}

inline std::string to_string(datatype dt) {
    return std::string {to_string_view(dt)};
}

inline std::ostream& operator<<(std::ostream& os, datatype dt) {
    os << to_string_view(dt);
    return os;
}

template<typename T>
inline constexpr TF_DataType deduce_tf_type() {
    if constexpr (std::is_same_v<T, float>) {
        return TF_FLOAT;
    }
    if constexpr (std::is_same_v<T, double>) {
        return TF_DOUBLE;
    }
    if constexpr (std::is_same_v<T, int32_t>) {
        return TF_INT32;
    }
    if constexpr (std::is_same_v<T, uint8_t>) {
        return TF_UINT8;
    }
    if constexpr (std::is_same_v<T, int16_t>) {
        return TF_INT16;
    }
    if constexpr (std::is_same_v<T, int8_t>) {
        return TF_INT8;
    }
    if constexpr (std::is_same_v<T, int64_t>) {
        return TF_INT64;
    }
    if constexpr (std::is_same_v<T, unsigned char> || std::is_same_v<T, bool>) {
        return TF_BOOL;
    }
    if constexpr (std::is_same_v<T, uint16_t>) {
        return TF_UINT16;
    }
    if constexpr (std::is_same_v<T, uint32_t>) {
        return TF_UINT32;
    }
    if constexpr (std::is_same_v<T, uint64_t>) {
        return TF_UINT64;
    }
    if constexpr (std::is_same_v<
                      T, std::string> || std::is_same_v<T, std::string_view>) {
        return TF_STRING;
    }

    // decode with `c++filt --type $output` for gcc
    throw std::runtime_error {"Could not deduce type! type_name: "
                              + std::string {typeid(T).name()}};
}

}    // namespace cppflow
#endif    // CPPFLOW2_DATATYPE_H
