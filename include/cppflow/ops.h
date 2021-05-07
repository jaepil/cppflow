//
// Created by serizba on 31/7/20.
//

#ifndef CPPFLOW2_OPS_H
#define CPPFLOW2_OPS_H

#include "raw_ops.h"
#include "tensor.h"


namespace cppflow {

inline Tensor operator+(const Tensor& x, const Tensor& y) {
    return ops::Add(x, y);
}

inline Tensor operator-(const Tensor& x, const Tensor& y) {
    return ops::Sub(x, y);
}

inline Tensor operator*(const Tensor& x, const Tensor& y) {
    return ops::Mul(x, y);
}

inline Tensor operator/(const Tensor& x, const Tensor& y) {
    return ops::Div(x, y);
}

/**
 * @return A string representing t in the form:
 * <Tensor: shape=?, dtype=?, data=?>
 * ?)
 */
inline std::string to_string(const Tensor& t) {
    auto output
        = ops::StringFormat({t.shape(), Tensor {to_string_view(t.dtype())}, t},
                            "<Tensor: shape=%s, dtype=%s, data=%s>");
    auto handle = output.get_tensor();

    auto* data = static_cast<TF_TString*>(TF_TensorData(handle.get()));
    auto result = std::string {TF_TString_GetDataPointer(data),
                               TF_TString_GetSize(data)};

    return result;
}

inline std::ostream& operator<<(std::ostream& os, const Tensor& t) {
    os << to_string(t);
    return os;
}

}    // namespace cppflow

#endif    // CPPFLOW2_OPS_H
