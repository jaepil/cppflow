//
// Created by serizba on 31/7/20.
//

#ifndef CPPFLOW2_OPS_H
#define CPPFLOW2_OPS_H


#include "tensor.h"
#include "raw_ops.h"

namespace cppflow {

    /**
     * @name Operators
     */
    //@{

    /**
     * @returns x + y elementwise
     */
    tensor operator+(const tensor& x, const tensor& y);

    /**
     * @returns x - y elementwise
     */
    tensor operator-(const tensor& x, const tensor& y);

    /**
     * @returns x * y elementwise
     */
    tensor operator*(const tensor& x, const tensor& y);

    /**
     * @return x / y elementwise
     */
    tensor operator/(const tensor& x, const tensor& y);

    std::ostream& operator<<(std::ostream& os, const cppflow::tensor& t);

    //@}

    /**
     * @return A string representing t in the form:
     * (tensor: shape=?, data=
     * ?)
     */
    std::string to_string(const tensor& t);
}

/******************************
 *   IMPLEMENTATION DETAILS   *
 ******************************/

namespace cppflow {

    // Operators

    inline tensor operator+(const tensor& x, const tensor& y) {
        return add(x, y);
    }

    inline tensor operator-(const tensor& x, const tensor& y) {
        return sub(x, y);
    }

    inline tensor operator*(const tensor& x, const tensor& y) {
        return mul(x, y);
    }

    inline tensor operator/(const tensor& x, const tensor& y) {
        return div(x, y);
    }

    inline std::ostream& operator<<(std::ostream& os, const cppflow::tensor& t) {
        std::string res =  to_string(t);
        return os << res;
    }

    inline std::string to_string(const tensor& t) {
        auto output = string_format({t.shape(), t}, "(tensor: shape=%s, dtype=" + to_string(t.dtype()) + ", data=%s)");
        auto handle = output.get_tensor();

        auto* data = static_cast<TF_TString*>(TF_TensorData(handle.get()));
        auto result = std::string {TF_TString_GetDataPointer(data), TF_TString_GetSize(data)};

        return result;
    }

}

#endif //CPPFLOW2_OPS_H
