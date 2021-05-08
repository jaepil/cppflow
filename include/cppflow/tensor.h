//
// Created by serizba on 27/6/20.
//

#ifndef CPPFLOW2_TENSOR_H
#define CPPFLOW2_TENSOR_H

#include "context.h"
#include "datatype.h"

#include <tensorflow/c/eager/c_api.h>
#include <tensorflow/c/tf_tensor.h>

#include <cstring>
#include <memory>
#include <span>
#include <string>
#include <vector>


namespace cppflow {

/**
 * @class tensor
 * @brief A TensorFlow eager tensor wrapper
 *
 */
class Tensor {
public:
    Tensor() = default;

    /**
     * Creates a tensor with the given values and specified shape
     * @tparam T A type that can be convertible into a tensor
     * @param values The values to be converted (in a flattened version)
     * @param shape The shape of the converted tensor
     */
    template<typename T>
    Tensor(const std::vector<T>& values, const std::vector<int64_t>& shape);

    /**
     * Creates a flat tensor with the given values
     * @tparam T A type that can be convertible into a tensor
     * @param values The values to be converted
     */
    template<typename T>
    Tensor(const std::initializer_list<T>& values,
           const std::initializer_list<int64_t>& shape);

    /**
     * Creates a tensor with the given value
     * @tparam T A type that can be convertible into a tensor
     * @param value The value to be converted
     */
    template<typename T>
    explicit Tensor(const T& value);

    explicit Tensor(const char* value);
    explicit Tensor(const char* value, size_t size);
    explicit Tensor(const std::string& value);
    explicit Tensor(const std::string_view& value);

    /**
     * @return Shape of the tensor
     */
    Tensor shape() const;

    /**
     * @param on_memory If false, the function will return the name of the
     * device that produced the tensor. If true, the function will return the
     * name of the device in whose memory the tensor resides
     * @return Returns the name of the device of the tensor
     */
    std::string_view device(bool on_memory = false) const;

    /**
     * @return The tensor datatype
     */
    datatype dtype() const;

    /**
     * Converts the tensor into a C++ vector
     * @tparam T The c++ type (must be equivalent to the tensor type)
     * @return A vector representing the flat tensor
     */
    template<typename T>
    std::span<T> get_data() const;

    ~Tensor() = default;
    Tensor(const Tensor& tensor) = default;
    Tensor(Tensor&& tensor) = default;
    Tensor& operator=(const Tensor& other) = default;
    Tensor& operator=(Tensor&& other) = default;

    explicit Tensor(TFE_TensorHandle* handle);
    explicit Tensor(TF_Tensor* t);

    // NOTE: Usually, one should not call get_eager_handle() or get_tensor()
    // below.
    //       They are designed for implementation details in cppflow.
    //       If you are calling them directly, it is likely that you are using
    //       some tenforflow APIs not supported in cppflow.

    // Additional NOTE: TF_Tensor is an immutable tensor inside tensorflow.
    // TFE_TensorHandle is a TF_Tensor and the associated device, plus some data
    // cache

    // TODO: Need to determine if we can mark the return value or *this as const
    std::shared_ptr<TFE_TensorHandle> get_eager_handle() const {
        return tfe_handle_;
    }

    // Get the TF_Tensor data from the eager handle
    // Call `get_data<T>()` instead if possible
    // NOTE: Changes to the returned TF_Tensor may not be reflected in the
    // actual device memory!
    //       Do *NOT* modify the returned TF_Tensor!
    //       See comments of `tf_tensor` for more details.
    std::shared_ptr<TF_Tensor> get_tensor() const;

private:
    Tensor(TF_DataType type, const void* data, size_t len,
           const std::vector<int64_t>& shape);

private:
    // This member serves as a local cache of the data in tfe_handle.
    // It refers to `local_mirrors_` if on device, or `data_` if on host CPU.
    // Changes to this variable may not be reflected in the actual device
    // memory, e.g. on GPUs or on remote nodes. Access it via get_tensor() if
    // not in constructor
    mutable std::shared_ptr<TF_Tensor> tf_tensor_;
    std::shared_ptr<TFE_TensorHandle> tfe_handle_;
};

}    // namespace cppflow


/******************************
 *   IMPLEMENTATION DETAILS   *
 ******************************/


namespace cppflow {

inline Tensor::Tensor(TF_DataType type, const void* data, size_t len,
                      const std::vector<int64_t>& shape)
    : tf_tensor_([&]() {
          auto tensor = std::shared_ptr<TF_Tensor>(
              TF_AllocateTensor(type, shape.data(), shape.size(), len),
              TF_DeleteTensor);
          std::memcpy(TF_TensorData(tensor.get()), data,
                      TF_TensorByteSize(tensor.get()));
          return tensor;
      }()),
      tfe_handle_([&]() {
          auto handle = std::shared_ptr<TFE_TensorHandle>(
              TFE_NewTensorHandle(tf_tensor_.get(), context::get_status()),
              TFE_DeleteTensorHandle);
          status_check(context::get_status());
          return handle;
      }()) {
}

template<typename T>
Tensor::Tensor(const std::vector<T>& values, const std::vector<int64_t>& shape)
    : Tensor(deduce_tf_type<T>(), values.data(), values.size() * sizeof(T),
             shape) {
}

template<typename T>
Tensor::Tensor(const std::initializer_list<T>& values,
               const std::initializer_list<int64_t>& shape)
    : Tensor(std::vector<T> {values}, std::vector {shape}) {
}

template<typename T>
Tensor::Tensor(const T& value)
    : Tensor(deduce_tf_type<T>(), &value, sizeof(T), {}) {
}

inline Tensor::Tensor(const char* value) : Tensor(std::string_view {value}) {
}

inline Tensor::Tensor(const char* value, size_t size)
    : Tensor(std::string_view {value, size}) {
}

inline Tensor::Tensor(const std::string& value)
    : Tensor(std::string_view {value}) {
}

inline Tensor::Tensor(const std::string_view& value)
    : tf_tensor_(), tfe_handle_() {
    TF_TString data;
    TF_TString_Init(&data);
    TF_TString_Copy(&data, value.data(), value.size());

    int64_t dims = -1;
    tf_tensor_.reset(TF_AllocateTensor(TF_STRING, &dims, 0, sizeof(data)),
                     TF_DeleteTensor);
    std::memcpy(TF_TensorData(tf_tensor_.get()), &data,
                TF_TensorByteSize(tf_tensor_.get()));
    tfe_handle_.reset(
        TFE_NewTensorHandle(tf_tensor_.get(), context::get_status()),
        TFE_DeleteTensorHandle);
    status_check(context::get_status());
}

inline Tensor::Tensor(TFE_TensorHandle* handle) {
    tfe_handle_ = {handle, TFE_DeleteTensorHandle};
}

inline Tensor::Tensor(TF_Tensor* t) {
    tf_tensor_ = {t, TF_DeleteTensor};
    tfe_handle_ = {TFE_NewTensorHandle(tf_tensor_.get(), context::get_status()),
                   TFE_DeleteTensorHandle};
    status_check(context::get_status());
}

inline Tensor Tensor::shape() const {
    auto op = TFE_NewOp(context::get_context(), "Shape", context::get_status());
    status_check(context::get_status());

    TFE_OpAddInput(op, tfe_handle_.get(), context::get_status());
    status_check(context::get_status());

    // Output type should be int64_t
    TFE_OpSetAttrType(op, "out_type", cppflow::datatype::TF_INT64);

    // EXECUTE
    int n = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &n, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);

    return Tensor {res[0]};
}

inline std::string_view Tensor::device(bool on_memory) const {
    auto name = std::string_view {};

    if (on_memory)
        name = TFE_TensorHandleBackingDeviceName(tfe_handle_.get(),
                                                 context::get_status());
    else
        name = TFE_TensorHandleDeviceName(tfe_handle_.get(),
                                          context::get_status());

    status_check(context::get_status());

    return name;
}

template<typename T>
std::span<T> Tensor::get_data() const {
    // Check if asked datatype and tensor datatype match
    if (this->dtype() != deduce_tf_type<T>()) {
        auto type1 = cppflow::to_string(deduce_tf_type<T>());
        auto type2 = cppflow::to_string(this->dtype());
        auto error = "Datatype in function get_data (" + type1
                     + ") does not match tensor datatype (" + type2 + ")";
        throw std::runtime_error(error);
    }

    auto res_tensor = get_tensor();

    // Check tensor data is not empty
    auto raw_data = TF_TensorData(res_tensor.get());
    // this->error_check(raw_data != nullptr, "Tensor data is empty");

    size_t size = TF_TensorByteSize(res_tensor.get())
                  / TF_DataTypeSize(TF_TensorType(res_tensor.get()));

    // Convert to correct type
    auto* begin = static_cast<T*>(raw_data);
    auto* end = begin + size;

    return std::span<T> {begin, end};
}

inline datatype Tensor::dtype() const {
    return TFE_TensorHandleDataType(tfe_handle_.get());
}

// NOTE: Changes to the returned TF_Tensor are not reflected in the actual
// device memory!
inline std::shared_ptr<TF_Tensor> Tensor::get_tensor() const {
    if (!tf_tensor_) {
        tf_tensor_ = {
            TFE_TensorHandleResolve(tfe_handle_.get(), context::get_status()),
            TF_DeleteTensor};
        status_check(context::get_status());
    }
    return tf_tensor_;
}
}    // namespace cppflow

#endif    // CPPFLOW2_TENSOR_H
