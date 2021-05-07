//
// Created by serizba on 29/6/20.
//

#ifndef CPPFLOW2_MODEL_H
#define CPPFLOW2_MODEL_H

#include "buffer.h"
#include "context.h"
#include "defer.h"
#include "session_options.h"
#include "tensor.h"

#include <tensorflow/c/c_api.h>

#include <charconv>
#include <memory>
#include <string>
#include <tuple>
#include <vector>


namespace cppflow {

class Model {
public:
    explicit Model(const std::string_view& filename)
        : Model(filename, "serve") {
    }
    explicit Model(const std::string_view& filename,
                   const std::string_view& tag)
        : Model(filename, {tag}) {
    }
    explicit Model(const std::string_view& filename,
                   const std::initializer_list<std::string_view>& tags);

    std::vector<std::string> get_operations() const;
    std::vector<int64_t> get_operation_shape(
        const std::string& operation) const;

    std::vector<Tensor> forward(
        const std::vector<std::tuple<std::string_view, Tensor>>& inputs,
        const std::vector<std::string_view>& outputs);
    Tensor forward(const Tensor& input);

    std::vector<Tensor> operator()(
        const std::vector<std::tuple<std::string_view, Tensor>>& inputs,
        const std::vector<std::string_view>& outputs) {
        return forward(inputs, outputs);
    }
    Tensor operator()(const Tensor& input) {
        return forward(input);
    }

    ~Model() = default;
    Model(const Model& model) = default;
    Model(Model&& model) = default;
    Model& operator=(const Model& other) = default;
    Model& operator=(Model&& other) = default;

private:
    std::shared_ptr<TF_Graph> graph_;
    std::shared_ptr<TF_Session> session_;
};

}    // namespace cppflow


namespace cppflow {

inline Model::Model(const std::string_view& filename,
                    const std::initializer_list<std::string_view>& tags)
    : graph_(TF_NewGraph(), TF_DeleteGraph) {
    auto session_options = SessionOptions {};
    auto run_options = Buffer {};
    auto meta_graph = Buffer {};

    auto session_deleter = [](TF_Session* sess) {
        TF_DeleteSession(sess, context::get_status());
        status_check(context::get_status());
    };

    auto tag_names = std::vector<const char*> {};
    for (const auto& tag : tags) {
        tag_names.emplace_back(tag.data());
    }
    session_.reset(
        TF_LoadSessionFromSavedModel(
            session_options.get(), run_options.get(), filename.data(),
            tag_names.data(), static_cast<int32_t>(tag_names.size()),
            graph_.get(), meta_graph.get(), context::get_status()),
        session_deleter);

    status_check(context::get_status());
}

inline std::vector<std::string> Model::get_operations() const {
    std::vector<std::string> result;
    size_t pos = 0;
    TF_Operation* oper = nullptr;

    // Iterate through the operations of a graph
    while ((oper = TF_GraphNextOperation(graph_.get(), &pos)) != nullptr) {
        result.emplace_back(TF_OperationName(oper));
    }

    return result;
}

inline std::vector<int64_t> Model::get_operation_shape(
    const std::string& operation) const {
    // Get operation by the name
    TF_Output out_op {TF_GraphOperationByName(graph_.get(), operation.c_str()),
                      0};
    // Operation does not exist
    if (!out_op.oper)
        throw std::runtime_error("No operation named \"" + operation
                                 + "\" exists");

    if (operation == "NoOp")
        throw std::runtime_error("NoOp doesn't have a shape");

    // DIMENSIONS

    // Get number of dimensions
    int n_dims
        = TF_GraphGetTensorNumDims(graph_.get(), out_op, context::get_status());

    // If is not a scalar
    auto shape = std::vector<int64_t>(n_dims);
    if (n_dims > 0) {
        // Get dimensions
        TF_GraphGetTensorShape(graph_.get(), out_op, shape.data(), n_dims,
                               context::get_status());
        // Check error on Model Status
        status_check(context::get_status());
    }

    return shape;
}

inline std::tuple<std::string, int> parse_name(const std::string_view& name) {
    auto pos = name.find(':');
    if (pos == std::string::npos) {
        return std::make_tuple(std::string {name}, 0);
    }

    auto prefix = name.substr(0, pos);
    auto suffix = name.substr(pos + 1);
    int32_t index = 0;
    std::from_chars(suffix.begin(), suffix.end(), index, 10);

    return std::make_tuple(std::string {prefix}, index);
}

inline std::vector<Tensor> Model::forward(
    const std::vector<std::tuple<std::string_view, Tensor>>& inputs,
    const std::vector<std::string_view>& outputs) {
    auto input_ops = std::vector<TF_Output> {};
    input_ops.reserve(inputs.size());
    auto input_values = std::vector<TF_Tensor*> {};
    input_values.reserve(inputs.size());

    for (const auto& [name, tensor] : inputs) {
        auto [op_name, op_index] = parse_name(name);
        auto* op = TF_GraphOperationByName(graph_.get(), op_name.c_str());
        if (!op) {
            throw std::runtime_error("No operation named \"" + op_name
                                     + "\" exists");
        }
        input_ops.emplace_back(TF_Output {op, op_index});
        input_values.emplace_back(tensor.get_tensor().get());
    }

    auto output_ops = std::vector<TF_Output> {};
    output_ops.reserve(outputs.size());
    auto output_values = std::vector<TF_Tensor*> {};
    output_values.reserve(outputs.size());
    for (const auto& output : outputs) {
        auto [op_name, op_index] = parse_name(output);
        auto* op = TF_GraphOperationByName(graph_.get(), op_name.c_str());
        if (!op) {
            throw std::runtime_error("No operation named \"" + op_name
                                     + "\" exists");
        }
        output_ops.emplace_back(TF_Output {op, op_index});
        output_values.emplace_back(nullptr);
    }

    TF_SessionRun(session_.get(), nullptr, input_ops.data(),
                  input_values.data(), inputs.size(), output_ops.data(),
                  output_values.data(), outputs.size(), nullptr, 0, nullptr,
                  context::get_status());
    status_check(context::get_status());

    auto output_tensors = std::vector<Tensor> {};
    output_tensors.reserve(output_values.size());
    for (auto* handle : output_values) {
        output_tensors.emplace_back(Tensor {handle});
    }

    return output_tensors;
}

inline Tensor Model::forward(const Tensor& input) {
    return forward({{"serving_default_input_1", input}},
                   {"StatefulPartitionedCall"})[0];
}

}    // namespace cppflow

#endif    // CPPFLOW2_MODEL_H
