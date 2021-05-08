//
// Created by serizba on 17/9/20.
//

#ifndef __CPPFLOW_CPPFLOW_H__
#define __CPPFLOW_CPPFLOW_H__

#include "buffer.h"
#include "context.h"
#include "datatype.h"
#include "library.h"
#include "model.h"
#include "ops.h"
#include "raw_ops.h"
#include "session_options.h"
#include "tensor.h"

#include <tensorflow/c/c_api.h>


namespace cppflow {

/**
 * Version of TensorFlow and CppFlow
 * @return A string containing the version of TensorFow and CppFlow
 */
inline std::string version() {
    return "TensorFlow: " + std::string {TF_Version()} + " CppFlow: 2.0.0";
}

}    // namespace cppflow

#endif
