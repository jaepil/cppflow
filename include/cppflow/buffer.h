//
// cppflow
//

#ifndef __CPPFLOW2_BUFFER_H__
#define __CPPFLOW2_BUFFER_H__

#include <tensorflow/c/c_api.h>

#include <memory>
#include <string_view>


namespace cppflow {

class Buffer {
public:
    Buffer() : impl_(TF_NewBuffer(), TF_DeleteBuffer) {
    }
    explicit Buffer(TF_Buffer* handle) : impl_(handle, TF_DeleteBuffer) {
    }
    explicit Buffer(const char* data, size_t size)
        : impl_(TF_NewBufferFromString(data, size), TF_DeleteBuffer) {
    }
    explicit Buffer(const std::string_view& data)
        : Buffer(data.data(), data.size()) {
    }

    Buffer(const Buffer& other) = default;
    Buffer(Buffer&& other) noexcept = default;
    Buffer& operator=(const Buffer& other) = default;
    Buffer& operator=(Buffer&& other) noexcept = default;

    const TF_Buffer* get() const {
        return impl_.get();
    }

    TF_Buffer* get() {
        return impl_.get();
    }

private:
    std::shared_ptr<TF_Buffer> impl_;
};

}    // namespace cppflow

#endif
