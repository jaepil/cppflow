//
// cppflow
//

#ifndef __CPPFLOW2_SESSION_H__
#define __CPPFLOW2_SESSION_H__

#include <tensorflow/c/c_api.h>

#include <memory>


namespace cppflow {

class SessionOptions {
public:
    SessionOptions() : impl_(TF_NewSessionOptions(), TF_DeleteSessionOptions) {
    }
    explicit SessionOptions(TF_SessionOptions* handle)
        : impl_(handle, TF_DeleteSessionOptions) {
    }

    SessionOptions(const SessionOptions& other) = default;
    SessionOptions(SessionOptions&& other) noexcept = default;
    SessionOptions& operator=(const SessionOptions& other) = default;
    SessionOptions& operator=(SessionOptions&& other) noexcept = default;

    const TF_SessionOptions* get() const {
        return impl_.get();
    }

    TF_SessionOptions* get() {
        return impl_.get();
    }

private:
    std::shared_ptr<TF_SessionOptions> impl_;
};

}    // namespace cppflow

#endif
