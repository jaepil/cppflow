//
// cppflow
//

#ifndef __CPPFLOW_DEFER_H__
#define __CPPFLOW_DEFER_H__

#include <functional>


namespace cppflow {

class defer {
public:
    typedef std::function<void()> Func;

    explicit defer(const Func& func) : _func(func) {
    }
    ~defer() {
        _func();
    }

    defer(const defer&) = delete;
    defer(defer&&) = delete;
    defer& operator=(const defer&) = delete;
    void* operator new(size_t) = delete;
    void operator delete(void*) = delete;

private:
    Func _func;
};

}    // namespace cppflow

#endif
