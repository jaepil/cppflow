//
// cppflow
//

#ifndef __CPPFLOW2_LIBRARY_H__
#define __CPPFLOW2_LIBRARY_H__

#include "context.h"

#include <tensorflow/c/c_api.h>

#include <filesystem>


namespace cppflow {

class Library {
public:
    explicit Library(const std::filesystem::path& filename)
        : filename_(filename),
          handle_(TF_LoadLibrary(filename.native().c_str(),
                                 cppflow::context::get_status()),
                  TF_DeleteLibraryHandle) {
    }

    Library(const Library& other) = default;
    Library(Library&& other) noexcept = default;
    Library& operator=(const Library& other) = default;
    Library& operator=(Library&& other) noexcept = default;

    const std::filesystem::path& get_filename() const {
        return filename_;
    }

private:
    std::filesystem::path filename_;
    std::shared_ptr<TF_Library> handle_;
};

}    // namespace cppflow

#endif
