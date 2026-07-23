#pragma once

#include <cstdlib>
#include <stdexcept>
#include <string>

namespace infinilm::models::deepseek_v4::utils {

inline const char *env_value(const char *name) {
    return std::getenv(name);
}

inline bool env_flag_enabled(const char *name) {
    const char *value = env_value(name);
    if (value == nullptr) {
        return false;
    }
    const std::string text(value);
    return text == "1" || text == "true" || text == "TRUE" || text == "on" || text == "ON" || text == "kernel" || text == "marlin";
}

inline int env_int_or(const char *name, int fallback) {
    const char *value = env_value(name);
    if (value == nullptr || value[0] == '\0') {
        return fallback;
    }
    return std::stoi(value);
}

inline bool kernel_backend_enabled(const char *name) {
    const char *value = env_value(name);
    if (value == nullptr || value[0] == '\0') {
        return false;
    }
    const std::string text(value);
    if (text == "kernel" || text == "1" || text == "true" || text == "TRUE" || text == "on" || text == "ON") {
        return true;
    }
    if (text == "naive" || text == "0" || text == "false" || text == "FALSE" || text == "off" || text == "OFF") {
        return false;
    }
    throw std::runtime_error(std::string(name) + " must be either naive or kernel");
}

inline bool mhc_kernel_backend_enabled(const char *name) {
    return kernel_backend_enabled(name);
}

} // namespace infinilm::models::deepseek_v4::utils
