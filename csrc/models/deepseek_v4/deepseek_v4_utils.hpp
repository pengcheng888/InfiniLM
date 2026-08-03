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

inline std::string env_string_or(const char *name, const std::string &fallback) {
    const char *value = env_value(name);
    if (value == nullptr || value[0] == 0) {
        return fallback;
    }
    return std::string(value);
}

inline bool debug_dump_enabled() {
    return env_flag_enabled("INFINILM_DSV4_DEBUG_DUMP");
}

inline bool moe_allreduce_outplace_enabled() {
    const std::string backend = env_string_or("INFINILM_DSV4_MOE_ALLREDUCE", "inplace");
    if (backend == "outplace") {
        return true;
    }
    if (backend == "inplace" || backend == "custom" || backend == "dcu_custom" || backend == "custom_ar") {
        return false;
    }
    throw std::runtime_error("INFINILM_DSV4_MOE_ALLREDUCE must be one of inplace, outplace, custom");
}

inline bool moe_custom_allreduce_enabled() {
    const std::string backend = env_string_or("INFINILM_DSV4_MOE_ALLREDUCE", "inplace");
    return backend == "custom" || backend == "dcu_custom" || backend == "custom_ar";
}

} // namespace infinilm::models::deepseek_v4::utils
