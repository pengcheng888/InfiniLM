#pragma once

#include "infinicore/context/context.hpp"

#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <mutex>

namespace infinilm::models::deepseek_v4::profile {

enum class Event : size_t {
    DecoderLayer = 0,
    DecoderAttnHcPre,
    DecoderAttnNorm,
    DecoderAttnHcPost,
    DecoderFfnHcPre,
    DecoderFfnNorm,
    DecoderMoe,
    DecoderFfnHcPost,
    MoeForward,
    MoeGate,
    MoeTopk,
    MoeExperts,
    MoeExpertsPrepare,
    MoeExpertsPrepareAlign,
    MoeExpertsPrepareQuant,
    MoeExpertsContiguous,
    MoeExpertsFusedCall,
    MoeSharedExperts,
    MoeAddShared,
    MoeAllReduce,
    Count,
};

enum class Phase : size_t {
    Overall = 0,
    Prefill,
    Decode,
    Count,
};

struct Stat {
    std::atomic<unsigned long long> calls{0};
    std::atomic<unsigned long long> micros{0};
};

inline std::array<std::array<Stat, static_cast<size_t>(Event::Count)>, static_cast<size_t>(Phase::Count)> &stats() {
    static std::array<std::array<Stat, static_cast<size_t>(Event::Count)>, static_cast<size_t>(Phase::Count)> value;
    return value;
}

inline const char *event_name(Event event) {
    switch (event) {
    case Event::DecoderLayer:
        return "decoder.layer";
    case Event::DecoderAttnHcPre:
        return "decoder.attn_hc_pre";
    case Event::DecoderAttnNorm:
        return "decoder.attn_norm";
    case Event::DecoderAttnHcPost:
        return "decoder.attn_hc_post";
    case Event::DecoderFfnHcPre:
        return "decoder.ffn_hc_pre";
    case Event::DecoderFfnNorm:
        return "decoder.ffn_norm";
    case Event::DecoderMoe:
        return "decoder.moe";
    case Event::DecoderFfnHcPost:
        return "decoder.ffn_hc_post";
    case Event::MoeForward:
        return "moe.forward";
    case Event::MoeGate:
        return "moe.gate";
    case Event::MoeTopk:
        return "moe.topk";
    case Event::MoeExperts:
        return "moe.experts";
    case Event::MoeExpertsPrepare:
        return "moe.experts.prepare";
    case Event::MoeExpertsPrepareAlign:
        return "moe.experts.prepare_align";
    case Event::MoeExpertsPrepareQuant:
        return "moe.experts.prepare_quant";
    case Event::MoeExpertsContiguous:
        return "moe.experts.contiguous";
    case Event::MoeExpertsFusedCall:
        return "moe.experts.fused_call";
    case Event::MoeSharedExperts:
        return "moe.shared_experts";
    case Event::MoeAddShared:
        return "moe.add_shared";
    case Event::MoeAllReduce:
        return "moe.allreduce";
    case Event::Count:
        break;
    }
    return "unknown";
}

inline const char *phase_name(Phase phase) {
    switch (phase) {
    case Phase::Overall:
        return "overall";
    case Phase::Prefill:
        return "prefill";
    case Phase::Decode:
        return "decode";
    case Phase::Count:
        break;
    }
    return "unknown";
}

inline bool env_enabled(const char *name) {
    const char *env = std::getenv(name);
    return env != nullptr && env[0] != '\0' && env[0] != '0';
}

inline bool enabled() {
    static const bool value = env_enabled("INFINILM_DSV4_PROFILE") || env_enabled("INFINILM_DSV4_FFN_PROFILE");
    return value;
}

inline Phase phase_from_token_count(size_t token_count) {
    if (token_count == 1) {
        return Phase::Decode;
    }
    if (token_count > 1) {
        return Phase::Prefill;
    }
    return Phase::Overall;
}

inline void add_stat(Phase phase, Event event, unsigned long long micros) {
    auto &stat = stats()[static_cast<size_t>(phase)][static_cast<size_t>(event)];
    stat.calls.fetch_add(1, std::memory_order_relaxed);
    stat.micros.fetch_add(micros, std::memory_order_relaxed);
}

inline void dump_phase(Phase phase) {
    std::fprintf(stderr, "[INFINILM_DSV4_PROFILE] phase=%s\n", phase_name(phase));
    for (size_t i = 0; i < static_cast<size_t>(Event::Count); ++i) {
        auto &stat = stats()[static_cast<size_t>(phase)][i];
        const auto calls = stat.calls.load(std::memory_order_relaxed);
        const auto micros = stat.micros.load(std::memory_order_relaxed);
        if (calls == 0) {
            continue;
        }
        const double total_ms = static_cast<double>(micros) / 1000.0;
        const double avg_ms = total_ms / static_cast<double>(calls);
        std::fprintf(stderr,
                     "[INFINILM_DSV4_PROFILE] %-24s calls=%llu total_ms=%.3f avg_ms=%.6f\n",
                     event_name(static_cast<Event>(i)),
                     calls,
                     total_ms,
                     avg_ms);
    }
}

inline void dump() {
    if (!enabled()) {
        return;
    }
    std::fprintf(stderr, "\n[INFINILM_DSV4_PROFILE] GPU-synced wall time\n");
    dump_phase(Phase::Overall);
    dump_phase(Phase::Prefill);
    dump_phase(Phase::Decode);
}

inline void register_dump_once() {
    static std::once_flag flag;
    std::call_once(flag, [] { std::atexit(dump); });
}

class ScopedTimer {
public:
    explicit ScopedTimer(Event event, size_t token_count = 0)
        : event_(event), phase_(phase_from_token_count(token_count)), active_(enabled()) {
        if (active_) {
            register_dump_once();
            infinicore::context::syncStream();
            start_ = Clock::now();
        }
    }

    ~ScopedTimer() {
        if (!active_) {
            return;
        }
        infinicore::context::syncStream();
        const auto end = Clock::now();
        const auto us = std::chrono::duration_cast<std::chrono::microseconds>(end - start_).count();
        const auto micros = static_cast<unsigned long long>(us);
        add_stat(Phase::Overall, event_, micros);
        if (phase_ != Phase::Overall) {
            add_stat(phase_, event_, micros);
        }
    }

private:
    using Clock = std::chrono::steady_clock;
    Event event_;
    Phase phase_;
    bool active_{false};
    Clock::time_point start_{};
};

} // namespace infinilm::models::deepseek_v4::profile
