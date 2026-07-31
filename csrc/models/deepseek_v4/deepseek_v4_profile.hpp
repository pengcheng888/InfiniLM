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
    CausalForward,
    CausalModel,
    CausalLmHead,
    CausalLogitsView,
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
    AttentionForward,
    AttentionQProjection,
    AttentionQProjA,
    AttentionQNorm,
    AttentionQProjB,
    AttentionQRmsNormSelf,
    AttentionKVProjection,
    AttentionKVProj,
    AttentionKVNorm,
    AttentionMetadata,
    AttentionRope,
    AttentionSWAStore,
    AttentionC4Compress,
    AttentionC4SparseAlloc,
    AttentionC4IndexerCompress,
    AttentionC4IndexerQuery,
    AttentionC4IndexerSparse,
    AttentionC128Compress,
    AttentionWorkspace,
    AttentionFlashMLASchedule,
    AttentionFlashMLAWorkspace,
    AttentionFlashMLAOutWorkspaceCall,
    AttentionFlashMLAWithMetadataCall,
    AttentionFlashMLACacheMetadata,
    AttentionFlashMLA,
    AttentionOutRope,
    AttentionWoA,
    AttentionWoB,
    Count,
};

enum class Phase : size_t {
    Overall = 0,
    Prefill,
    Decode,
    Count,
};

enum class LayerType : size_t {
    Unscoped = 0,
    NoCompress,
    Compress4,
    Compress128,
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

inline std::array<std::array<std::array<Stat, static_cast<size_t>(Event::Count)>, static_cast<size_t>(Phase::Count)>, static_cast<size_t>(LayerType::Count)> &layer_type_stats() {
    static std::array<std::array<std::array<Stat, static_cast<size_t>(Event::Count)>, static_cast<size_t>(Phase::Count)>, static_cast<size_t>(LayerType::Count)> value;
    return value;
}

inline LayerType &current_layer_type() {
    static thread_local LayerType value = LayerType::Unscoped;
    return value;
}

inline const char *event_name(Event event) {
    switch (event) {
    case Event::DecoderLayer:
        return "decoder.layer";
    case Event::CausalForward:
        return "causal.forward";
    case Event::CausalModel:
        return "causal.model";
    case Event::CausalLmHead:
        return "causal.lm_head";
    case Event::CausalLogitsView:
        return "causal.logits_view";
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
    case Event::AttentionForward:
        return "attention.forward";
    case Event::AttentionQProjection:
        return "attention.q_projection";
    case Event::AttentionQProjA:
        return "attention.q_proj_a";
    case Event::AttentionQNorm:
        return "attention.q_norm";
    case Event::AttentionQProjB:
        return "attention.q_proj_b";
    case Event::AttentionQRmsNormSelf:
        return "attention.q_rmsnorm_self";
    case Event::AttentionKVProjection:
        return "attention.kv_projection";
    case Event::AttentionKVProj:
        return "attention.kv_proj";
    case Event::AttentionKVNorm:
        return "attention.kv_norm";
    case Event::AttentionMetadata:
        return "attention.metadata";
    case Event::AttentionRope:
        return "attention.rope";
    case Event::AttentionSWAStore:
        return "attention.swa_store";
    case Event::AttentionC4Compress:
        return "attention.c4_compress";
    case Event::AttentionC4SparseAlloc:
        return "attention.c4_sparse_alloc";
    case Event::AttentionC4IndexerCompress:
        return "attention.c4_indexer.compress";
    case Event::AttentionC4IndexerQuery:
        return "attention.c4_indexer.query";
    case Event::AttentionC4IndexerSparse:
        return "attention.c4_indexer.sparse";
    case Event::AttentionC128Compress:
        return "attention.c128_compress";
    case Event::AttentionWorkspace:
        return "attention.workspace";
    case Event::AttentionFlashMLASchedule:
        return "attention.flashmla_schedule";
    case Event::AttentionFlashMLAWorkspace:
        return "attention.flashmla_workspace";
    case Event::AttentionFlashMLAOutWorkspaceCall:
        return "attention.flashmla_out_workspace_call";
    case Event::AttentionFlashMLAWithMetadataCall:
        return "attention.flashmla_with_metadata_call";
    case Event::AttentionFlashMLACacheMetadata:
        return "attention.flashmla_cache_metadata";
    case Event::AttentionFlashMLA:
        return "attention.flashmla";
    case Event::AttentionOutRope:
        return "attention.out_rope";
    case Event::AttentionWoA:
        return "attention.wo_a";
    case Event::AttentionWoB:
        return "attention.wo_b";
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

inline const char *layer_type_name(LayerType layer_type) {
    switch (layer_type) {
    case LayerType::Unscoped:
        return "unscoped";
    case LayerType::NoCompress:
        return "no_compress";
    case LayerType::Compress4:
        return "compress_ratio_4";
    case LayerType::Compress128:
        return "compress_ratio_128";
    case LayerType::Count:
        break;
    }
    return "unknown";
}

inline LayerType layer_type_from_compress_ratio(size_t compress_ratio) {
    if (compress_ratio == 0) {
        return LayerType::NoCompress;
    }
    if (compress_ratio == 4) {
        return LayerType::Compress4;
    }
    if (compress_ratio == 128) {
        return LayerType::Compress128;
    }
    return LayerType::Unscoped;
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

inline void add_layer_type_stat(LayerType layer_type, Phase phase, Event event, unsigned long long micros) {
    auto &stat = layer_type_stats()[static_cast<size_t>(layer_type)][static_cast<size_t>(phase)][static_cast<size_t>(event)];
    stat.calls.fetch_add(1, std::memory_order_relaxed);
    stat.micros.fetch_add(micros, std::memory_order_relaxed);
}

inline bool has_phase_stats(Phase phase) {
    for (size_t i = 0; i < static_cast<size_t>(Event::Count); ++i) {
        auto &stat = stats()[static_cast<size_t>(phase)][i];
        if (stat.calls.load(std::memory_order_relaxed) != 0) {
            return true;
        }
    }
    return false;
}

inline bool has_layer_type_phase_stats(LayerType layer_type, Phase phase) {
    for (size_t i = 0; i < static_cast<size_t>(Event::Count); ++i) {
        auto &stat = layer_type_stats()[static_cast<size_t>(layer_type)][static_cast<size_t>(phase)][i];
        if (stat.calls.load(std::memory_order_relaxed) != 0) {
            return true;
        }
    }
    return false;
}

inline void dump_phase(Phase phase) {
    if (!has_phase_stats(phase)) {
        return;
    }
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

inline void dump_layer_type_phase(LayerType layer_type, Phase phase) {
    if (!has_layer_type_phase_stats(layer_type, phase)) {
        return;
    }
    std::fprintf(stderr, "[INFINILM_DSV4_PROFILE] layer_type=%s phase=%s\n", layer_type_name(layer_type), phase_name(phase));
    for (size_t i = 0; i < static_cast<size_t>(Event::Count); ++i) {
        auto &stat = layer_type_stats()[static_cast<size_t>(layer_type)][static_cast<size_t>(phase)][i];
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

inline void dump_layer_type(LayerType layer_type) {
    bool has_stats = false;
    for (size_t i = 0; i < static_cast<size_t>(Phase::Count); ++i) {
        has_stats = has_stats || has_layer_type_phase_stats(layer_type, static_cast<Phase>(i));
    }
    if (!has_stats) {
        return;
    }
    std::fprintf(stderr, "[INFINILM_DSV4_PROFILE] layer_type=%s\n", layer_type_name(layer_type));
    dump_layer_type_phase(layer_type, Phase::Overall);
    dump_layer_type_phase(layer_type, Phase::Prefill);
    dump_layer_type_phase(layer_type, Phase::Decode);
}

inline void dump() {
    if (!enabled()) {
        return;
    }
    std::fprintf(stderr, "\n[INFINILM_DSV4_PROFILE] GPU-synced wall time\n");
    dump_phase(Phase::Overall);
    dump_phase(Phase::Prefill);
    dump_phase(Phase::Decode);
    for (size_t i = 1; i < static_cast<size_t>(LayerType::Count); ++i) {
        dump_layer_type(static_cast<LayerType>(i));
    }
}

inline void register_dump_once() {
    static std::once_flag flag;
    std::call_once(flag, [] { std::atexit(dump); });
}

class ScopedLayerContext {
public:
    explicit ScopedLayerContext(LayerType layer_type)
        : previous_(current_layer_type()) {
        current_layer_type() = layer_type;
    }

    ~ScopedLayerContext() {
        current_layer_type() = previous_;
    }

private:
    LayerType previous_;
};

class ScopedTimer {
public:
    explicit ScopedTimer(Event event, size_t token_count = 0)
        : event_(event), phase_(phase_from_token_count(token_count)), layer_type_(current_layer_type()), active_(enabled()) {
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
        if (layer_type_ != LayerType::Unscoped) {
            add_layer_type_stat(layer_type_, Phase::Overall, event_, micros);
            if (phase_ != Phase::Overall) {
                add_layer_type_stat(layer_type_, phase_, event_, micros);
            }
        }
    }

private:
    using Clock = std::chrono::steady_clock;
    Event event_;
    Phase phase_;
    LayerType layer_type_;
    bool active_{false};
    Clock::time_point start_{};
};

} // namespace infinilm::models::deepseek_v4::profile
