#include "infer_engine.hpp"
#include "../config/config_factory.hpp"
#include "spdlog/spdlog.h"
#include <algorithm>
#include <cstdint>
#include <future>
#include <limits>
#include <stdexcept>
#include <unordered_set>

namespace infinilm::engine {
namespace {

size_t max_length_from_offsets(
    const std::optional<infinicore::Tensor> &offsets,
    const char *name) {
    if (!offsets.has_value()) {
        return 0;
    }

    auto cpu_offsets = offsets.value();
    if (cpu_offsets->device().getType() != infinicore::Device::Type::CPU) {
        cpu_offsets = cpu_offsets->to(infinicore::Device::cpu());
        infinicore::context::syncStream();
    }

    if (cpu_offsets->dtype() != infinicore::DataType::I32
        || cpu_offsets->shape().size() != 1
        || cpu_offsets->shape()[0] < 2) {
        throw std::invalid_argument(
            std::string(name) + " must be a one-dimensional int32 tensor with at least two entries");
    }

    const auto *values = reinterpret_cast<const int32_t *>(cpu_offsets->data());
    size_t max_length = 0;
    for (size_t i = 1; i < cpu_offsets->shape()[0]; ++i) {
        if (values[i] < values[i - 1]) {
            throw std::invalid_argument(std::string(name) + " must be nondecreasing");
        }
        max_length = std::max(
            max_length,
            static_cast<size_t>(values[i] - values[i - 1]));
    }
    return max_length;
}

} // namespace

//------------------------------------------------------
// Constructor
//------------------------------------------------------
InferEngine::InferEngine(
    const std::string &config_str,
    const distributed::DistConfig &distributed_config,
    infinicore::Device::Type device_type,
    const cache::CacheConfig *cache_config,
    bool enable_graph_compiling,
    backends::AttentionBackend attention_backend,
    std::optional<infinicore::DataType> kv_cache_dtype,
    bool use_mla,
    const std::string &weight_load_mode,
    bool pre_transpose,
    bool enable_workspace_manager,
    size_t max_num_batched_tokens,
    double gpu_memory_utilization)
    : communication_group_(distributed_config, device_type),
      attention_backend_(attention_backend),
      weight_load_mode_(weight_load_mode),
      use_mla_(use_mla),
      enable_workspace_manager_(enable_workspace_manager),
      max_num_batched_tokens_(max_num_batched_tokens),
      gpu_memory_utilization_(gpu_memory_utilization),
      enable_graph_compiling_(enable_graph_compiling) {
    if (weight_load_mode_ != "async" && weight_load_mode_ != "sync") {
        throw std::invalid_argument("weight_load_mode must be either 'async' or 'sync'");
    }
    if (enable_workspace_manager && max_num_batched_tokens == 0) {
        throw std::invalid_argument(
            "enable_workspace_manager requires max_num_batched_tokens > 0");
    }
    if (!(gpu_memory_utilization_ > 0.0 && gpu_memory_utilization_ <= 1.0)) {
        throw std::invalid_argument("gpu_memory_utilization must be in (0, 1]");
    }
    if (enable_workspace_manager_) {
        spdlog::info("Workspace manager enabled, max_num_batched_tokens={}",
                     max_num_batched_tokens_);
    }
    if (cache_config != nullptr) {
        cache_config_ = cache_config->unique_copy();
    }

    // Load model config if model_path is provided, model_path must be valid, and config.json exists
    this->model_config_ = infinilm::config::ConfigFactory::createConfig(config_str);
    auto infinilm_config = std::make_shared<infinilm::global_state::InfinilmConfig>(
        attention_backend,
        this->model_config_,
        use_mla,
        distributed_config.moe_ep_backend,
        distributed_config.moe_ep_size,
        pre_transpose,
        enable_workspace_manager,
        max_num_batched_tokens);

    // Only support offline int8 kv cache quantization in this version
    if (kv_cache_dtype.has_value()) {
        this->model_config_->set_kv_quant_scheme(kv_cache_dtype.value());
    }
    // Create one RankWorker per rank
    int world_size = communication_group_.get_world_size();
    barrier_ = std::make_unique<RankBarrier>((size_t)world_size);
    workers_.reserve(world_size);
    for (int r = 0; r < world_size; ++r) {
        workers_.emplace_back(std::make_unique<RankWorker>(
            infinilm_config,
            communication_group_.get_rank_info(r),
            cache_config_ != nullptr ? cache_config_.get() : nullptr,
            barrier_.get(),
            enable_graph_compiling,
            attention_backend_));
    }

    for (auto &worker : workers_) {
        worker->wait_for_init();
    }
    // Graphs must be compiled after weights are loaded and post-processed.
    // Quantized models may replace their linear implementations during
    // process_weights_after_loading(), so compiling here would capture stale
    // fallback operators.
}

//------------------------------------------------------
// load_param
//------------------------------------------------------
void InferEngine::load_param(const std::string &name, const infinicore::Tensor &param) {
    // Load the parameter on all workers
    for (auto &worker : workers_) {
        worker->load_param(name, param);
    }
}

void InferEngine::load_params(const std::unordered_map<std::string, infinicore::Tensor> &params, bool strict) {
    if (workers_.size() <= 1 || weight_load_mode_ == "sync") {
        for (auto &worker : workers_) {
            worker->load_params(params, strict);
        }
        return;
    }

    std::vector<std::future<void>> futures;
    futures.reserve(workers_.size());
    for (auto &worker : workers_) {
        futures.emplace_back(std::async(std::launch::async, [&worker, &params, strict] {
            worker->load_params(params, strict);
        }));
    }
    for (auto &future : futures) {
        future.get();
    }
}

//------------------------------------------------------
// load_param
//------------------------------------------------------
void InferEngine::process_weights_after_loading() {
    // Process the weights after loading on all workers
    for (auto &worker : workers_) {
        worker->process_weights_after_loading();
    }
    weights_finalized_ = true;

    // auto 模式必须在正式 graph compile 前完成 profiling 和 KV resize；
    // 否则 graph 会捕获 probe cache 的地址，之后重分配将导致地址失效。
    resize_cache_for_gpu_memory_utilization();
}

std::vector<RankWorker::DeviceMemoryInfo> InferEngine::install_cache_and_compile(
    size_t num_blocks) {
    const auto *current_config = dynamic_cast<const cache::PagedKVCacheConfig *>(
        cache_config_.get());
    if (current_config == nullptr) {
        throw std::runtime_error("gpu memory auto-sizing requires paged KV cache");
    }

    // 先释放旧 graph，避免 reset cache 后旧 graph 继续引用已释放的 KV 地址。
    for (auto &worker : workers_) {
        worker->clear_compiled_graphs();
    }

    cache::PagedKVCacheConfig new_config(
        num_blocks,
        current_config->block_size(),
        current_config->max_batch_size());
    for (auto &worker : workers_) {
        worker->reset_cache(&new_config);
    }
    // reset_cache 是异步 job；必须等所有 rank 完成重分配后再 compile，
    // 与原有 reset_cache() 的语义保持一致。
    for (auto &worker : workers_) {
        worker->wait();
    }
    cache_config_ = new_config.unique_copy();

    std::vector<RankWorker::DeviceMemoryInfo> after_reset_memory;
    after_reset_memory.reserve(workers_.size());
    for (auto &worker : workers_) {
        after_reset_memory.push_back(worker->get_memory_info());
    }
    for (size_t rank = 0; rank < after_reset_memory.size(); ++rank) {
        spdlog::info(
            "GPU memory auto-sizing installed rank={} num_blocks={} used={}",
            rank,
            num_blocks,
            after_reset_memory[rank].used_bytes);
    }

    // compile 会跨 rank barrier；先向所有 worker 提交，再统一等待。
    for (auto &worker : workers_) {
        worker->compile();
    }
    for (auto &worker : workers_) {
        worker->wait();
    }

    std::vector<RankWorker::DeviceMemoryInfo> memory_info;
    memory_info.reserve(workers_.size());
    for (auto &worker : workers_) {
        memory_info.push_back(worker->get_memory_info());
    }
    return memory_info;
}

void InferEngine::resize_cache_for_gpu_memory_utilization() {
    const auto *probe_config = dynamic_cast<const cache::PagedKVCacheConfig *>(
        cache_config_.get());
    if (probe_config == nullptr) {
        throw std::runtime_error("gpu_memory_utilization requires paged KV cache");
    }
    if (probe_config->num_blocks() == 0) {
        throw std::runtime_error("gpu_memory_utilization requires non-zero probe blocks");
    }

    // 全量 profiling compile 与正常 compile 一样会跨 rank barrier，因此必须
    // 先向所有 worker 提交任务，再逐个等待结果。
    for (auto &worker : workers_) {
        worker->start_profile_compile();
    }
    std::vector<RankWorker::GraphCompileProfile> profiles;
    profiles.reserve(workers_.size());
    for (auto &worker : workers_) {
        profiles.push_back(worker->wait_profile_compile());
    }

    const size_t probe_blocks = probe_config->num_blocks();
    const double utilization = gpu_memory_utilization_;
    size_t candidate_blocks = std::numeric_limits<size_t>::max();

    for (size_t rank = 0; rank < profiles.size(); ++rank) {
        const auto &profile = profiles[rank];
        if (profile.kv_cache_bytes == 0 || profile.max_decode_batch_size == 0) {
            throw std::runtime_error("invalid KV cache profiling result");
        }

        const size_t kv_bytes_per_block = profile.kv_cache_bytes / probe_blocks;
        const size_t block_table_bytes_per_block =
            profile.max_decode_batch_size * sizeof(int32_t);
        const size_t per_block_bytes =
            kv_bytes_per_block + block_table_bytes_per_block;
        if (per_block_bytes == 0) {
            throw std::runtime_error("invalid per-block memory estimate");
        }

        // profiling graph 的增量中包含 probe block table；扣除这部分后，
        // 其余 graph exec/input 开销在候选 num_blocks 变化时近似固定。
        const size_t probe_block_table_bytes =
            probe_blocks * block_table_bytes_per_block;
        const size_t graph_delta = profile.after.used_bytes > profile.before.used_bytes
                                       ? profile.after.used_bytes - profile.before.used_bytes
                                       : 0;
        const size_t graph_fixed_bytes = graph_delta > probe_block_table_bytes
                                             ? graph_delta - probe_block_table_bytes
                                             : 0;

        // cleared 快照已释放 graph，但保留权重、workspace、probe KV 和外部占用。
        // 从中扣除 probe KV，得到最终分配 KV 前的非 KV 基础占用。
        const size_t non_kv_base_bytes = profile.cleared.used_bytes >
                                                 profile.kv_cache_bytes
                                             ? profile.cleared.used_bytes -
                                                   profile.kv_cache_bytes
                                             : 0;
        const size_t target_bytes = static_cast<size_t>(
            static_cast<double>(profile.cleared.total_bytes) * utilization);
        const size_t safety_reserve_bytes = std::max<size_t>(
            512ULL * 1024 * 1024,
            profile.cleared.total_bytes / 50);

        size_t available_bytes = 0;
        if (target_bytes > non_kv_base_bytes + graph_fixed_bytes + safety_reserve_bytes) {
            available_bytes = target_bytes - non_kv_base_bytes -
                              graph_fixed_bytes - safety_reserve_bytes;
        }
        const size_t rank_blocks = available_bytes / per_block_bytes;
        candidate_blocks = std::min(candidate_blocks, rank_blocks);

        spdlog::info(
            "GPU memory auto-sizing rank={}: total={} used_before_profile={} "
            "graph_delta={} graph_fixed={} kv_per_block={} block_table_per_block={} "
            "candidate_blocks={}",
            rank,
            profile.cleared.total_bytes,
            profile.before.used_bytes,
            graph_delta,
            graph_fixed_bytes,
            kv_bytes_per_block,
            block_table_bytes_per_block,
            rank_blocks);
    }

    if (candidate_blocks < 2) {
        throw std::runtime_error(
            "gpu_memory_utilization cannot allocate at least 2 KV blocks");
    }

    constexpr size_t max_attempts = 3;
    for (size_t attempt = 0; attempt < max_attempts; ++attempt) {
        const auto memory_info = install_cache_and_compile(candidate_blocks);
        bool exceeds_target = false;
        size_t max_excess_bytes = 0;
        size_t per_block_bytes = 1;

        for (size_t rank = 0; rank < memory_info.size(); ++rank) {
            const size_t target_bytes = static_cast<size_t>(
                static_cast<double>(memory_info[rank].total_bytes) * utilization);
            if (memory_info[rank].used_bytes > target_bytes) {
                exceeds_target = true;
                max_excess_bytes = std::max(
                    max_excess_bytes,
                    memory_info[rank].used_bytes - target_bytes);
            }
        }

        if (!exceeds_target) {
            spdlog::info(
                "GPU memory auto-sizing selected num_blocks={} utilization={}",
                candidate_blocks,
                utilization);
            return;
        }

        // profiling 公式可能低估少量 graph/allocator 开销。compile 后复查若超
        // 目标，则按实际超出量降低 blocks，并加一个 block 的安全余量。
        const auto &profile = profiles.front();
        const size_t kv_bytes_per_block = profile.kv_cache_bytes / probe_blocks;
        const size_t block_table_bytes_per_block =
            profile.max_decode_batch_size * sizeof(int32_t);
        per_block_bytes = kv_bytes_per_block + block_table_bytes_per_block;
        const size_t blocks_to_drop =
            (max_excess_bytes + per_block_bytes - 1) / per_block_bytes + 1;
        if (candidate_blocks <= blocks_to_drop) {
            break;
        }
        candidate_blocks -= blocks_to_drop;
        spdlog::warn(
            "GPU memory target exceeded after compile; retry num_blocks={}",
            candidate_blocks);
    }

    throw std::runtime_error(
        "gpu_memory_utilization could not satisfy target after retries");
}

//------------------------------------------------------
// state_dict
//------------------------------------------------------
std::vector<std::unordered_map<std::string, infinicore::nn::Parameter>> InferEngine::state_dict() {
    std::vector<std::unordered_map<std::string, infinicore::nn::Parameter>> results;
    if (0 == workers_.size()) {
        throw std::runtime_error(" Model object not found. ");
    }

    for (auto &worker : workers_) {
        results.push_back(worker->state_dict());
    }
    return results;
}

std::vector<std::string> InferEngine::state_dict_keys() {
    if (0 == workers_.size()) {
        throw std::runtime_error(" Model object not found. ");
    }
    std::vector<std::string> ordered_keys;
    std::unordered_set<std::string> seen_keys;
    for (auto &worker : workers_) {
        for (const auto &key : worker->state_dict_keys()) {
            // Preserve first-seen worker order while removing duplicate TP keys.
            if (seen_keys.emplace(key).second) {
                ordered_keys.push_back(key);
            }
        }
    }
    return ordered_keys;
}

//------------------------------------------------------
// forward
//------------------------------------------------------
infinilm::InfinilmModel::Input
InferEngine::Input::to_model_input(infinicore::Device device) const {

    auto to_device = [&](const std::optional<infinicore::Tensor> &t)
        -> std::optional<infinicore::Tensor> {
        return t.has_value() ? t.value()->to(device) : t;
    };
    auto to_device_vec = [&](const std::optional<std::vector<infinicore::Tensor>> &vec)
        -> std::optional<std::vector<infinicore::Tensor>> {
        if (!vec.has_value()) {
            return vec;
        }
        std::vector<infinicore::Tensor> result;
        result.reserve(vec->size());
        for (const auto &t : vec.value()) {
            result.push_back(t->to(device));
        }
        return result;
    };

    const bool is_prefill = input_ids.has_value()
                         && total_sequence_lengths.has_value()
                         && input_ids.value()->numel()
                                != total_sequence_lengths.value()->numel();
    const size_t max_query_length = is_prefill ? max_length_from_offsets(input_offsets, "input_offsets") : 0;
    const size_t max_sequence_length = is_prefill ? max_length_from_offsets(cu_seqlens, "cu_seqlens") : 0;

    infinilm::InfinilmModel::Input input = {
        to_device(input_ids), // @todo: on device in the future
        to_device(position_ids),
        to_device(past_sequence_lengths), // @todo: on device in the future
        to_device(total_sequence_lengths),
        to_device(input_offsets),
        to_device(cu_seqlens),
        to_device(block_tables),
        to_device(slot_mapping),
        to_device(mamba_init_state_indices),
        to_device(mamba_final_state_indices),
        to_device_vec(pixel_values),
        to_device_vec(image_bound),
        to_device_vec(tgt_sizes),
        to_device_vec(image_grid_thw),
        image_req_ids,
        visual_token_ranges,
        to_device(target_hidden_states),
        sample_all_positions};

    infinilm::global_state::get_forward_context().attn_metadata = {
        input.past_sequence_lengths,
        input.total_sequence_lengths,
        input.input_offsets,
        input.cu_seqlens,
        input.block_tables,
        input.slot_mapping,
        max_query_length,
        max_sequence_length};

    infinilm::global_state::get_forward_context().mamba_metadata = {
        input.input_offsets,
        input.mamba_init_state_indices,
        input.mamba_final_state_indices};

    global_state::get_forward_context().mm_metadata = {
        image_req_ids,
        visual_token_ranges};

    return input;
}

InferEngine::Output InferEngine::forward(const InferEngine::Input &input) {
    if (enable_workspace_manager_) {
        size_t num_input_tokens = 0;
        if (input.input_ids.has_value()) {
            num_input_tokens = input.input_ids.value()->numel();
        } else if (input.target_hidden_states.has_value()) {
            const auto &hidden_states = input.target_hidden_states.value();
            if (hidden_states->ndim() > 0 && hidden_states->shape().back() > 0) {
                num_input_tokens = hidden_states->numel() / hidden_states->shape().back();
            }
        }
        if (num_input_tokens > max_num_batched_tokens_) {
            throw std::invalid_argument("Input token count " + std::to_string(num_input_tokens)
                                        + " exceeds workspace capacity " + std::to_string(max_num_batched_tokens_));
        }
    }
    // Trigger each worker to run inference
    for (auto &worker : workers_) {
        worker->run(input);
    }
    // Wait for all workers
    for (auto &worker : workers_) {
        worker->wait();
    }

    return workers_[0]->get_output();
}

void InferEngine::compile() {
    if (!weights_finalized_) {
        return;
    }
    for (auto &worker : workers_) {
        worker->compile();
    }
    // Wait for all workers
    for (auto &worker : workers_) {
        worker->wait();
    }
}

//------------------------------------------------------
// Destructor
//------------------------------------------------------
InferEngine::~InferEngine() {
    // Close all workers
    for (auto &worker : workers_) {
        worker->close();
    }
}

const distributed::DistConfig &InferEngine::get_dist_config() const {
    return communication_group_.get_dist_config();
}

//------------------------------------------------------
// reset_cache (overloaded with CacheConfig)
//------------------------------------------------------
void InferEngine::reset_cache(const cache::CacheConfig *new_config) {
    for (auto &worker : workers_) {
        worker->reset_cache(new_config);
    }
    for (auto &worker : workers_) {
        worker->wait();
    }
    cache_config_ = new_config->unique_copy();
    this->compile();
}

std::vector<std::vector<infinicore::Tensor>> InferEngine::get_kv_cache() {
    std::vector<std::vector<infinicore::Tensor>> kv_cache_list;
    if (workers_.empty()) {
        throw std::runtime_error("InferEngine::get_cache_vec: no workers");
    }

    kv_cache_list.reserve(workers_.size());
    for (auto &worker : workers_) {
        kv_cache_list.push_back(std::move(worker->get_kv_cache()));
    }

    for (auto &worker : workers_) {
        worker->wait();
    }

    return kv_cache_list;
}

} // namespace infinilm::engine
