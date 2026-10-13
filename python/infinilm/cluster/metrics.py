"""In-memory Prometheus metrics for the InfiniLM serving cluster."""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable


def _escape_label(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


@dataclass
class Histogram:
    """Prometheus histogram 的内存表示。"""

    # 观测次数，对应 Prometheus 的 `<name>_count`。
    count: int = 0
    # 所有观测值的总和，对应 Prometheus 的 `<name>_sum`。
    sum: float = 0.0
    # 累计直方图桶：key 是桶上界 `le`，value 是小于等于该上界的观测次数。
    buckets: dict[float, int] = field(default_factory=dict)

    def observe(self, value: float, bucket_bounds: tuple[float, ...]) -> None:
        value = max(0.0, float(value))
        self.count += 1
        self.sum += value
        for bound in bucket_bounds:
            if value <= bound:
                self.buckets[bound] = self.buckets.get(bound, 0) + 1

    def quantile(
        self,
        quantile: float,
        bucket_bounds: tuple[float, ...],
    ) -> float:
        """Approximate a quantile from cumulative Prometheus buckets."""
        if self.count == 0:
            return 0.0

        rank = max(1.0, min(float(self.count), quantile * self.count))
        previous_bound = 0.0
        previous_count = 0
        for bound in bucket_bounds:
            current_count = self.buckets.get(bound, 0)
            if current_count >= rank:
                span = current_count - previous_count
                if span <= 0:
                    return bound
                fraction = (rank - previous_count) / span
                return previous_bound + fraction * (bound - previous_bound)
            previous_bound = bound
            previous_count = current_count

        # The finite bucket layout has no +Inf bucket. Observations beyond the
        # last bound are conservatively represented by that last bound.
        return previous_bound


REQUEST_DURATION_BUCKETS = (
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
    30.0,
    60.0,
    120.0,
)

# 请求总耗时桶边界，单位秒；覆盖毫秒级异常请求到 120 秒长流式请求。
# 对应指标 `infinilm_router_request_duration_seconds`：请求从进入 router
# 到最终完成的端到端耗时。

TTFT_BUCKETS = (
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
    30.0,
)

# TTFT 桶边界，单位秒；用于衡量客户端等待首个生成 token 的时间。
# 对应指标 `infinilm_router_ttft_seconds`：从请求进入 router 到收到
# 第一个流式 token 的耗时。

INTER_TOKEN_BUCKETS = (
    0.001,
    0.002,
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
)

# token 间隔桶边界，单位秒；用于观察流式输出稳定性和 decode 阶段延迟。
# 对应指标 `infinilm_router_inter_token_latency_seconds`：相邻两个
# 流式 token chunk 之间的时间间隔。


@dataclass
class RouterMetrics:
    """router 内存指标集合。

    字段名与 `/metrics` 中的 Prometheus 指标名保持对应关系；带 `worker_`
    前缀的字段均以 `worker_id` 作为标签，用于定位单个推理副本。
    """

    # 当前配置的路由模式：least-loaded / round-robin。
    # 输出为 `infinilm_router_mode{mode="..."} 1`。
    router_mode: str = "round-robin"
    # frontend 收到的逻辑 chat 请求数；一次请求即使发生 worker 重试也只计一次。
    # 输出为 `infinilm_router_requests_total`。
    requests_total: int = 0
    # 实际访问 worker 的总次数；包含首次尝试和失败后的重试。
    # 输出为 `infinilm_router_attempts_total`。
    attempts_total: int = 0
    # 成功完整返回的请求数；输出为 `infinilm_router_successful_requests_total`。
    successful_requests_total: int = 0
    # 客户端请求 stream=true 的数量；输出为
    # `infinilm_router_streaming_requests_total`。
    streaming_requests_total: int = 0
    # 客户端请求 stream=false 或未设置 stream 的数量；输出为
    # `infinilm_router_non_streaming_requests_total`。
    non_streaming_requests_total: int = 0
    # 路由、worker 调用或流式处理中的错误总数。
    # 输出为 `infinilm_router_errors_total`。
    errors_total: int = 0
    # 首个输出前因连接、超时或 5xx 触发的重试次数。
    # 输出为 `infinilm_router_retries_total`。
    retries_total: int = 0
    # 返回给客户端的内部 token chunk 数；不是 tokenizer 精确 usage token 数。
    # 输出为 `infinilm_router_tokens_total`。
    tokens_total: int = 0
    # worker 返回的 OpenAI usage prompt token 累计值；输出为
    # `infinilm_router_usage_prompt_tokens_total`。
    usage_prompt_tokens_total: int = 0
    # worker 返回的 OpenAI usage completion token 累计值；输出为
    # `infinilm_router_usage_completion_tokens_total`。
    usage_completion_tokens_total: int = 0
    # worker 返回的 OpenAI usage total token 累计值；输出为
    # `infinilm_router_usage_total_tokens_total`。
    usage_total_tokens_total: int = 0
    # 当前 frontend 上仍在处理中的请求数。
    # 输出为 `infinilm_router_inflight`。
    inflight: int = 0
    # 按 worker 统计的请求尝试次数；
    # 输出为 `infinilm_router_worker_requests_total{worker_id=...}`。
    worker_requests: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    # 按 worker 统计的 OpenAI usage prompt tokens；
    # 输出为 `infinilm_router_worker_usage_prompt_tokens_total`。
    worker_usage_prompt_tokens: dict[str, int] = field(
        default_factory=lambda: defaultdict(int)
    )
    # 按 worker 统计的 OpenAI usage completion tokens；
    # 输出为 `infinilm_router_worker_usage_completion_tokens_total`。
    worker_usage_completion_tokens: dict[str, int] = field(
        default_factory=lambda: defaultdict(int)
    )
    # 按 worker 统计的 OpenAI usage total tokens；
    # 输出为 `infinilm_router_worker_usage_total_tokens_total`。
    worker_usage_total_tokens: dict[str, int] = field(
        default_factory=lambda: defaultdict(int)
    )
    # 按 worker 统计的错误次数；
    # 输出为 `infinilm_router_worker_errors_total{worker_id=...}`。
    worker_errors: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    # 按 worker 统计的重试落点次数；用于判断某个副本是否频繁故障。
    # 输出为 `infinilm_router_worker_retries_total{worker_id=...}`。
    worker_retries: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    # frontend 视角下每个 worker 的当前在途请求数。
    # 输出为 `infinilm_router_worker_inflight{worker_id=...}`。
    worker_inflight: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    # `least-loaded` 路由的综合评分；分数越低表示当前越适合接收新请求。
    # 输出为 `infinilm_router_worker_load_score{worker_id=...}`。
    worker_load_scores: dict[str, float] = field(default_factory=dict)
    # worker scheduler 上报的 waiting queue 长度。
    # 输出为 `infinilm_router_worker_waiting_requests{worker_id=...}`。
    worker_waiting_requests: dict[str, int] = field(default_factory=dict)
    # worker scheduler 上报的 running request 数量。
    # 输出为 `infinilm_router_worker_running_requests{worker_id=...}`。
    worker_running_requests: dict[str, int] = field(default_factory=dict)
    # worker KV cache 使用比例，范围 0 到 1。
    # 输出为 `infinilm_router_worker_kv_usage_ratio{worker_id=...}`。
    worker_kv_usage: dict[str, float] = field(default_factory=dict)
    # worker 最近一次成功完成请求的 Unix 时间戳。
    # 输出为 `infinilm_router_worker_last_success_timestamp{worker_id=...}`。
    worker_last_success_timestamp: dict[str, float] = field(default_factory=dict)
    # 全局端到端请求耗时 histogram；
    # 输出为 `infinilm_router_request_duration_seconds_*`。
    request_duration: Histogram = field(default_factory=Histogram)
    # 全局 TTFT histogram；输出为 `infinilm_router_ttft_seconds_*`。
    ttft: Histogram = field(default_factory=Histogram)
    # 全局相邻 token chunk 间隔 histogram；
    # 输出为 `infinilm_router_inter_token_latency_seconds_*`。
    inter_token_latency: Histogram = field(default_factory=Histogram)
    # 按 worker 统计的 TTFT histogram；
    # 输出为 `infinilm_router_worker_ttft_seconds_*{worker_id=...}`。
    worker_ttft: dict[str, Histogram] = field(
        default_factory=lambda: defaultdict(Histogram)
    )
    # 按 worker 统计的端到端请求耗时 histogram；
    # 输出为 `infinilm_router_worker_request_duration_seconds_*{worker_id=...}`。
    worker_duration: dict[str, Histogram] = field(
        default_factory=lambda: defaultdict(Histogram)
    )

    def observe_ttft(self, worker_id: str, value: float) -> None:
        self.ttft.observe(value, TTFT_BUCKETS)
        self.worker_ttft[worker_id].observe(value, TTFT_BUCKETS)

    def observe_duration(self, worker_id: str | None, value: float) -> None:
        self.request_duration.observe(value, REQUEST_DURATION_BUCKETS)
        if worker_id is not None:
            self.worker_duration[worker_id].observe(value, REQUEST_DURATION_BUCKETS)

    def observe_inter_token(self, value: float) -> None:
        self.inter_token_latency.observe(value, INTER_TOKEN_BUCKETS)

    def set_worker_load(
        self,
        worker_id: str,
        *,
        score: float,
        waiting_requests: int,
        running_requests: int,
        kv_usage: float,
    ) -> None:
        self.worker_load_scores[worker_id] = max(0.0, float(score))
        self.worker_waiting_requests[worker_id] = max(0, int(waiting_requests))
        self.worker_running_requests[worker_id] = max(0, int(running_requests))
        self.worker_kv_usage[worker_id] = min(1.0, max(0.0, float(kv_usage)))

    def mark_worker_success(self, worker_id: str) -> None:
        self.successful_requests_total += 1
        self.worker_last_success_timestamp[worker_id] = time.time()

    def mark_request_mode(self, streaming: bool) -> None:
        if streaming:
            self.streaming_requests_total += 1
        else:
            self.non_streaming_requests_total += 1

    def record_usage(self, worker_id: str, usage: dict[str, Any]) -> None:
        prompt_tokens = max(0, int(usage.get("prompt_tokens", 0)))
        completion_tokens = max(0, int(usage.get("completion_tokens", 0)))
        total_tokens = max(
            0, int(usage.get("total_tokens", prompt_tokens + completion_tokens))
        )
        self.usage_prompt_tokens_total += prompt_tokens
        self.usage_completion_tokens_total += completion_tokens
        self.usage_total_tokens_total += total_tokens
        self.worker_usage_prompt_tokens[worker_id] += prompt_tokens
        self.worker_usage_completion_tokens[worker_id] += completion_tokens
        self.worker_usage_total_tokens[worker_id] += total_tokens

    def _histogram_lines(
        self,
        name: str,
        help_text: str,
        histogram: Histogram,
        bucket_bounds: tuple[float, ...],
        labels: str | None = None,
        include_metadata: bool = True,
    ) -> list[str]:
        lines = []
        if include_metadata:
            lines.extend(
                [f"# HELP {name} {help_text}.", f"# TYPE {name} histogram"]
            )
        for bound in bucket_bounds:
            count = histogram.buckets.get(bound, 0)
            le_value = str(bound)
            if labels:
                bucket_labels = f'{labels},le="{le_value}"'
            else:
                bucket_labels = f'le="{le_value}"'
            lines.append(f"{name}_bucket{{{bucket_labels}}} {count}")

        if labels:
            lines.append(f'{name}_bucket{{{labels},le="+Inf"}} {histogram.count}')
            lines.append(f"{name}_sum{{{labels}}} {histogram.sum}")
            lines.append(f"{name}_count{{{labels}}} {histogram.count}")
        else:
            lines.append(f'{name}_bucket{{le="+Inf"}} {histogram.count}')
            lines.append(f"{name}_sum {histogram.sum}")
            lines.append(f"{name}_count {histogram.count}")
        return lines

    def render(self, worker_count: int, workers: Iterable[object] = ()) -> str:
        """渲染 Prometheus 文本格式指标。

        `worker_count` 是当前 discovery 可见 worker 数；`workers` 保留给调用方
        传递 discovery 快照，具体 worker 指标值来自本对象维护的字典。
        """
        lines = [
            # 路由模式是配置型枚举值，用值恒为 1 的 gauge 携带 mode 标签。
            "# HELP infinilm_router_mode Configured worker selection mode.",
            "# TYPE infinilm_router_mode gauge",
            f'infinilm_router_mode{{mode="{_escape_label(self.router_mode)}"}} 1',
            # 指标来源说明：仅统计经过本 frontend ClusterRouter 的请求。
            "# HELP infinilm_router_metrics_source Metrics accounting source.",
            "# TYPE infinilm_router_metrics_source gauge",
            'infinilm_router_metrics_source{source="frontend-router"} 1',
            # 当前发现并可用于路由的 worker 数量。
            "# HELP infinilm_router_workers Number of discovered workers.",
            "# TYPE infinilm_router_workers gauge",
            f"infinilm_router_workers {worker_count}",
            # 逻辑请求数：重试不会重复增加该计数。
            "# HELP infinilm_router_requests_total Total chat requests routed.",
            "# TYPE infinilm_router_requests_total counter",
            f"infinilm_router_requests_total {self.requests_total}",
            # worker 尝试数：等于首次尝试加重试。
            "# HELP infinilm_router_attempts_total Total worker attempts, including retries.",
            "# TYPE infinilm_router_attempts_total counter",
            f"infinilm_router_attempts_total {self.attempts_total}",
            # 成功完成并返回的请求数。
            "# HELP infinilm_router_successful_requests_total Successfully completed requests.",
            "# TYPE infinilm_router_successful_requests_total counter",
            f"infinilm_router_successful_requests_total {self.successful_requests_total}",
            # stream=true 的请求数；TTFT 和 token chunk 指标仅该类请求更新。
            "# HELP infinilm_router_streaming_requests_total Streaming chat requests.",
            "# TYPE infinilm_router_streaming_requests_total counter",
            f"infinilm_router_streaming_requests_total {self.streaming_requests_total}",
            # stream=false 或未设置 stream 的请求数。
            "# HELP infinilm_router_non_streaming_requests_total Non-streaming chat requests.",
            "# TYPE infinilm_router_non_streaming_requests_total counter",
            f"infinilm_router_non_streaming_requests_total {self.non_streaming_requests_total}",
            # 路由或 worker 失败次数；包含未重试的失败和最终失败。
            "# HELP infinilm_router_errors_total Total routing or worker errors.",
            "# TYPE infinilm_router_errors_total counter",
            f"infinilm_router_errors_total {self.errors_total}",
            # 首个输出前安全重试到其他 worker 的次数。
            "# HELP infinilm_router_retries_total Total retry attempts after worker failure.",
            "# TYPE infinilm_router_retries_total counter",
            f"infinilm_router_retries_total {self.retries_total}",
            # 输出 token chunk 数，不表示精确 tokenizer token 用量。
            "# HELP infinilm_router_tokens_total Token chunks streamed to clients.",
            "# TYPE infinilm_router_tokens_total counter",
            f"infinilm_router_tokens_total {self.tokens_total}",
            # worker 返回的 OpenAI usage 精确 token 累计值。
            "# HELP infinilm_router_usage_prompt_tokens_total Prompt usage tokens returned by workers.",
            "# TYPE infinilm_router_usage_prompt_tokens_total counter",
            f"infinilm_router_usage_prompt_tokens_total {self.usage_prompt_tokens_total}",
            "# HELP infinilm_router_usage_completion_tokens_total Completion usage tokens returned by workers.",
            "# TYPE infinilm_router_usage_completion_tokens_total counter",
            f"infinilm_router_usage_completion_tokens_total {self.usage_completion_tokens_total}",
            "# HELP infinilm_router_usage_total_tokens_total Total usage tokens returned by workers.",
            "# TYPE infinilm_router_usage_total_tokens_total counter",
            f"infinilm_router_usage_total_tokens_total {self.usage_total_tokens_total}",
            # frontend 当前在途请求 gauge；正常请求结束或异常退出后递减。
            "# HELP infinilm_router_inflight Requests currently routed by this frontend.",
            "# TYPE infinilm_router_inflight gauge",
            f"infinilm_router_inflight {self.inflight}",
        ]

        # 以下 per-worker counter 均携带 worker_id 标签，用于对比副本健康和负载。
        for worker_id, value in sorted(self.worker_requests.items()):
            safe = _escape_label(worker_id)
            lines.append(
                f'infinilm_router_worker_requests_total{{worker_id="{safe}"}} {value}'
            )
        for worker_id, value in sorted(self.worker_usage_prompt_tokens.items()):
            safe = _escape_label(worker_id)
            lines.append(
                f'infinilm_router_worker_usage_prompt_tokens_total{{worker_id="{safe}"}} {value}'
            )
        for worker_id, value in sorted(self.worker_usage_completion_tokens.items()):
            safe = _escape_label(worker_id)
            lines.append(
                f'infinilm_router_worker_usage_completion_tokens_total{{worker_id="{safe}"}} {value}'
            )
        for worker_id, value in sorted(self.worker_usage_total_tokens.items()):
            safe = _escape_label(worker_id)
            lines.append(
                f'infinilm_router_worker_usage_total_tokens_total{{worker_id="{safe}"}} {value}'
            )
        for worker_id, value in sorted(self.worker_errors.items()):
            safe = _escape_label(worker_id)
            lines.append(
                f'infinilm_router_worker_errors_total{{worker_id="{safe}"}} {value}'
            )
        for worker_id, value in sorted(self.worker_retries.items()):
            safe = _escape_label(worker_id)
            lines.append(
                f'infinilm_router_worker_retries_total{{worker_id="{safe}"}} {value}'
            )
        for worker_id, value in sorted(self.worker_inflight.items()):
            safe = _escape_label(worker_id)
            lines.append(
                f'infinilm_router_worker_inflight{{worker_id="{safe}"}} {value}'
            )
        for worker_id, value in sorted(self.worker_load_scores.items()):
            safe = _escape_label(worker_id)
            lines.append(
                f'infinilm_router_worker_load_score{{worker_id="{safe}"}} {value}'
            )
        for worker_id, value in sorted(self.worker_waiting_requests.items()):
            safe = _escape_label(worker_id)
            lines.append(
                f'infinilm_router_worker_waiting_requests{{worker_id="{safe}"}} {value}'
            )
        for worker_id, value in sorted(self.worker_running_requests.items()):
            safe = _escape_label(worker_id)
            lines.append(
                f'infinilm_router_worker_running_requests{{worker_id="{safe}"}} {value}'
            )
        for worker_id, value in sorted(self.worker_kv_usage.items()):
            safe = _escape_label(worker_id)
            lines.append(
                f'infinilm_router_worker_kv_usage_ratio{{worker_id="{safe}"}} {value}'
            )
        for worker_id, value in sorted(self.worker_last_success_timestamp.items()):
            safe = _escape_label(worker_id)
            lines.append(
                f'infinilm_router_worker_last_success_timestamp{{worker_id="{safe}"}} {value}'
            )

        # 全局耗时、TTFT 和 token 间隔 histogram；输出 `_bucket/_sum/_count`。
        lines.extend(
            self._histogram_lines(
                "infinilm_router_request_duration_seconds",
                "End-to-end routed request duration",
                self.request_duration,
                REQUEST_DURATION_BUCKETS,
            )
        )
        lines.extend(
            self._histogram_lines(
                "infinilm_router_ttft_seconds",
                "Time to first streamed token",
                self.ttft,
                TTFT_BUCKETS,
            )
        )
        lines.extend(
            self._histogram_lines(
                "infinilm_router_inter_token_latency_seconds",
                "Interval between streamed token chunks",
                self.inter_token_latency,
                INTER_TOKEN_BUCKETS,
            )
        )

        # per-worker histogram 只在第一个序列前输出 HELP/TYPE，
        # 避免 Prometheus 文本格式中同一 metric family 的元数据重复。
        worker_ttft = sorted(self.worker_ttft.items())
        for index, (worker_id, histogram) in enumerate(worker_ttft):
            lines.extend(
                self._histogram_lines(
                    "infinilm_router_worker_ttft_seconds",
                    "Per-worker time to first streamed token",
                    histogram,
                    TTFT_BUCKETS,
                    labels=f'worker_id="{_escape_label(worker_id)}"',
                    include_metadata=index == 0,
                )
            )
        worker_duration = sorted(self.worker_duration.items())
        for index, (worker_id, histogram) in enumerate(worker_duration):
            lines.extend(
                self._histogram_lines(
                    "infinilm_router_worker_request_duration_seconds",
                    "Per-worker end-to-end routed request duration",
                    histogram,
                    REQUEST_DURATION_BUCKETS,
                    labels=f'worker_id="{_escape_label(worker_id)}"',
                    include_metadata=index == 0,
                )
            )

        return "\n".join(lines) + "\n"


@dataclass
class FrontendMetrics:
    """frontend 聚合指标入口，目前直接委托给 RouterMetrics。"""

    router: RouterMetrics = field(default_factory=RouterMetrics)

    def render(self, worker_count: int, workers: Iterable[object] = ()) -> str:
        return self.router.render(worker_count, workers)
