"""Timestamp containers used by the summary timing path."""

from dataclasses import dataclass, field


@dataclass
class RequestTiming:
    service_entered_at_ns: int | None = None
    scheduler_enqueue_before_at_ns: int | None = None

    first_forward_started_at_ns: int | None = None
    first_forward_completed_at_ns: int | None = None
    first_loop_submit_started_at_ns: int | None = None
    first_batch_put_started_at_ns: int | None = None
    first_batch_put_completed_at_ns: int | None = None
    first_route_token_received_at_ns: int | None = None
    first_route_token_end_at_ns: int | None = None

    prefill_batch_size: int | None = None
    decode_step_count: int = 0
    decode_batch_size_counts: dict[int, int] = field(default_factory=dict)
    finish_recorded: bool = False


@dataclass
class ForwardTiming:
    batch_id: str
    forward_started_at_ns: int
    forward_completed_at_ns: int


@dataclass
class TokenTiming:
    token_index: int
    batch_id: str | None = None
    phase: str | None = None
    forward_started_at_ns: int | None = None
    forward_completed_at_ns: int | None = None
    loop_submit_started_at_ns: int | None = None
    batch_put_started_at_ns: int | None = None
    batch_put_completed_at_ns: int | None = None
    route_token_received_at_ns: int | None = None
    route_token_end_at_ns: int | None = None
