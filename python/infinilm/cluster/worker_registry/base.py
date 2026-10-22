"""Backend-neutral worker registry contract."""

from __future__ import annotations

from typing import Any, Iterable, Protocol

from .types import WorkerRecord, WorkerType


class WorkerRegistry(Protocol):
    """The registration and discovery contract for worker records."""

    lease_seconds: float

    def new_record(self, **kwargs: Any) -> WorkerRecord: ...

    async def register(self, record: WorkerRecord) -> WorkerRecord: ...

    async def heartbeat(
        self,
        record: WorkerRecord,
        *,
        engine_healthy: bool | None = None,
    ) -> WorkerRecord: ...

    async def unregister(self, record: WorkerRecord) -> None: ...

    async def snapshot(
        self,
        namespace: str,
        endpoint: str | None = None,
        served_model_name: str | None = None,
        worker_types: Iterable[WorkerType | str] | None = None,
    ) -> list[WorkerRecord]: ...

    async def aclose(self) -> None: ...
