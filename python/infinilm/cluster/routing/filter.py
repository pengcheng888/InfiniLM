"""Candidate filtering for routable workers."""

from __future__ import annotations

from ..worker_registry import WorkerRecord


class WorkerCandidateFilter:
    """Filter workers that can safely receive a new frontend request."""

    def filter(
        self,
        workers: list[WorkerRecord],
        model_name: str | None = None,
    ) -> list[WorkerRecord]:
        result = [
            worker
            for worker in workers
            if worker.state == "ready" and worker.engine_healthy
        ]
        if model_name:
            result = [
                worker for worker in result if worker.served_model_name == model_name
            ]
        return result
