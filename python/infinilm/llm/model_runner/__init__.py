"""Model runner package entry point."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from infinilm.cluster.environment import cluster_communication_test
from infinilm.config.engine_config import EngineConfig


def create_model_runner(
    config: EngineConfig,
    hf_config: Mapping[str, Any] | None = None,
    initialize_processor: bool = True,
) -> Any:
    """创建 ModelRunner，并按配置选择进程边界和 runner 实现。

    是否启用独立 ModelRunner 进程统一由环境变量决定。
    """
    from infinilm.llm.model_runner.model_runner_process import (
        model_runner_process_enabled,
    )

    use_process = model_runner_process_enabled()

    if use_process:
        from infinilm.llm.model_runner.model_runner_process import (
            ModelRunnerProcessClient,
        )

        return ModelRunnerProcessClient(config, dict(hf_config or {}))

    if cluster_communication_test():
        from infinilm.llm.model_runner.communication_model_runner import (
            CommunicationModelRunner,
        )

        return CommunicationModelRunner(
            config,
            hf_config=hf_config,
            initialize_processor=initialize_processor,
        )

    from infinilm.llm.model_runner.model_runner import ModelRunner

    return ModelRunner(config, initialize_processor=initialize_processor)


__all__ = ["create_model_runner"]
