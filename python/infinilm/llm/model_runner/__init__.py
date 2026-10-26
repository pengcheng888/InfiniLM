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
    """根据环境变量选择普通 runner 或轻量通信 runner。

    两个 runner 都在函数内延迟导入，避免包初始化阶段形成循环导入，也让
    普通进程在没有通信测试环境变量时不额外构造轻量通信 runner 的依赖。
    """
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
