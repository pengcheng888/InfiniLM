"""Built-in monitoring package for ClusterFrontend."""

from .page import MONITORING_HTML
from .snapshot import monitoring_snapshot

__all__ = [
    "MONITORING_HTML",
    "monitoring_snapshot",
]
