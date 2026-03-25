"""
Package inti untuk project ini / vehicle line counter.

Agar dapat mengimport dengan cara yang lebih mudah:

    from pedestrian_line_counter import Detector, Tracker, LineCounter
    from pedestrian_line_counter import AppConfig, get_default_config
"""

from .config import (  
    AppConfig,
    IOConfig,
    LineConfig,
    ModelConfig,
    ReportConfig,
    TrackerConfig,
    get_default_config,
)
from .detector import Detector  
from .line_counter import LineCounter  
from .structures import Detection, Track  
from .tracker import Tracker  


def create_app(*args, **kwargs):
    """
    Lazy import to keep detector/runtime entry points usable even when FastAPI
    is not installed in the active Python environment.
    """

    from .api import create_app as _create_app

    return _create_app(*args, **kwargs)
