# -*- coding: utf-8 -*-
from pathlib import Path

from pkg_resources import DistributionNotFound, get_distribution

try:
    # Change here if project is renamed and does not equal the package name
    dist_name = __name__
    __version__ = get_distribution(dist_name).version
except DistributionNotFound:
    __version__ = "unknown"
finally:
    del get_distribution, DistributionNotFound

LOGGER_BASE_NAME = __name__

__tool_name__ = Path(__file__).parent.stem
