# System modules for the PianoConcertoAccompaniment benchmark.
# Each module implements offline_processing() and online_processing().

from .oltw import verify_oltw_installation, parse_oltw_alignment, online_processing as online_processing_oltw