"""
Query generation utilities for piano concerto accompaniment.

This module provides tools for generating time-scale modified audio queries
and annotation files for piano concerto accompaniment research.
"""

# Import the main QueryGenerator class
from .query_generator import QueryGenerator
from .utils import get_audio_files, get_query_timestamps

__all__ = [
    'QueryGenerator',
    'get_audio_files',
    'get_query_timestamps'
]
