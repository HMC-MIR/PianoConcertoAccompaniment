"""
TSM utilities for piano concerto accompaniment.

This module provides tools for time-scale modification.
"""

# Import the main QueryGenerator class
from .tsm import TSM, online_tsm

__all__ = [
    'TSM',
    'online_tsm'
]
