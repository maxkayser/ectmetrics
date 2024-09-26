# ectmetrics/__init__.py

"""
EctMetrics: A package for ECT EEG signal processing and seizure quality metrics calculations.

Modules:
    - eeg: Contains functions for EEG signal processing.
    - metrics: Contains functions for metric calculations.
"""

# Importing necessary modules
from .eeg import *
from .metrics import *

__all__ = [
    'eeg',
    'metrics'
]