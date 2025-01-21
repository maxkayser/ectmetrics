# ectmetrics/__init__.py

"""
EctMetrics: A package for ECT EEG signal processing and seizure quality metrics calculations.

Modules:
    - eeg: Contains functions for EEG signal processing.
    - metric: Contains functions for ECT seizure quality metric calculations.
"""
__version__ = '0.1.0'

# Importing necessary modules
from .eeg import *
from .metric import *

__all__ = [
    'eeg',
    'metric'
]
