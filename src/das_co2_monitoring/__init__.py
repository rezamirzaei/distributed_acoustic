"""
DAS CO2 Monitoring Package
=========================

A Python package for processing Distributed Acoustic Sensing (DAS) data
for CO2 storage monitoring applications.

Modules:
--------
- data_loader: Download and load DAS data from public repositories
- preprocessing: Signal conditioning and filtering
- event_detection: Microseismic event detection algorithms
- visualization: DAS-specific plotting utilities
- monitoring: Time-lapse analysis for CO2 plume tracking
"""

from .data_loader import DASDataLoader, download_sample_data
from .datasets import dataset_path, ensure_real_datasets
from .preprocessing import DASPreprocessor
from .event_detection import EventDetector
from .visualization import DASVisualizer
from .monitoring import CO2Monitor

__version__ = "0.1.0"
__author__ = "DAS Research Team"

__all__ = [
    "DASDataLoader",
    "DASPreprocessor",
    "EventDetector",
    "DASVisualizer",
    "CO2Monitor",
    "download_sample_data",
    "dataset_path",
    "ensure_real_datasets",
]
