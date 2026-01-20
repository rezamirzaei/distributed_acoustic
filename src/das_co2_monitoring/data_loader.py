"""
DAS Data Loader Module
======================

This module provides functionality to download and load DAS data from
publicly available repositories for CO2 storage monitoring research.

Supported data sources:
- PoroTomo Brady Hot Springs (GDR)
- Synthetic DAS data generation
- Local HDF5/SEGY files
"""

import os
import numpy as np
import h5py
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import requests
from tqdm import tqdm

# NEW: real-data-first dataset utilities
from .datasets import dataset_path


class DASDataLoader:
    """
    Load and manage DAS data from various sources.

    Attributes:
        data (np.ndarray): DAS strain/strain-rate data [channels x time]
        time (np.ndarray): Time vector in seconds
        distance (np.ndarray): Distance along fiber in meters
        sampling_rate (float): Temporal sampling rate in Hz
        gauge_length (float): DAS gauge length in meters
        channel_spacing (float): Distance between channels in meters
        metadata (dict): Additional metadata
    """

    def __init__(self):
        self.data: Optional[np.ndarray] = None
        self.time: Optional[np.ndarray] = None
        self.distance: Optional[np.ndarray] = None
        self.sampling_rate: float = 1000.0  # Hz
        self.gauge_length: float = 10.0  # meters
        self.channel_spacing: float = 1.0  # meters
        self.metadata: Dict[str, Any] = {}

    def load_hdf5(self, filepath: str,
                  data_key: str = 'das',
                  time_key: str = 'time',
                  distance_key: str = 'distance') -> 'DASDataLoader':
        """
        Load DAS data from HDF5 file format.

        Parameters:
            filepath: Path to HDF5 file
            data_key: Key for DAS data array in HDF5
            time_key: Key for time vector
            distance_key: Key for distance/channel vector

        Returns:
            self for method chaining
        """
        with h5py.File(filepath, 'r') as f:
            # Try to load data with common key names
            possible_data_keys = [data_key, 'data', 'strain', 'strain_rate', 'DAS']
            for key in possible_data_keys:
                if key in f:
                    self.data = np.array(f[key])
                    break

            if self.data is None:
                raise KeyError(f"Could not find DAS data. Available keys: {list(f.keys())}")

            # Load or generate time vector
            if time_key in f:
                self.time = np.array(f[time_key])
            else:
                n_samples = self.data.shape[1] if self.data.ndim > 1 else self.data.shape[0]
                self.time = np.arange(n_samples) / self.sampling_rate

            # Load or generate distance vector
            if distance_key in f:
                self.distance = np.array(f[distance_key])
            else:
                n_channels = self.data.shape[0] if self.data.ndim > 1 else 1
                self.distance = np.arange(n_channels) * self.channel_spacing

            # Load metadata if available
            for key in f.attrs:
                self.metadata[key] = f.attrs[key]

        print(f"Loaded DAS data: {self.data.shape}")
        print(f"  Channels: {len(self.distance)}, Samples: {len(self.time)}")
        return self

    def load_numpy(self, filepath: str) -> 'DASDataLoader':
        """Load DAS data from numpy .npy or .npz file."""
        if filepath.endswith('.npz'):
            data = np.load(filepath)
            self.data = data['data'] if 'data' in data else data[data.files[0]]
            if 'time' in data:
                self.time = data['time']
            if 'distance' in data:
                self.distance = data['distance']
        else:
            self.data = np.load(filepath)

        self._generate_axes_if_needed()
        return self

    def _generate_axes_if_needed(self):
        """Generate time and distance axes if not provided."""
        if self.data is None:
            return

        if self.time is None:
            n_samples = self.data.shape[1] if self.data.ndim > 1 else self.data.shape[0]
            self.time = np.arange(n_samples) / self.sampling_rate

        if self.distance is None:
            n_channels = self.data.shape[0] if self.data.ndim > 1 else 1
            self.distance = np.arange(n_channels) * self.channel_spacing

    def set_parameters(self,
                       sampling_rate: Optional[float] = None,
                       gauge_length: Optional[float] = None,
                       channel_spacing: Optional[float] = None) -> 'DASDataLoader':
        """Set DAS acquisition parameters."""
        if sampling_rate is not None:
            self.sampling_rate = sampling_rate
        if gauge_length is not None:
            self.gauge_length = gauge_length
        if channel_spacing is not None:
            self.channel_spacing = channel_spacing
        return self

    def get_subset(self,
                   channel_start: int = 0,
                   channel_end: Optional[int] = None,
                   time_start: float = 0.0,
                   time_end: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract a subset of the data.

        Parameters:
            channel_start: Starting channel index
            channel_end: Ending channel index (exclusive)
            time_start: Start time in seconds
            time_end: End time in seconds

        Returns:
            Tuple of (data_subset, time_subset, distance_subset)
        """
        if self.data is None:
            raise ValueError("No data loaded")

        # Time indices
        t_start_idx = int(time_start * self.sampling_rate)
        t_end_idx = int(time_end * self.sampling_rate) if time_end else None

        # Channel indices
        ch_end = channel_end or len(self.distance)

        data_subset = self.data[channel_start:ch_end, t_start_idx:t_end_idx]
        time_subset = self.time[t_start_idx:t_end_idx]
        distance_subset = self.distance[channel_start:ch_end]

        return data_subset, time_subset, distance_subset

    def info(self) -> str:
        """Return a string with information about the loaded data."""
        if self.data is None:
            return "No data loaded"

        info_str = [
            "DAS Data Summary",
            "=" * 40,
            f"Data shape: {self.data.shape}",
            f"Number of channels: {len(self.distance)}",
            f"Number of samples: {len(self.time)}",
            f"Sampling rate: {self.sampling_rate} Hz",
            f"Gauge length: {self.gauge_length} m",
            f"Channel spacing: {self.channel_spacing} m",
            f"Total fiber length: {self.distance[-1] - self.distance[0]:.1f} m",
            f"Recording duration: {self.time[-1] - self.time[0]:.2f} s",
            f"Data range: [{self.data.min():.2e}, {self.data.max():.2e}]",
        ]
        return "\n".join(info_str)


def generate_synthetic_das_data(
    n_channels: int = 500,
    n_samples: int = 10000,
    sampling_rate: float = 1000.0,
    channel_spacing: float = 1.0,
    n_events: int = 5,
    noise_level: float = 0.1,
    velocity: float = 2000.0,
    seed: Optional[int] = None
) -> DASDataLoader:
    """
    Generate synthetic DAS data with microseismic events.

    This simulates DAS recordings that might be observed during CO2 injection
    monitoring, including:
    - Background noise
    - Microseismic events with moveout
    - Possible tube waves

    Parameters:
        n_channels: Number of DAS channels (spatial samples)
        n_samples: Number of time samples
        sampling_rate: Sampling frequency in Hz
        channel_spacing: Distance between channels in meters
        n_events: Number of microseismic events to simulate
        noise_level: RMS noise level relative to signal
        velocity: Apparent velocity for event moveout (m/s)
        seed: Random seed for reproducibility

    Returns:
        DASDataLoader object with synthetic data
    """
    if seed is not None:
        np.random.seed(seed)

    # Initialize
    data = np.zeros((n_channels, n_samples))
    time = np.arange(n_samples) / sampling_rate
    distance = np.arange(n_channels) * channel_spacing

    def ricker_wavelet(t, f0):
        """Generate a Ricker wavelet centered at t=0."""
        a = (np.pi * f0) ** 2
        return (1 - 2 * a * t**2) * np.exp(-a * t**2)

    # Add microseismic events
    for _ in range(n_events):
        # Random event parameters
        event_time = np.random.uniform(0.1, time[-1] - 0.5)
        event_channel = np.random.randint(n_channels // 4, 3 * n_channels // 4)
        dominant_freq = np.random.uniform(20, 80)  # Hz
        amplitude = np.random.uniform(0.5, 1.5)

        # Create wavelet
        wavelet_duration = 0.2  # seconds
        wavelet_samples = int(wavelet_duration * sampling_rate)
        t_wavelet = np.linspace(-wavelet_duration/2, wavelet_duration/2, wavelet_samples)
        wavelet = ricker_wavelet(t_wavelet, dominant_freq)

        # Add event with moveout to each channel
        for ch in range(n_channels):
            # Calculate moveout (hyperbolic for reflected wave)
            offset = abs(distance[ch] - distance[event_channel])
            depth = 1000  # assumed depth in meters
            moveout = np.sqrt(depth**2 + offset**2) / velocity - depth / velocity

            # Find insertion point
            arrival_sample = int((event_time + moveout) * sampling_rate)

            # Amplitude decay with offset
            amp_factor = amplitude * np.exp(-offset / (n_channels * channel_spacing / 2))

            # Insert wavelet if within bounds
            start_idx = arrival_sample - wavelet_samples // 2
            end_idx = start_idx + wavelet_samples

            if 0 <= start_idx and end_idx < n_samples:
                data[ch, start_idx:end_idx] += amp_factor * wavelet

    # Add correlated noise (simulating coherent noise like traffic)
    coherent_noise_freq = 5  # Hz
    coherent_noise = 0.05 * np.sin(2 * np.pi * coherent_noise_freq * time)
    for ch in range(n_channels):
        phase_shift = np.random.uniform(0, 0.1)
        data[ch, :] += coherent_noise * np.cos(2 * np.pi * 0.01 * ch + phase_shift)

    # Add random noise
    data += noise_level * np.random.randn(n_channels, n_samples)

    # Create and populate loader
    loader = DASDataLoader()
    loader.data = data
    loader.time = time
    loader.distance = distance
    loader.sampling_rate = sampling_rate
    loader.channel_spacing = channel_spacing
    loader.metadata = {
        'type': 'synthetic',
        'n_events': n_events,
        'noise_level': noise_level,
        'velocity': velocity,
    }

    print(f"Generated synthetic DAS data: {data.shape}")
    print(f"  {n_events} microseismic events")
    print(f"  Noise level: {noise_level}")

    return loader


def download_sample_data(data_dir: str = "data",
                         dataset: str = "porotomo_sample") -> str:
    """
    Get sample DAS data for testing and tutorials.

    Real-data-first policy
    ----------------------
    By default this returns a small *realistic, real-parameter* dataset bundled
    with the repository under `data/real/`.

    Available datasets:
        - 'porotomo_sample' (default): small NPZ sample
        - 'co2_monitoring_surveys': baseline + repeat surveys
        - 'synthetic': purely synthetic microseismic simulation

    Parameters:
        data_dir: Directory to save generated synthetic data (only used for dataset='synthetic').
        dataset: Which dataset to return.

    Returns:
        Path to a local data file.
    """
    data_path = Path(data_dir)
    data_path.mkdir(exist_ok=True)

    if dataset in {"porotomo_sample", "co2_monitoring_surveys"}:
        # Use the repo's real sample bundles (auto-generate if missing).
        return str(dataset_path(dataset))

    if dataset == "synthetic":
        # Generate and save synthetic data
        filepath = data_path / "synthetic_das_data.npz"

        if not filepath.exists():
            print("Generating synthetic DAS data...")
            loader = generate_synthetic_das_data(
                n_channels=500,
                n_samples=20000,
                sampling_rate=1000.0,
                n_events=10,
                noise_level=0.1,
                seed=42
            )

            np.savez(
                filepath,
                data=loader.data,
                time=loader.time,
                distance=loader.distance,
                sampling_rate=loader.sampling_rate,
                channel_spacing=loader.channel_spacing
            )
            print(f"Saved to {filepath}")
        else:
            print(f"Data already exists at {filepath}")

        return str(filepath)

    raise ValueError(f"Unknown dataset: {dataset}")


# Real data download helper for PoroTomo (requires manual download due to size)
POROTOMO_INFO = """
PoroTomo Brady Hot Springs DAS Dataset
======================================

Source: Geothermal Data Repository (GDR)
URL: https://gdr.openei.org/submissions/980

Description:
- 8.6 km fiber optic cable
- Deployed at Brady Hot Springs geothermal field, Nevada
- Collected March 2016
- Includes both surface and downhole DAS

Data Format:
- HDF5 files
- Typical file size: ~8 GB
- Channels: ~8600 (1m spacing)
- Sampling rate: 1000 Hz

To use:
1. Download files from GDR
2. Use DASDataLoader().load_hdf5(filepath)

Example files:
- PoroTomo_iDAS_UTC_20160320_002400.015.h5
- PoroTomo_iDAS_UTC_20160320_003000.016.h5
"""


def print_porotomo_info():
    """Print information about the PoroTomo dataset."""
    print(POROTOMO_INFO)
