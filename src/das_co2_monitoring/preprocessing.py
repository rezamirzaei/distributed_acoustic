"""
DAS Preprocessing Module
========================

Signal conditioning and filtering routines for DAS data processing.

Includes:
- Strain rate to strain conversion
- Bandpass, highpass, and lowpass filtering
- Denoising (median filter, SVD, f-k filtering)
- Normalization and gain control
- Temporal and spatial resampling
"""

import numpy as np
from scipy import signal
from scipy.ndimage import median_filter
from typing import Tuple, Optional, Literal
import warnings


class DASPreprocessor:
    """
    Preprocessing pipeline for DAS data.

    Example usage:
        preprocessor = DASPreprocessor(sampling_rate=1000.0)
        clean_data = (preprocessor
            .set_data(raw_data)
            .remove_mean()
            .bandpass_filter(1.0, 100.0)
            .median_denoise()
            .get_data())
    """

    def __init__(self, sampling_rate: float = 1000.0, channel_spacing: float = 1.0):
        """
        Initialize preprocessor.

        Parameters:
            sampling_rate: Temporal sampling rate in Hz
            channel_spacing: Spatial sampling in meters
        """
        self.sampling_rate = sampling_rate
        self.channel_spacing = channel_spacing
        self.data: Optional[np.ndarray] = None
        self._history: list = []

    def set_data(self, data: np.ndarray) -> 'DASPreprocessor':
        """Set the data to process [channels x time]."""
        self.data = data.copy()
        self._history = ['set_data']
        return self

    def get_data(self) -> np.ndarray:
        """Return the processed data."""
        if self.data is None:
            raise ValueError("No data set")
        return self.data

    def get_history(self) -> list:
        """Return processing history."""
        return self._history

    # ========================
    # Basic Operations
    # ========================

    def remove_mean(self, axis: int = 1) -> 'DASPreprocessor':
        """
        Remove mean from each channel (axis=1) or each time sample (axis=0).

        Parameters:
            axis: 1 for temporal mean (per channel), 0 for spatial mean (per sample)
        """
        self.data = self.data - np.mean(self.data, axis=axis, keepdims=True)
        self._history.append(f'remove_mean(axis={axis})')
        return self

    def remove_trend(self, order: int = 1) -> 'DASPreprocessor':
        """
        Remove polynomial trend from each channel.

        Parameters:
            order: Polynomial order (1=linear, 2=quadratic)
        """
        n_channels, n_samples = self.data.shape
        x = np.arange(n_samples)

        for i in range(n_channels):
            coeffs = np.polyfit(x, self.data[i, :], order)
            trend = np.polyval(coeffs, x)
            self.data[i, :] -= trend

        self._history.append(f'remove_trend(order={order})')
        return self

    def normalize(self, method: Literal['std', 'max', 'rms'] = 'std') -> 'DASPreprocessor':
        """
        Normalize each channel.

        Parameters:
            method: 'std' (divide by standard deviation),
                   'max' (divide by max absolute value),
                   'rms' (divide by RMS)
        """
        if method == 'std':
            scales = np.std(self.data, axis=1, keepdims=True)
        elif method == 'max':
            scales = np.max(np.abs(self.data), axis=1, keepdims=True)
        elif method == 'rms':
            scales = np.sqrt(np.mean(self.data**2, axis=1, keepdims=True))
        else:
            raise ValueError(f"Unknown normalization method: {method}")

        # Avoid division by zero
        scales[scales == 0] = 1.0
        self.data = self.data / scales

        self._history.append(f'normalize(method={method})')
        return self

    # ========================
    # Filtering
    # ========================

    def bandpass_filter(self,
                        low_freq: float,
                        high_freq: float,
                        order: int = 4,
                        zerophase: bool = True) -> 'DASPreprocessor':
        """
        Apply bandpass filter to each channel.

        Parameters:
            low_freq: Low cutoff frequency in Hz
            high_freq: High cutoff frequency in Hz
            order: Filter order
            zerophase: If True, apply zero-phase filtering (filtfilt)
        """
        nyquist = self.sampling_rate / 2
        low = low_freq / nyquist
        high = high_freq / nyquist

        if high >= 1.0:
            high = 0.99
            warnings.warn(f"High frequency clamped to {high * nyquist:.1f} Hz (below Nyquist)")

        b, a = signal.butter(order, [low, high], btype='band')

        if zerophase:
            self.data = signal.filtfilt(b, a, self.data, axis=1)
        else:
            self.data = signal.lfilter(b, a, self.data, axis=1)

        self._history.append(f'bandpass_filter({low_freq}, {high_freq})')
        return self

    def highpass_filter(self,
                        cutoff: float,
                        order: int = 4,
                        zerophase: bool = True) -> 'DASPreprocessor':
        """Apply highpass filter to remove low-frequency noise."""
        nyquist = self.sampling_rate / 2
        normalized_cutoff = cutoff / nyquist

        b, a = signal.butter(order, normalized_cutoff, btype='high')

        if zerophase:
            self.data = signal.filtfilt(b, a, self.data, axis=1)
        else:
            self.data = signal.lfilter(b, a, self.data, axis=1)

        self._history.append(f'highpass_filter({cutoff})')
        return self

    def lowpass_filter(self,
                       cutoff: float,
                       order: int = 4,
                       zerophase: bool = True) -> 'DASPreprocessor':
        """Apply lowpass filter to remove high-frequency noise."""
        nyquist = self.sampling_rate / 2
        normalized_cutoff = cutoff / nyquist

        if normalized_cutoff >= 1.0:
            normalized_cutoff = 0.99

        b, a = signal.butter(order, normalized_cutoff, btype='low')

        if zerophase:
            self.data = signal.filtfilt(b, a, self.data, axis=1)
        else:
            self.data = signal.lfilter(b, a, self.data, axis=1)

        self._history.append(f'lowpass_filter({cutoff})')
        return self

    def notch_filter(self,
                     freq: float,
                     quality: float = 30.0) -> 'DASPreprocessor':
        """
        Apply notch filter to remove specific frequency (e.g., 50/60 Hz powerline).

        Parameters:
            freq: Frequency to remove in Hz
            quality: Quality factor (higher = narrower notch)
        """
        b, a = signal.iirnotch(freq, quality, self.sampling_rate)
        self.data = signal.filtfilt(b, a, self.data, axis=1)

        self._history.append(f'notch_filter({freq})')
        return self

    # ========================
    # Denoising
    # ========================

    def median_denoise(self,
                       kernel_size: Tuple[int, int] = (1, 5)) -> 'DASPreprocessor':
        """
        Apply median filter for denoising.

        Parameters:
            kernel_size: (spatial, temporal) kernel size
        """
        self.data = median_filter(self.data, size=kernel_size)
        self._history.append(f'median_denoise(kernel={kernel_size})')
        return self

    def svd_denoise(self,
                    n_components: int = 10,
                    remove_components: Optional[list] = None) -> 'DASPreprocessor':
        """
        SVD-based denoising to remove coherent noise.

        Parameters:
            n_components: Number of components to keep for reconstruction
            remove_components: Specific component indices to remove (e.g., [0, 1] for first two)
        """
        U, s, Vt = np.linalg.svd(self.data, full_matrices=False)

        if remove_components is not None:
            # Remove specific components
            s_filtered = s.copy()
            for idx in remove_components:
                if idx < len(s):
                    s_filtered[idx] = 0
            self.data = U @ np.diag(s_filtered) @ Vt
        else:
            # Keep only top n_components
            self.data = U[:, :n_components] @ np.diag(s[:n_components]) @ Vt[:n_components, :]

        self._history.append(f'svd_denoise(n_components={n_components})')
        return self

    def fk_filter(self,
                  velocity_min: Optional[float] = None,
                  velocity_max: Optional[float] = None,
                  mode: Literal['pass', 'reject'] = 'reject') -> 'DASPreprocessor':
        """
        Apply frequency-wavenumber (f-k) filter.

        Useful for removing coherent noise with specific apparent velocities
        (e.g., surface waves, traffic noise).

        Parameters:
            velocity_min: Minimum apparent velocity to filter (m/s)
            velocity_max: Maximum apparent velocity to filter (m/s)
            mode: 'reject' to remove velocities in range, 'pass' to keep only those
        """
        n_channels, n_samples = self.data.shape

        # 2D FFT
        fk_data = np.fft.fft2(self.data)

        # Frequency and wavenumber axes
        freqs = np.fft.fftfreq(n_samples, 1/self.sampling_rate)
        wavenumbers = np.fft.fftfreq(n_channels, self.channel_spacing)

        # Create velocity mask
        f_grid, k_grid = np.meshgrid(freqs, wavenumbers)

        # Avoid division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            apparent_velocity = np.abs(f_grid / k_grid)
            apparent_velocity[np.isinf(apparent_velocity)] = np.inf
            apparent_velocity[np.isnan(apparent_velocity)] = np.inf

        # Create filter mask
        mask = np.ones_like(fk_data, dtype=float)

        if velocity_min is not None and velocity_max is not None:
            in_range = (apparent_velocity >= velocity_min) & (apparent_velocity <= velocity_max)
            if mode == 'reject':
                mask[in_range] = 0
            else:  # pass
                mask[~in_range] = 0

        # Apply taper to avoid ringing
        taper_width = 10
        for i in range(taper_width):
            taper_val = 0.5 * (1 + np.cos(np.pi * (taper_width - i) / taper_width))
            # Apply taper at mask boundaries (simplified)

        # Apply filter and inverse FFT
        fk_filtered = fk_data * mask
        self.data = np.real(np.fft.ifft2(fk_filtered))

        self._history.append(f'fk_filter(v=[{velocity_min}, {velocity_max}], mode={mode})')
        return self

    # ========================
    # Resampling
    # ========================

    def temporal_downsample(self, factor: int) -> 'DASPreprocessor':
        """
        Downsample in time by given factor.

        Parameters:
            factor: Downsampling factor (e.g., 2 = half the samples)
        """
        # Anti-alias filter first
        nyquist_new = (self.sampling_rate / factor) / 2
        self.lowpass_filter(nyquist_new * 0.9)

        # Downsample
        self.data = self.data[:, ::factor]
        self.sampling_rate = self.sampling_rate / factor

        self._history.append(f'temporal_downsample(factor={factor})')
        return self

    def spatial_downsample(self, factor: int) -> 'DASPreprocessor':
        """
        Downsample in space (stack adjacent channels).

        Parameters:
            factor: Number of channels to stack
        """
        n_channels, n_samples = self.data.shape
        new_n_channels = n_channels // factor

        # Stack channels
        reshaped = self.data[:new_n_channels * factor, :].reshape(
            new_n_channels, factor, n_samples
        )
        self.data = np.mean(reshaped, axis=1)
        self.channel_spacing = self.channel_spacing * factor

        self._history.append(f'spatial_downsample(factor={factor})')
        return self

    # ========================
    # DAS-Specific Operations
    # ========================

    def strain_rate_to_strain(self,
                              integration_method: Literal['cumsum', 'trapz'] = 'cumsum'
                              ) -> 'DASPreprocessor':
        """
        Convert strain rate to strain by integration.

        Parameters:
            integration_method: 'cumsum' for simple summation, 'trapz' for trapezoidal
        """
        dt = 1.0 / self.sampling_rate

        if integration_method == 'cumsum':
            self.data = np.cumsum(self.data, axis=1) * dt
        elif integration_method == 'trapz':
            # Trapezoidal integration
            integrated = np.zeros_like(self.data)
            for i in range(1, self.data.shape[1]):
                integrated[:, i] = integrated[:, i-1] + 0.5 * (
                    self.data[:, i] + self.data[:, i-1]
                ) * dt
            self.data = integrated

        self._history.append(f'strain_rate_to_strain(method={integration_method})')
        return self

    def apply_gauge_length_correction(self,
                                       gauge_length: float,
                                       target_gauge_length: float) -> 'DASPreprocessor':
        """
        Apply gauge length correction for comparing data with different gauge lengths.

        Parameters:
            gauge_length: Original gauge length in meters
            target_gauge_length: Desired effective gauge length in meters
        """
        # Number of channels to combine
        n_combine = int(target_gauge_length / gauge_length)

        if n_combine > 1:
            n_channels = self.data.shape[0]
            new_n_channels = n_channels - n_combine + 1

            corrected = np.zeros((new_n_channels, self.data.shape[1]))
            for i in range(new_n_channels):
                corrected[i, :] = np.mean(self.data[i:i+n_combine, :], axis=0)

            self.data = corrected

        self._history.append(f'gauge_length_correction({gauge_length}->{target_gauge_length})')
        return self

    def automatic_gain_control(self,
                               window_length: float = 0.5) -> 'DASPreprocessor':
        """
        Apply automatic gain control (AGC) for display purposes.

        Parameters:
            window_length: AGC window length in seconds
        """
        window_samples = int(window_length * self.sampling_rate)

        for i in range(self.data.shape[0]):
            # Calculate RMS in sliding window
            rms = np.sqrt(
                np.convolve(
                    self.data[i, :]**2,
                    np.ones(window_samples)/window_samples,
                    mode='same'
                )
            )
            # Avoid division by zero
            rms[rms < 1e-10] = 1e-10
            self.data[i, :] = self.data[i, :] / rms

        self._history.append(f'automatic_gain_control(window={window_length}s)')
        return self


def preprocess_for_event_detection(data: np.ndarray,
                                   sampling_rate: float = 1000.0) -> np.ndarray:
    """
    Standard preprocessing pipeline for microseismic event detection.

    Parameters:
        data: Raw DAS data [channels x time]
        sampling_rate: Sampling rate in Hz

    Returns:
        Preprocessed data ready for event detection
    """
    preprocessor = DASPreprocessor(sampling_rate=sampling_rate)

    processed = (preprocessor
                 .set_data(data)
                 .remove_mean()
                 .highpass_filter(1.0)  # Remove DC and very low frequencies
                 .bandpass_filter(5.0, 100.0)  # Typical microseismic band
                 .normalize(method='std')
                 .get_data())

    return processed


def preprocess_for_co2_monitoring(data: np.ndarray,
                                  sampling_rate: float = 1000.0) -> np.ndarray:
    """
    Preprocessing pipeline optimized for CO2 storage monitoring.

    Focuses on:
    - Removing surface/cultural noise
    - Enhancing subtle strain changes
    - Preserving low-frequency signals from fluid movement

    Parameters:
        data: Raw DAS data [channels x time]
        sampling_rate: Sampling rate in Hz

    Returns:
        Preprocessed data for CO2 monitoring analysis
    """
    preprocessor = DASPreprocessor(sampling_rate=sampling_rate)

    processed = (preprocessor
                 .set_data(data)
                 .remove_mean()
                 .remove_trend(order=1)
                 .bandpass_filter(0.5, 50.0)  # Include lower frequencies for strain
                 .svd_denoise(n_components=20)  # Remove coherent noise
                 .normalize(method='rms')
                 .get_data())

    return processed
