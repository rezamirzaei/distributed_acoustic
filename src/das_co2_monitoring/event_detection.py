"""
Event Detection Module
======================

Microseismic event detection algorithms for DAS data.

Includes:
- STA/LTA (Short-Term Average / Long-Term Average) triggering
- Template matching for repeating events
- Amplitude threshold detection
- Arrival time picking
"""

import numpy as np
from scipy import signal
from scipy.ndimage import label
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class DetectedEvent:
    """Container for detected microseismic event information."""
    event_id: int
    time: float  # Event time in seconds
    channel: int  # Channel with highest amplitude
    amplitude: float  # Peak amplitude
    duration: float  # Event duration in seconds
    n_channels: int  # Number of channels that detected the event
    snr: float  # Signal-to-noise ratio
    picks: Optional[Dict[int, float]] = None  # Channel: arrival time


class EventDetector:
    """
    Microseismic event detection for DAS data.

    Implements multiple detection methods suitable for CO2 storage monitoring.

    Example usage:
        detector = EventDetector(sampling_rate=1000.0)
        events = detector.sta_lta_detect(data, sta_window=0.05, lta_window=0.5)
    """

    def __init__(self, sampling_rate: float = 1000.0, channel_spacing: float = 1.0):
        """
        Initialize event detector.

        Parameters:
            sampling_rate: Temporal sampling rate in Hz
            channel_spacing: Distance between channels in meters
        """
        self.sampling_rate = sampling_rate
        self.channel_spacing = channel_spacing

    def sta_lta_detect(self,
                       data: np.ndarray,
                       sta_window: float = 0.05,
                       lta_window: float = 0.5,
                       trigger_on: float = 3.0,
                       trigger_off: float = 1.5,
                       min_channels: int = 10,
                       min_duration: float = 0.02) -> List[DetectedEvent]:
        """
        STA/LTA event detection across all channels.

        Parameters:
            data: DAS data [channels x time]
            sta_window: Short-term average window in seconds
            lta_window: Long-term average window in seconds
            trigger_on: STA/LTA ratio to trigger event start
            trigger_off: STA/LTA ratio to end event
            min_channels: Minimum channels required to declare an event
            min_duration: Minimum event duration in seconds

        Returns:
            List of DetectedEvent objects
        """
        n_channels, n_samples = data.shape
        sta_samples = int(sta_window * self.sampling_rate)
        lta_samples = int(lta_window * self.sampling_rate)
        min_samples = int(min_duration * self.sampling_rate)

        # Calculate STA/LTA ratio for each channel
        triggers = np.zeros_like(data, dtype=bool)
        characteristic_function = np.zeros_like(data)

        for ch in range(n_channels):
            cf = self._characteristic_function(data[ch, :])
            ratio = self._sta_lta_ratio(cf, sta_samples, lta_samples)
            characteristic_function[ch, :] = ratio
            triggers[ch, :] = ratio > trigger_on

        # Find events where multiple channels trigger simultaneously
        channel_count = np.sum(triggers, axis=0)
        event_mask = channel_count >= min_channels

        # Label connected event regions
        labeled_array, n_events = label(event_mask)

        events = []
        for event_id in range(1, n_events + 1):
            event_indices = np.where(labeled_array == event_id)[0]

            if len(event_indices) < min_samples:
                continue

            # Event timing
            start_idx = event_indices[0]
            end_idx = event_indices[-1]
            event_time = start_idx / self.sampling_rate
            duration = (end_idx - start_idx) / self.sampling_rate

            # Find peak channel and amplitude
            event_data = data[:, start_idx:end_idx+1]
            peak_amplitudes = np.max(np.abs(event_data), axis=1)
            peak_channel = np.argmax(peak_amplitudes)
            peak_amplitude = peak_amplitudes[peak_channel]

            # Count channels with significant amplitude
            noise_level = np.std(data[:, :start_idx], axis=1) if start_idx > 0 else np.std(data, axis=1)
            noise_level[noise_level == 0] = 1e-10
            snr_per_channel = peak_amplitudes / noise_level
            n_detected_channels = np.sum(snr_per_channel > trigger_on)

            # Calculate SNR
            snr = np.median(snr_per_channel[snr_per_channel > trigger_on])

            events.append(DetectedEvent(
                event_id=len(events) + 1,
                time=event_time,
                channel=peak_channel,
                amplitude=peak_amplitude,
                duration=duration,
                n_channels=n_detected_channels,
                snr=snr
            ))

        print(f"Detected {len(events)} events using STA/LTA")
        return events

    def _characteristic_function(self, trace: np.ndarray) -> np.ndarray:
        """Calculate characteristic function (squared amplitude)."""
        return trace ** 2

    def _sta_lta_ratio(self, cf: np.ndarray, sta_len: int, lta_len: int) -> np.ndarray:
        """Calculate STA/LTA ratio."""
        # Cumulative sum for efficient moving average
        cumsum = np.cumsum(cf)

        # STA
        sta = np.zeros_like(cf)
        sta[sta_len:] = (cumsum[sta_len:] - cumsum[:-sta_len]) / sta_len

        # LTA
        lta = np.zeros_like(cf)
        lta[lta_len:] = (cumsum[lta_len:] - cumsum[:-lta_len]) / lta_len

        # Avoid division by zero
        lta[lta < 1e-10] = 1e-10

        return sta / lta

    def amplitude_threshold_detect(self,
                                   data: np.ndarray,
                                   threshold: float = 3.0,
                                   min_channels: int = 5,
                                   min_duration: float = 0.01,
                                   merge_window: float = 0.1) -> List[DetectedEvent]:
        """
        Simple amplitude threshold detection.

        Parameters:
            data: DAS data [channels x time]
            threshold: Number of standard deviations above mean for detection
            min_channels: Minimum channels required
            min_duration: Minimum event duration in seconds
            merge_window: Window to merge nearby detections in seconds

        Returns:
            List of DetectedEvent objects
        """
        n_channels, n_samples = data.shape

        # Calculate threshold per channel
        means = np.mean(data, axis=1, keepdims=True)
        stds = np.std(data, axis=1, keepdims=True)
        stds[stds == 0] = 1e-10

        # Detect above threshold
        triggers = np.abs(data - means) > threshold * stds

        # Count channels triggering at each time
        channel_count = np.sum(triggers, axis=0)
        event_mask = channel_count >= min_channels

        # Label and extract events
        min_samples = int(min_duration * self.sampling_rate)
        merge_samples = int(merge_window * self.sampling_rate)

        # Dilate to merge nearby detections
        if merge_samples > 1:
            kernel = np.ones(merge_samples)
            event_mask = np.convolve(event_mask.astype(float), kernel, mode='same') > 0

        labeled_array, n_events = label(event_mask)

        events = []
        for event_id in range(1, n_events + 1):
            event_indices = np.where(labeled_array == event_id)[0]

            if len(event_indices) < min_samples:
                continue

            start_idx = event_indices[0]
            end_idx = event_indices[-1]

            event_data = data[:, start_idx:end_idx+1]
            peak_amplitudes = np.max(np.abs(event_data), axis=1)
            peak_channel = np.argmax(peak_amplitudes)

            events.append(DetectedEvent(
                event_id=len(events) + 1,
                time=start_idx / self.sampling_rate,
                channel=peak_channel,
                amplitude=peak_amplitudes[peak_channel],
                duration=(end_idx - start_idx) / self.sampling_rate,
                n_channels=int(channel_count[start_idx:end_idx+1].max()),
                snr=peak_amplitudes[peak_channel] / stds[peak_channel, 0]
            ))

        print(f"Detected {len(events)} events using amplitude threshold")
        return events

    def template_matching(self,
                          data: np.ndarray,
                          template: np.ndarray,
                          threshold: float = 0.7,
                          channel_range: Optional[Tuple[int, int]] = None
                          ) -> List[DetectedEvent]:
        """
        Template matching detection for finding repeating events.

        Parameters:
            data: DAS data [channels x time]
            template: Template event [channels x time]
            threshold: Correlation coefficient threshold (0-1)
            channel_range: (start, end) channels to use for matching

        Returns:
            List of DetectedEvent objects
        """
        n_channels, n_samples = data.shape
        t_channels, t_samples = template.shape

        if channel_range:
            ch_start, ch_end = channel_range
        else:
            ch_start, ch_end = 0, min(n_channels, t_channels)

        # Stack channels for cross-correlation
        data_stack = data[ch_start:ch_end, :].flatten()
        template_stack = template[:ch_end-ch_start, :].flatten()

        # Normalized cross-correlation
        correlation = signal.correlate(data_stack, template_stack, mode='valid')

        # Normalize
        data_norm = np.sqrt(
            signal.correlate(data_stack**2, np.ones(len(template_stack)), mode='valid')
        )
        template_norm = np.sqrt(np.sum(template_stack**2))
        data_norm[data_norm == 0] = 1e-10

        correlation = correlation / (data_norm * template_norm)

        # Find peaks above threshold
        peaks, properties = signal.find_peaks(correlation, height=threshold, distance=t_samples)

        events = []
        for i, peak_idx in enumerate(peaks):
            # Convert back to original indexing
            time_idx = peak_idx

            events.append(DetectedEvent(
                event_id=i + 1,
                time=time_idx / self.sampling_rate,
                channel=(ch_start + ch_end) // 2,
                amplitude=properties['peak_heights'][i],
                duration=t_samples / self.sampling_rate,
                n_channels=ch_end - ch_start,
                snr=properties['peak_heights'][i] / (1 - properties['peak_heights'][i] + 0.01)
            ))

        print(f"Detected {len(events)} events using template matching")
        return events

    def pick_arrivals(self,
                      data: np.ndarray,
                      event: DetectedEvent,
                      window: float = 0.5,
                      method: str = 'aic') -> DetectedEvent:
        """
        Pick arrival times for a detected event on all channels.

        Parameters:
            data: DAS data [channels x time]
            event: DetectedEvent to pick arrivals for
            window: Time window around event to search for arrivals
            method: 'aic' (Akaike Information Criterion) or 'threshold'

        Returns:
            Updated DetectedEvent with picks
        """
        n_channels, n_samples = data.shape
        window_samples = int(window * self.sampling_rate)

        # Window around event
        center_idx = int(event.time * self.sampling_rate)
        start_idx = max(0, center_idx - window_samples // 2)
        end_idx = min(n_samples, center_idx + window_samples // 2)

        picks = {}

        for ch in range(n_channels):
            trace = data[ch, start_idx:end_idx]

            if method == 'aic':
                pick_idx = self._aic_pick(trace)
            else:
                pick_idx = self._threshold_pick(trace)

            if pick_idx is not None:
                picks[ch] = (start_idx + pick_idx) / self.sampling_rate

        event.picks = picks
        return event

    def _aic_pick(self, trace: np.ndarray) -> Optional[int]:
        """AIC-based arrival time picking."""
        n = len(trace)
        if n < 10:
            return None

        aic = np.zeros(n)

        for k in range(1, n - 1):
            var1 = np.var(trace[:k]) if k > 0 else 1e-10
            var2 = np.var(trace[k:]) if k < n else 1e-10
            var1 = max(var1, 1e-10)
            var2 = max(var2, 1e-10)

            aic[k] = k * np.log(var1) + (n - k) * np.log(var2)

        # Find minimum AIC
        pick_idx = np.argmin(aic[1:-1]) + 1

        return pick_idx

    def _threshold_pick(self, trace: np.ndarray, threshold: float = 3.0) -> Optional[int]:
        """Threshold-based arrival time picking."""
        noise_level = np.std(trace[:len(trace)//4])
        if noise_level == 0:
            noise_level = 1e-10

        above_threshold = np.abs(trace) > threshold * noise_level

        if np.any(above_threshold):
            return np.argmax(above_threshold)
        return None


def detect_microseismic_events(data: np.ndarray,
                               sampling_rate: float = 1000.0,
                               method: str = 'sta_lta',
                               **kwargs) -> List[DetectedEvent]:
    """
    Convenience function to detect microseismic events.

    Parameters:
        data: DAS data [channels x time]
        sampling_rate: Sampling rate in Hz
        method: Detection method ('sta_lta', 'threshold', 'template')
        **kwargs: Additional arguments for the detection method

    Returns:
        List of DetectedEvent objects
    """
    detector = EventDetector(sampling_rate=sampling_rate)

    if method == 'sta_lta':
        return detector.sta_lta_detect(data, **kwargs)
    elif method == 'threshold':
        return detector.amplitude_threshold_detect(data, **kwargs)
    else:
        raise ValueError(f"Unknown detection method: {method}")


def estimate_event_location(event: DetectedEvent,
                            picks: Dict[int, float],
                            channel_spacing: float,
                            velocity: float = 2000.0) -> Tuple[float, float]:
    """
    Simple event location estimation from arrival times.

    Parameters:
        event: DetectedEvent with picks
        picks: Dictionary of channel: arrival_time
        channel_spacing: Distance between channels in meters
        velocity: Assumed velocity in m/s

    Returns:
        Tuple of (along-fiber distance, depth estimate) in meters
    """
    if len(picks) < 3:
        return (event.channel * channel_spacing, 0.0)

    channels = np.array(list(picks.keys()))
    times = np.array(list(picks.values()))

    # Find channel with earliest arrival
    min_time_idx = np.argmin(times)
    epicenter_channel = channels[min_time_idx]

    # Simple depth estimate from moveout
    moveouts = times - times[min_time_idx]
    distances = np.abs(channels - epicenter_channel) * channel_spacing

    # Avoid division by zero
    valid = distances > 0
    if np.sum(valid) > 0:
        avg_velocity = np.mean(distances[valid] / moveouts[valid])
        depth_estimate = avg_velocity * np.mean(moveouts[valid]) if avg_velocity > 0 else 0
    else:
        depth_estimate = 0

    return (epicenter_channel * channel_spacing, depth_estimate)
