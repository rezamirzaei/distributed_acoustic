"""
CO2 Storage Monitoring Module
=============================

Time-lapse analysis and monitoring tools for CO2 storage applications.

Includes:
- Baseline comparison
- Velocity change detection (dv/v)
- Strain accumulation monitoring
- Plume migration indicators
"""

import numpy as np
from scipy import signal
from scipy.ndimage import gaussian_filter
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import warnings


@dataclass
class MonitoringResult:
    """Container for time-lapse monitoring results."""
    timestamp: float  # Relative time since baseline
    strain_change: np.ndarray  # Strain change map
    velocity_change: Optional[np.ndarray]  # dv/v map
    anomaly_locations: List[Tuple[int, float]]  # (channel, anomaly_value)
    quality_metric: float  # Data quality indicator


class CO2Monitor:
    """
    CO2 storage monitoring using DAS time-lapse analysis.

    This class implements techniques for detecting subsurface changes
    related to CO2 injection and migration using repeated DAS measurements.

    Example usage:
        monitor = CO2Monitor()
        monitor.set_baseline(baseline_data)
        results = monitor.analyze_repeat(repeat_data)
    """

    def __init__(self,
                 sampling_rate: float = 1000.0,
                 channel_spacing: float = 1.0):
        """
        Initialize CO2 monitor.

        Parameters:
            sampling_rate: Temporal sampling rate in Hz
            channel_spacing: Spatial sampling in meters
        """
        self.sampling_rate = sampling_rate
        self.channel_spacing = channel_spacing
        self.baseline: Optional[np.ndarray] = None
        self.baseline_spectrum: Optional[np.ndarray] = None
        self.monitoring_results: List[MonitoringResult] = []

    def set_baseline(self,
                     data: np.ndarray,
                     preprocess: bool = True) -> 'CO2Monitor':
        """
        Set baseline (pre-injection) data.

        Parameters:
            data: Baseline DAS data [channels x time]
            preprocess: Apply standard preprocessing

        Returns:
            self for method chaining
        """
        self.baseline = data.copy()

        if preprocess:
            # Remove mean
            self.baseline = self.baseline - np.mean(self.baseline, axis=1, keepdims=True)

        # Store baseline spectrum for velocity change analysis
        self.baseline_spectrum = np.fft.rfft(self.baseline, axis=1)

        print(f"Baseline set: {self.baseline.shape}")
        return self

    def analyze_repeat(self,
                       data: np.ndarray,
                       timestamp: float = 0.0,
                       compute_velocity_change: bool = True) -> MonitoringResult:
        """
        Analyze repeat survey relative to baseline.

        Parameters:
            data: Repeat survey DAS data [channels x time]
            timestamp: Time relative to injection start
            compute_velocity_change: Compute dv/v analysis

        Returns:
            MonitoringResult object
        """
        if self.baseline is None:
            raise ValueError("Baseline not set. Call set_baseline() first.")

        # Ensure same shape
        min_channels = min(self.baseline.shape[0], data.shape[0])
        min_samples = min(self.baseline.shape[1], data.shape[1])

        baseline_trimmed = self.baseline[:min_channels, :min_samples]
        data_trimmed = data[:min_channels, :min_samples]

        # Remove mean
        data_trimmed = data_trimmed - np.mean(data_trimmed, axis=1, keepdims=True)

        # Calculate strain change
        strain_change = self._compute_strain_change(baseline_trimmed, data_trimmed)

        # Calculate velocity change
        velocity_change = None
        if compute_velocity_change:
            velocity_change = self._compute_velocity_change(baseline_trimmed, data_trimmed)

        # Find anomaly locations
        anomaly_locations = self._find_anomalies(strain_change)

        # Quality metric
        quality = self._compute_quality_metric(baseline_trimmed, data_trimmed)

        result = MonitoringResult(
            timestamp=timestamp,
            strain_change=strain_change,
            velocity_change=velocity_change,
            anomaly_locations=anomaly_locations,
            quality_metric=quality
        )

        self.monitoring_results.append(result)

        return result

    def _compute_strain_change(self,
                               baseline: np.ndarray,
                               repeat: np.ndarray) -> np.ndarray:
        """
        Compute strain change between baseline and repeat.

        Uses RMS difference to quantify changes.
        """
        # Time-domain difference
        diff = repeat - baseline

        # RMS of difference per channel
        rms_diff = np.sqrt(np.mean(diff**2, axis=1))

        # Normalize by baseline RMS
        baseline_rms = np.sqrt(np.mean(baseline**2, axis=1))
        baseline_rms[baseline_rms == 0] = 1e-10

        normalized_change = rms_diff / baseline_rms

        return normalized_change

    def _compute_velocity_change(self,
                                 baseline: np.ndarray,
                                 repeat: np.ndarray,
                                 freq_band: Tuple[float, float] = (1.0, 50.0),
                                 max_lag: float = 0.1) -> np.ndarray:
        """
        Estimate velocity change (dv/v) using cross-correlation stretching.

        dv/v is related to dt/t (time shift) which indicates velocity changes
        in the medium.

        Parameters:
            baseline: Baseline data
            repeat: Repeat data
            freq_band: Frequency band for analysis
            max_lag: Maximum lag in seconds

        Returns:
            Array of dv/v values per channel
        """
        n_channels, n_samples = baseline.shape
        max_lag_samples = int(max_lag * self.sampling_rate)

        dv_v = np.zeros(n_channels)

        for ch in range(n_channels):
            # Cross-correlation
            correlation = signal.correlate(repeat[ch, :], baseline[ch, :], mode='full')
            lags = signal.correlation_lags(n_samples, n_samples, mode='full')

            # Find peak near zero lag
            center = len(correlation) // 2
            search_range = slice(center - max_lag_samples, center + max_lag_samples + 1)

            if len(correlation[search_range]) > 0:
                local_corr = correlation[search_range]
                local_lags = lags[search_range]

                peak_idx = np.argmax(local_corr)
                dt = local_lags[peak_idx] / self.sampling_rate

                # dv/v ≈ -dt/t for small changes
                # Use dominant period as reference time
                t_ref = 0.1  # Reference time scale
                dv_v[ch] = -dt / t_ref if t_ref > 0 else 0

        return dv_v

    def _find_anomalies(self,
                        strain_change: np.ndarray,
                        threshold: float = 2.0) -> List[Tuple[int, float]]:
        """
        Find channels with anomalous strain changes.

        Parameters:
            strain_change: Array of strain changes per channel
            threshold: Number of standard deviations for anomaly

        Returns:
            List of (channel_index, anomaly_value) tuples
        """
        mean_change = np.mean(strain_change)
        std_change = np.std(strain_change)

        if std_change == 0:
            return []

        z_scores = (strain_change - mean_change) / std_change
        anomaly_mask = np.abs(z_scores) > threshold

        anomalies = [
            (int(ch), float(strain_change[ch]))
            for ch in np.where(anomaly_mask)[0]
        ]

        return anomalies

    def _compute_quality_metric(self,
                                baseline: np.ndarray,
                                repeat: np.ndarray) -> float:
        """
        Compute data quality metric based on correlation.

        Returns value between 0 (poor) and 1 (excellent).
        """
        # Average correlation across channels
        correlations = []

        for ch in range(baseline.shape[0]):
            corr = np.corrcoef(baseline[ch, :], repeat[ch, :])[0, 1]
            if not np.isnan(corr):
                correlations.append(corr)

        if correlations:
            return float(np.mean(correlations))
        return 0.0

    def get_strain_evolution(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get strain change evolution over time.

        Returns:
            Tuple of (timestamps, strain_changes) where strain_changes
            is [n_times x n_channels]
        """
        if not self.monitoring_results:
            raise ValueError("No monitoring results available")

        timestamps = np.array([r.timestamp for r in self.monitoring_results])
        strain_changes = np.array([r.strain_change for r in self.monitoring_results])

        return timestamps, strain_changes

    def detect_plume_migration(self,
                               depth_range: Optional[Tuple[float, float]] = None
                               ) -> Dict[str, Any]:
        """
        Analyze strain change patterns for CO2 plume migration indicators.

        Parameters:
            depth_range: Depth range of interest in meters (mapped to channels)

        Returns:
            Dictionary with plume migration analysis results
        """
        if len(self.monitoring_results) < 2:
            warnings.warn("Need at least 2 monitoring surveys for migration analysis")
            return {}

        timestamps, strain_changes = self.get_strain_evolution()

        # Find center of mass of strain anomaly over time
        n_times, n_channels = strain_changes.shape
        channel_indices = np.arange(n_channels)

        centroids = []
        spreads = []

        for t in range(n_times):
            # Weight by squared strain change (emphasize larger changes)
            weights = strain_changes[t, :] ** 2
            total_weight = np.sum(weights)

            if total_weight > 0:
                centroid = np.sum(weights * channel_indices) / total_weight
                spread = np.sqrt(np.sum(weights * (channel_indices - centroid)**2) / total_weight)
            else:
                centroid = n_channels / 2
                spread = 0

            centroids.append(centroid)
            spreads.append(spread)

        centroids = np.array(centroids)
        spreads = np.array(spreads)

        # Calculate migration rate
        if len(timestamps) > 1 and np.std(timestamps) > 0:
            migration_rate = np.polyfit(timestamps, centroids * self.channel_spacing, 1)[0]
            spread_rate = np.polyfit(timestamps, spreads * self.channel_spacing, 1)[0]
        else:
            migration_rate = 0
            spread_rate = 0

        return {
            'timestamps': timestamps,
            'centroid_positions': centroids * self.channel_spacing,
            'plume_spread': spreads * self.channel_spacing,
            'migration_rate': migration_rate,  # m/s or m/day depending on timestamp units
            'spread_rate': spread_rate,
            'total_migration': (centroids[-1] - centroids[0]) * self.channel_spacing if len(centroids) > 1 else 0
        }

    def generate_report(self) -> str:
        """Generate a text summary of monitoring results."""
        if not self.monitoring_results:
            return "No monitoring data available."

        lines = [
            "=" * 60,
            "CO2 STORAGE MONITORING REPORT",
            "=" * 60,
            "",
            f"Number of surveys: {len(self.monitoring_results)}",
            f"Baseline shape: {self.baseline.shape if self.baseline is not None else 'Not set'}",
            "",
            "SURVEY SUMMARY:",
            "-" * 40,
        ]

        for i, result in enumerate(self.monitoring_results):
            lines.extend([
                f"\nSurvey {i+1} (t = {result.timestamp:.2f}):",
                f"  Quality metric: {result.quality_metric:.3f}",
                f"  Mean strain change: {np.mean(result.strain_change):.4f}",
                f"  Max strain change: {np.max(result.strain_change):.4f}",
                f"  Anomaly count: {len(result.anomaly_locations)}",
            ])

            if result.velocity_change is not None:
                lines.append(f"  Mean dv/v: {np.mean(result.velocity_change)*100:.2f}%")

        # Migration analysis if available
        if len(self.monitoring_results) >= 2:
            migration = self.detect_plume_migration()
            lines.extend([
                "",
                "MIGRATION ANALYSIS:",
                "-" * 40,
                f"Migration rate: {migration['migration_rate']:.2f} m/time_unit",
                f"Spread rate: {migration['spread_rate']:.2f} m/time_unit",
                f"Total migration: {migration['total_migration']:.1f} m",
            ])

        lines.append("\n" + "=" * 60)

        return "\n".join(lines)


def simulate_co2_injection_effect(
    baseline_data: np.ndarray,
    injection_channel: int,
    effect_radius: int = 50,
    strain_increase: float = 0.2,
    velocity_decrease: float = 0.05
) -> np.ndarray:
    """
    Simulate the effect of CO2 injection on DAS data.

    This creates synthetic repeat survey data with realistic CO2-induced changes.

    Parameters:
        baseline_data: Original baseline DAS data
        injection_channel: Channel corresponding to injection zone
        effect_radius: Radius of effect in channels
        strain_increase: Fractional strain increase at injection point
        velocity_decrease: Fractional velocity decrease (causes time shift)

    Returns:
        Modified data simulating post-injection survey
    """
    repeat_data = baseline_data.copy()
    n_channels, n_samples = repeat_data.shape

    # Create Gaussian-shaped effect zone
    channels = np.arange(n_channels)
    effect = np.exp(-((channels - injection_channel) ** 2) / (2 * effect_radius ** 2))

    # Apply strain increase
    for ch in range(n_channels):
        repeat_data[ch, :] *= (1 + strain_increase * effect[ch])

    # Apply time stretch (velocity decrease causes time dilation)
    for ch in range(n_channels):
        stretch_factor = 1 + velocity_decrease * effect[ch]
        if stretch_factor != 1.0:
            # Resample to simulate velocity change
            old_times = np.arange(n_samples)
            new_times = old_times / stretch_factor
            repeat_data[ch, :] = np.interp(old_times, new_times, repeat_data[ch, :])

    # Add some noise
    noise_level = 0.01 * np.std(baseline_data)
    repeat_data += noise_level * np.random.randn(*repeat_data.shape)

    return repeat_data


def create_monitoring_scenario(
    n_channels: int = 500,
    n_samples: int = 10000,
    sampling_rate: float = 1000.0,
    n_surveys: int = 5,
    injection_channel: int = 250
) -> Tuple[np.ndarray, List[np.ndarray], List[float]]:
    """
    Create a complete CO2 monitoring scenario with baseline and repeat surveys.

    Parameters:
        n_channels: Number of DAS channels
        n_samples: Number of time samples
        sampling_rate: Sampling rate in Hz
        n_surveys: Number of repeat surveys
        injection_channel: Channel at injection location

    Returns:
        Tuple of (baseline_data, list_of_repeat_surveys, timestamps)
    """
    from .data_loader import generate_synthetic_das_data

    # Generate baseline
    baseline_loader = generate_synthetic_das_data(
        n_channels=n_channels,
        n_samples=n_samples,
        sampling_rate=sampling_rate,
        n_events=3,
        noise_level=0.05,
        seed=42
    )
    baseline_data = baseline_loader.data

    # Generate repeat surveys with progressive CO2 effects
    repeat_surveys = []
    timestamps = []

    for i in range(n_surveys):
        # Progressive effect over time
        time_factor = (i + 1) / n_surveys
        timestamps.append(time_factor * 100)  # Arbitrary time units (e.g., days)

        repeat = simulate_co2_injection_effect(
            baseline_data,
            injection_channel=injection_channel,
            effect_radius=30 + int(20 * time_factor),  # Growing plume
            strain_increase=0.1 * time_factor,
            velocity_decrease=0.03 * time_factor
        )
        repeat_surveys.append(repeat)

    print(f"Created monitoring scenario:")
    print(f"  Baseline: {baseline_data.shape}")
    print(f"  {n_surveys} repeat surveys")
    print(f"  Injection at channel {injection_channel}")

    return baseline_data, repeat_surveys, timestamps
