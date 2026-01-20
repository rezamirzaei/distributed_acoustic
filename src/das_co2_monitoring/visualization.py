"""
DAS Visualization Module
========================

Visualization tools for DAS data analysis and presentation.

Includes:
- Waterfall plots (distance-time)
- FK spectra plots
- Event catalogs
- Time-lapse comparisons
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.patches import Rectangle
import matplotlib.dates as mdates
from typing import List, Optional, Tuple, Union
from datetime import datetime, timedelta

# Import local event type
from .event_detection import DetectedEvent


class DASVisualizer:
    """
    Visualization tools for DAS data.

    Example usage:
        viz = DASVisualizer()
        viz.waterfall_plot(data, time, distance)
        viz.fk_spectrum(data, sampling_rate, channel_spacing)
    """

    def __init__(self, figsize: Tuple[int, int] = (12, 8), style: str = 'default'):
        """
        Initialize visualizer.

        Parameters:
            figsize: Default figure size
            style: Matplotlib style to use
        """
        self.figsize = figsize
        if style != 'default':
            plt.style.use(style)

        # DAS-specific colormap (blue-white-red for strain)
        self.strain_cmap = 'seismic'
        self.amplitude_cmap = 'viridis'

    def waterfall_plot(self,
                       data: np.ndarray,
                       time: Optional[np.ndarray] = None,
                       distance: Optional[np.ndarray] = None,
                       sampling_rate: float = 1000.0,
                       channel_spacing: float = 1.0,
                       title: str = 'DAS Waterfall Plot',
                       clim: Optional[Tuple[float, float]] = None,
                       clim_percentile: float = 99,
                       cmap: str = 'seismic',
                       events: Optional[List[DetectedEvent]] = None,
                       ax: Optional[plt.Axes] = None,
                       colorbar: bool = True) -> plt.Figure:
        """
        Create a DAS waterfall plot (distance vs. time).

        Parameters:
            data: DAS data [channels x time]
            time: Time vector in seconds
            distance: Distance vector in meters
            sampling_rate: Sampling rate if time not provided
            channel_spacing: Channel spacing if distance not provided
            title: Plot title
            clim: Color limits (min, max)
            clim_percentile: Percentile for automatic clim
            cmap: Colormap
            events: List of detected events to mark
            ax: Existing axes to plot on
            colorbar: Whether to show colorbar

        Returns:
            matplotlib Figure
        """
        n_channels, n_samples = data.shape

        # Generate axes if not provided
        if time is None:
            time = np.arange(n_samples) / sampling_rate
        if distance is None:
            distance = np.arange(n_channels) * channel_spacing

        # Create figure if needed
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
        else:
            fig = ax.figure

        # Determine color limits
        if clim is None:
            vmax = np.percentile(np.abs(data), clim_percentile)
            clim = (-vmax, vmax)

        # Create waterfall plot
        extent = [time[0], time[-1], distance[-1], distance[0]]
        im = ax.imshow(data, aspect='auto', extent=extent,
                       cmap=cmap, vmin=clim[0], vmax=clim[1])

        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel('Distance along fiber (m)', fontsize=12)
        ax.set_title(title, fontsize=14)

        if colorbar:
            cbar = plt.colorbar(im, ax=ax, label='Strain rate (or strain)')

        # Mark detected events
        if events:
            for event in events:
                event_dist = distance[event.channel] if event.channel < len(distance) else 0
                ax.axvline(event.time, color='lime', linestyle='--', alpha=0.7, linewidth=1)
                ax.plot(event.time, event_dist, 'g*', markersize=10)

        plt.tight_layout()
        return fig

    def multi_panel_waterfall(self,
                              data: np.ndarray,
                              time: np.ndarray,
                              distance: np.ndarray,
                              time_windows: List[Tuple[float, float]],
                              titles: Optional[List[str]] = None,
                              **kwargs) -> plt.Figure:
        """
        Create multiple waterfall plots for different time windows.

        Parameters:
            data: DAS data [channels x time]
            time: Time vector
            distance: Distance vector
            time_windows: List of (start, end) time tuples
            titles: Titles for each panel
            **kwargs: Additional arguments passed to waterfall_plot

        Returns:
            matplotlib Figure
        """
        n_panels = len(time_windows)
        fig, axes = plt.subplots(1, n_panels, figsize=(5*n_panels, 8))

        if n_panels == 1:
            axes = [axes]

        for i, (t_start, t_end) in enumerate(time_windows):
            # Find indices
            t_mask = (time >= t_start) & (time <= t_end)
            time_subset = time[t_mask]
            data_subset = data[:, t_mask]

            title = titles[i] if titles else f'{t_start:.2f}s - {t_end:.2f}s'

            self.waterfall_plot(data_subset, time_subset, distance,
                               title=title, ax=axes[i], colorbar=(i == n_panels-1),
                               **kwargs)

        plt.tight_layout()
        return fig

    def fk_spectrum(self,
                    data: np.ndarray,
                    sampling_rate: float = 1000.0,
                    channel_spacing: float = 1.0,
                    title: str = 'F-K Spectrum',
                    freq_max: Optional[float] = None,
                    velocity_lines: Optional[List[float]] = None,
                    ax: Optional[plt.Axes] = None,
                    log_scale: bool = True) -> plt.Figure:
        """
        Plot the frequency-wavenumber (f-k) spectrum.

        Parameters:
            data: DAS data [channels x time]
            sampling_rate: Temporal sampling rate in Hz
            channel_spacing: Spatial sampling in meters
            title: Plot title
            freq_max: Maximum frequency to display
            velocity_lines: List of velocities to draw as reference lines
            ax: Existing axes
            log_scale: Use logarithmic color scale

        Returns:
            matplotlib Figure
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
        else:
            fig = ax.figure

        n_channels, n_samples = data.shape

        # 2D FFT
        fk_spectrum = np.fft.fft2(data)
        fk_spectrum = np.fft.fftshift(fk_spectrum)
        fk_amplitude = np.abs(fk_spectrum)

        # Frequency and wavenumber axes
        freqs = np.fft.fftshift(np.fft.fftfreq(n_samples, 1/sampling_rate))
        wavenumbers = np.fft.fftshift(np.fft.fftfreq(n_channels, channel_spacing))

        # Limit frequency range
        if freq_max is None:
            freq_max = sampling_rate / 2
        freq_mask = np.abs(freqs) <= freq_max

        fk_display = fk_amplitude[:, freq_mask]
        freqs_display = freqs[freq_mask]

        # Plot
        extent = [freqs_display[0], freqs_display[-1],
                  wavenumbers[-1], wavenumbers[0]]

        if log_scale:
            fk_display = np.log10(fk_display + 1e-10)

        im = ax.imshow(fk_display, aspect='auto', extent=extent, cmap='magma')

        ax.set_xlabel('Frequency (Hz)', fontsize=12)
        ax.set_ylabel('Wavenumber (1/m)', fontsize=12)
        ax.set_title(title, fontsize=14)

        # Draw velocity reference lines
        if velocity_lines:
            for v in velocity_lines:
                # k = f / v
                f_line = np.linspace(-freq_max, freq_max, 100)
                k_line = f_line / v
                ax.plot(f_line, k_line, 'w--', alpha=0.5, label=f'{v} m/s')
                ax.plot(f_line, -k_line, 'w--', alpha=0.5)
            ax.legend(loc='upper right')

        plt.colorbar(im, ax=ax, label='Log amplitude' if log_scale else 'Amplitude')
        plt.tight_layout()
        return fig

    def channel_gather(self,
                       data: np.ndarray,
                       channels: List[int],
                       time: Optional[np.ndarray] = None,
                       sampling_rate: float = 1000.0,
                       normalize: bool = True,
                       offset: float = 1.0,
                       title: str = 'Channel Gather',
                       ax: Optional[plt.Axes] = None) -> plt.Figure:
        """
        Plot individual channel waveforms (seismic gather style).

        Parameters:
            data: DAS data [channels x time]
            channels: List of channel indices to plot
            time: Time vector
            sampling_rate: Sampling rate if time not provided
            normalize: Normalize each trace
            offset: Vertical offset between traces
            title: Plot title
            ax: Existing axes

        Returns:
            matplotlib Figure
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
        else:
            fig = ax.figure

        n_samples = data.shape[1]
        if time is None:
            time = np.arange(n_samples) / sampling_rate

        for i, ch in enumerate(channels):
            trace = data[ch, :]

            if normalize:
                max_amp = np.max(np.abs(trace))
                if max_amp > 0:
                    trace = trace / max_amp

            ax.plot(time, trace + i * offset, 'k-', linewidth=0.5)
            ax.text(time[0] - 0.02 * (time[-1] - time[0]), i * offset,
                   f'Ch {ch}', fontsize=8, ha='right', va='center')

        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel('Channel', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.set_yticks([])

        plt.tight_layout()
        return fig

    def event_catalog_plot(self,
                           events: List[DetectedEvent],
                           distance: Optional[np.ndarray] = None,
                           channel_spacing: float = 1.0,
                           title: str = 'Event Catalog') -> plt.Figure:
        """
        Plot detected events catalog.

        Parameters:
            events: List of DetectedEvent objects
            distance: Distance vector for channel-to-distance conversion
            channel_spacing: Channel spacing if distance not provided
            title: Plot title

        Returns:
            matplotlib Figure
        """
        if not events:
            print("No events to plot")
            return None

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Extract event properties
        times = [e.time for e in events]
        channels = [e.channel for e in events]
        amplitudes = [e.amplitude for e in events]
        snrs = [e.snr for e in events]

        if distance is not None:
            distances = [distance[min(e.channel, len(distance)-1)] for e in events]
        else:
            distances = [e.channel * channel_spacing for e in events]

        # Time vs Distance
        ax = axes[0, 0]
        scatter = ax.scatter(times, distances, c=snrs, cmap='viridis',
                            s=50, alpha=0.7)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Distance (m)')
        ax.set_title('Event Locations')
        plt.colorbar(scatter, ax=ax, label='SNR')

        # Amplitude histogram
        ax = axes[0, 1]
        ax.hist(amplitudes, bins=20, edgecolor='black', alpha=0.7)
        ax.set_xlabel('Amplitude')
        ax.set_ylabel('Count')
        ax.set_title('Amplitude Distribution')

        # SNR histogram
        ax = axes[1, 0]
        ax.hist(snrs, bins=20, edgecolor='black', alpha=0.7, color='orange')
        ax.set_xlabel('Signal-to-Noise Ratio')
        ax.set_ylabel('Count')
        ax.set_title('SNR Distribution')

        # Events over time (cumulative)
        ax = axes[1, 1]
        ax.plot(sorted(times), range(1, len(times) + 1), 'b-', linewidth=2)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Cumulative Event Count')
        ax.set_title('Event Rate')
        ax.grid(True, alpha=0.3)

        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        return fig

    def comparison_plot(self,
                        data_before: np.ndarray,
                        data_after: np.ndarray,
                        time: np.ndarray,
                        distance: np.ndarray,
                        titles: Tuple[str, str] = ('Before', 'After'),
                        show_difference: bool = True) -> plt.Figure:
        """
        Create comparison plot for time-lapse analysis.

        Parameters:
            data_before: First dataset
            data_after: Second dataset
            time: Time vector
            distance: Distance vector
            titles: Titles for the two datasets
            show_difference: Whether to show difference panel

        Returns:
            matplotlib Figure
        """
        n_plots = 3 if show_difference else 2
        fig, axes = plt.subplots(1, n_plots, figsize=(5*n_plots, 8))

        # Determine common color scale
        vmax = max(np.percentile(np.abs(data_before), 99),
                   np.percentile(np.abs(data_after), 99))

        extent = [time[0], time[-1], distance[-1], distance[0]]

        # Before
        im0 = axes[0].imshow(data_before, aspect='auto', extent=extent,
                            cmap='seismic', vmin=-vmax, vmax=vmax)
        axes[0].set_xlabel('Time (s)')
        axes[0].set_ylabel('Distance (m)')
        axes[0].set_title(titles[0])

        # After
        im1 = axes[1].imshow(data_after, aspect='auto', extent=extent,
                            cmap='seismic', vmin=-vmax, vmax=vmax)
        axes[1].set_xlabel('Time (s)')
        axes[1].set_ylabel('Distance (m)')
        axes[1].set_title(titles[1])

        if show_difference:
            # Difference
            diff = data_after - data_before
            diff_vmax = np.percentile(np.abs(diff), 99)

            im2 = axes[2].imshow(diff, aspect='auto', extent=extent,
                                cmap='seismic', vmin=-diff_vmax, vmax=diff_vmax)
            axes[2].set_xlabel('Time (s)')
            axes[2].set_ylabel('Distance (m)')
            axes[2].set_title('Difference')
            plt.colorbar(im2, ax=axes[2], label='Strain difference')

        plt.colorbar(im0, ax=axes[0], label='Strain')
        plt.colorbar(im1, ax=axes[1], label='Strain')

        plt.tight_layout()
        return fig

    def spectrum_plot(self,
                      data: np.ndarray,
                      sampling_rate: float = 1000.0,
                      channels: Optional[List[int]] = None,
                      title: str = 'Frequency Spectrum',
                      ax: Optional[plt.Axes] = None) -> plt.Figure:
        """
        Plot frequency spectrum of selected channels.

        Parameters:
            data: DAS data [channels x time]
            sampling_rate: Sampling rate in Hz
            channels: Channels to plot (None = average all)
            title: Plot title
            ax: Existing axes

        Returns:
            matplotlib Figure
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        else:
            fig = ax.figure

        n_samples = data.shape[1]
        freqs = np.fft.rfftfreq(n_samples, 1/sampling_rate)

        if channels is None:
            # Average spectrum across all channels
            spectrum = np.mean(np.abs(np.fft.rfft(data, axis=1)), axis=0)
            ax.semilogy(freqs, spectrum, 'b-', linewidth=1, label='Average')
        else:
            for ch in channels:
                spectrum = np.abs(np.fft.rfft(data[ch, :]))
                ax.semilogy(freqs, spectrum, linewidth=0.5, alpha=0.7, label=f'Ch {ch}')

        ax.set_xlabel('Frequency (Hz)', fontsize=12)
        ax.set_ylabel('Amplitude', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend()

        plt.tight_layout()
        return fig


def quick_plot(data: np.ndarray,
               sampling_rate: float = 1000.0,
               channel_spacing: float = 1.0,
               title: str = 'DAS Data') -> plt.Figure:
    """
    Quick visualization of DAS data.

    Parameters:
        data: DAS data [channels x time]
        sampling_rate: Sampling rate in Hz
        channel_spacing: Channel spacing in meters
        title: Plot title

    Returns:
        matplotlib Figure
    """
    viz = DASVisualizer()
    return viz.waterfall_plot(data,
                              sampling_rate=sampling_rate,
                              channel_spacing=channel_spacing,
                              title=title)


def save_figure(fig: plt.Figure,
                filename: str,
                dpi: int = 150,
                formats: List[str] = ['png']) -> None:
    """
    Save figure in multiple formats.

    Parameters:
        fig: matplotlib Figure
        filename: Base filename (without extension)
        dpi: Resolution
        formats: List of formats ('png', 'pdf', 'svg')
    """
    for fmt in formats:
        fig.savefig(f'{filename}.{fmt}', dpi=dpi, bbox_inches='tight')
        print(f'Saved {filename}.{fmt}')
