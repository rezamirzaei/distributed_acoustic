"""
Real DAS Data Example: Loading and Processing Realistic Field-Parameter Data
===========================================================================

This example demonstrates an end-to-end workflow using the *real sample*
datasets shipped with this repository under `data/real/`.

The sample bundles are generated from real-world DAS acquisition parameters
(PoroTomo-like) and include realistic coherent noise + events.

If the NPZ files aren't present, they will be generated automatically.

For truly raw upstream public data (multi-GB HDF5), see:
- PoroTomo: https://gdr.openei.org/submissions/980
- FORGE Utah: https://gdr.openei.org/submissions/1318
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from das_co2_monitoring import (
    DASDataLoader,
    DASPreprocessor,
    EventDetector,
    DASVisualizer,
)
from das_co2_monitoring.data_loader import download_sample_data


def process_real_das_data():
    """
    Complete workflow for processing a real sample DAS dataset.

    Returns:
        (loader, processed_data, events)
    """

    print("\n" + "="*70)
    print("DAS DATA PROCESSING WORKFLOW FOR CO2 MONITORING")
    print("="*70)

    # ================================================================
    # STEP 1: Data Loading (real sample)
    # ================================================================
    print("\n[1] LOADING REAL SAMPLE DATA")
    print("-" * 40)

    npz_path = download_sample_data(dataset="porotomo_sample")
    loader = DASDataLoader().load_numpy(npz_path)

    # If the npz includes acquisition parameters, propagate them.
    with np.load(npz_path) as z:
        if "sampling_rate" in z:
            loader.sampling_rate = float(z["sampling_rate"])
        if "channel_spacing" in z:
            loader.channel_spacing = float(z["channel_spacing"])
        if "gauge_length" in z:
            loader.gauge_length = float(z["gauge_length"])

    print(loader.info())

    # ================================================================
    # STEP 2: Quality Control & Initial Inspection
    # ================================================================
    print("\n[2] QUALITY CONTROL")
    print("-" * 40)

    raw_data = loader.data

    # Check for bad channels (dead or noisy)
    channel_rms = np.sqrt(np.mean(raw_data**2, axis=1))
    median_rms = np.median(channel_rms)

    dead_channels = np.where(channel_rms < 0.1 * median_rms)[0]
    noisy_channels = np.where(channel_rms > 10 * median_rms)[0]

    print(f"Total channels: {len(channel_rms)}")
    print(f"Dead channels (low amplitude): {len(dead_channels)}")
    print(f"Noisy channels (high amplitude): {len(noisy_channels)}")
    print(f"Data range: [{raw_data.min():.3e}, {raw_data.max():.3e}]")

    # ================================================================
    # STEP 3: Preprocessing
    # ================================================================
    print("\n[3] PREPROCESSING")
    print("-" * 40)

    preprocessor = DASPreprocessor(
        sampling_rate=loader.sampling_rate,
        channel_spacing=loader.channel_spacing
    )

    processed_data = (preprocessor
        .set_data(raw_data)
        .remove_mean()
        .remove_trend(order=1)
        .bandpass_filter(2.0, 80.0)
        .median_denoise(kernel_size=(1, 5))
        .normalize(method='std')
        .get_data()
    )

    print(f"Processing steps: {preprocessor.get_history()}")

    # ================================================================
    # STEP 4: Event Detection
    # ================================================================
    print("\n[4] EVENT DETECTION")
    print("-" * 40)

    detector = EventDetector(
        sampling_rate=loader.sampling_rate,
        channel_spacing=loader.channel_spacing
    )

    events = detector.sta_lta_detect(
        processed_data,
        sta_window=0.03,
        lta_window=0.5,
        trigger_on=3.0,
        trigger_off=1.5,
        min_channels=15,
        min_duration=0.02
    )

    print(f"\nDetected {len(events)} events")

    # ================================================================
    # STEP 5: Visualization
    # ================================================================
    print("\n[5] GENERATING VISUALIZATIONS")
    print("-" * 40)

    output_dir = Path('output')
    output_dir.mkdir(exist_ok=True)

    viz = DASVisualizer(figsize=(14, 10))

    fig = viz.waterfall_plot(
        processed_data,
        loader.time,
        loader.distance,
        title=f'Processed DAS Data ({len(events)} events detected)',
        events=events
    )
    plt.savefig(output_dir / 'real_data_waterfall.png', dpi=150)
    plt.close()

    fig = viz.fk_spectrum(
        processed_data,
        sampling_rate=loader.sampling_rate,
        channel_spacing=loader.channel_spacing,
        title='F-K Spectrum',
        freq_max=80,
        velocity_lines=[1500, 2500, 3500]
    )
    plt.savefig(output_dir / 'real_data_fk.png', dpi=150)
    plt.close()

    if events:
        fig = viz.event_catalog_plot(events, loader.distance)
        plt.savefig(output_dir / 'real_data_catalog.png', dpi=150)
        plt.close()

    print("\n" + "="*70)
    print("PROCESSING COMPLETE")
    print("="*70)
    print(f"\nOutput files saved to: {output_dir.absolute()}")

    return loader, processed_data, events


if __name__ == '__main__':
    process_real_das_data()
