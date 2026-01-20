"""
Distributed Acoustic Sensing (DAS) for CO2 Storage Monitoring
=============================================================

This project demonstrates DAS data processing and analysis techniques
for monitoring CO2 sequestration activities.

Key capabilities:
- Load and process DAS data (real or synthetic)
- Apply standard preprocessing (filtering, denoising)
- Detect microseismic events
- Perform time-lapse analysis for CO2 plume monitoring
- Visualize results with DAS-specific plots

Author: DAS Research Team
Application: CO2 Storage Monitoring / Carbon Capture and Storage (CCS)
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add source directory to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from das_co2_monitoring import (
    DASDataLoader,
    DASPreprocessor,
    EventDetector,
    DASVisualizer,
    CO2Monitor,
    download_sample_data
)
from das_co2_monitoring.data_loader import generate_synthetic_das_data, print_porotomo_info
from das_co2_monitoring.preprocessing import preprocess_for_event_detection, preprocess_for_co2_monitoring
from das_co2_monitoring.event_detection import detect_microseismic_events
from das_co2_monitoring.monitoring import create_monitoring_scenario, simulate_co2_injection_effect


def demo_basic_das_processing():
    """
    Demonstrate basic DAS data loading and processing.
    """
    print("\n" + "="*60)
    print("DEMO 1: Basic DAS Data Processing")
    print("="*60 + "\n")

    # Generate synthetic DAS data
    print("Generating synthetic DAS data...")
    loader = generate_synthetic_das_data(
        n_channels=500,
        n_samples=15000,
        sampling_rate=1000.0,
        channel_spacing=1.0,
        n_events=8,
        noise_level=0.1,
        seed=42
    )

    print("\n" + loader.info())

    # Get raw data
    raw_data = loader.data
    time = loader.time
    distance = loader.distance

    # Apply preprocessing
    print("\nApplying preprocessing pipeline...")
    preprocessor = DASPreprocessor(sampling_rate=loader.sampling_rate)

    processed_data = (preprocessor
        .set_data(raw_data)
        .remove_mean()
        .bandpass_filter(5.0, 100.0)
        .median_denoise(kernel_size=(1, 5))
        .normalize(method='std')
        .get_data())

    print(f"Processing history: {preprocessor.get_history()}")

    # Visualize
    viz = DASVisualizer(figsize=(14, 10))

    # Create comparison plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    viz.waterfall_plot(raw_data, time, distance,
                       title='Raw DAS Data', ax=axes[0], colorbar=False)
    viz.waterfall_plot(processed_data, time, distance,
                       title='Processed DAS Data', ax=axes[1])

    plt.tight_layout()
    plt.savefig('output/demo1_preprocessing.png', dpi=150)
    print("\nSaved: output/demo1_preprocessing.png")
    plt.show()

    return loader, processed_data


def demo_event_detection(loader, processed_data):
    """
    Demonstrate microseismic event detection.
    """
    print("\n" + "="*60)
    print("DEMO 2: Microseismic Event Detection")
    print("="*60 + "\n")

    # Create detector
    detector = EventDetector(
        sampling_rate=loader.sampling_rate,
        channel_spacing=loader.channel_spacing
    )

    # Detect events using STA/LTA
    print("Running STA/LTA detection...")
    events_stalta = detector.sta_lta_detect(
        processed_data,
        sta_window=0.02,
        lta_window=0.3,
        trigger_on=2.5,
        trigger_off=1.0,
        min_channels=10
    )

    # Also try amplitude threshold
    print("\nRunning amplitude threshold detection...")
    events_threshold = detector.amplitude_threshold_detect(
        processed_data,
        threshold=3.0,
        min_channels=5
    )

    # Print detected events
    print("\nDetected Events (STA/LTA):")
    print("-" * 50)
    for event in events_stalta[:10]:  # Show first 10
        print(f"  Event {event.event_id}: t={event.time:.3f}s, "
              f"ch={event.channel}, amp={event.amplitude:.3f}, "
              f"SNR={event.snr:.1f}")

    # Visualize with events
    viz = DASVisualizer(figsize=(14, 8))
    fig = viz.waterfall_plot(
        processed_data,
        loader.time,
        loader.distance,
        title=f'DAS Data with Detected Events (n={len(events_stalta)})',
        events=events_stalta
    )

    plt.savefig('output/demo2_events.png', dpi=150)
    print("\nSaved: output/demo2_events.png")
    plt.show()

    # Event catalog
    fig = viz.event_catalog_plot(events_stalta, loader.distance)
    plt.savefig('output/demo2_catalog.png', dpi=150)
    print("Saved: output/demo2_catalog.png")
    plt.show()

    return events_stalta


def demo_fk_analysis(loader, processed_data):
    """
    Demonstrate frequency-wavenumber (F-K) analysis.
    """
    print("\n" + "="*60)
    print("DEMO 3: F-K Spectrum Analysis")
    print("="*60 + "\n")

    viz = DASVisualizer(figsize=(12, 8))

    # F-K spectrum
    print("Computing F-K spectrum...")
    fig = viz.fk_spectrum(
        processed_data,
        sampling_rate=loader.sampling_rate,
        channel_spacing=loader.channel_spacing,
        title='F-K Spectrum of DAS Data',
        freq_max=100,
        velocity_lines=[1500, 2000, 3000]  # Reference velocities
    )

    plt.savefig('output/demo3_fk_spectrum.png', dpi=150)
    print("\nSaved: output/demo3_fk_spectrum.png")
    plt.show()

    # Frequency spectrum
    print("Computing frequency spectrum...")
    fig = viz.spectrum_plot(
        processed_data,
        sampling_rate=loader.sampling_rate,
        channels=[100, 200, 300, 400],  # Selected channels
        title='Frequency Spectrum of Selected Channels'
    )

    plt.savefig('output/demo3_frequency_spectrum.png', dpi=150)
    print("Saved: output/demo3_frequency_spectrum.png")
    plt.show()


def demo_co2_monitoring():
    """
    Demonstrate CO2 storage monitoring with time-lapse analysis.
    """
    print("\n" + "="*60)
    print("DEMO 4: CO2 Storage Monitoring")
    print("="*60 + "\n")

    # Create monitoring scenario
    print("Creating CO2 injection monitoring scenario...")
    baseline_data, repeat_surveys, timestamps = create_monitoring_scenario(
        n_channels=500,
        n_samples=10000,
        sampling_rate=1000.0,
        n_surveys=5,
        injection_channel=250
    )

    # Initialize monitor
    monitor = CO2Monitor(sampling_rate=1000.0, channel_spacing=1.0)
    monitor.set_baseline(baseline_data)

    # Analyze each repeat survey
    print("\nAnalyzing repeat surveys...")
    for i, (repeat, timestamp) in enumerate(zip(repeat_surveys, timestamps)):
        print(f"  Survey {i+1}: t = {timestamp:.1f} days")
        result = monitor.analyze_repeat(repeat, timestamp=timestamp)
        print(f"    Quality: {result.quality_metric:.3f}, "
              f"Anomalies: {len(result.anomaly_locations)}")

    # Generate report
    print("\n" + monitor.generate_report())

    # Plume migration analysis
    migration = monitor.detect_plume_migration()
    print("\nPlume Migration Analysis:")
    print(f"  Migration rate: {migration['migration_rate']:.2f} m/day")
    print(f"  Spread rate: {migration['spread_rate']:.2f} m/day")
    print(f"  Total migration: {migration['total_migration']:.1f} m")

    # Visualize time-lapse changes
    viz = DASVisualizer(figsize=(16, 10))

    # Compare first and last survey
    time = np.arange(baseline_data.shape[1]) / 1000.0
    distance = np.arange(baseline_data.shape[0]) * 1.0

    fig = viz.comparison_plot(
        baseline_data,
        repeat_surveys[-1],
        time,
        distance,
        titles=('Baseline', f'After {timestamps[-1]:.0f} days'),
        show_difference=True
    )

    plt.savefig('output/demo4_timelapse.png', dpi=150)
    print("\nSaved: output/demo4_timelapse.png")
    plt.show()

    # Plot strain evolution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Strain change over time
    ts, strain_changes = monitor.get_strain_evolution()
    ax = axes[0]
    im = ax.imshow(strain_changes.T, aspect='auto', cmap='hot',
                   extent=[ts[0], ts[-1], distance[-1], distance[0]])
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Distance (m)')
    ax.set_title('Strain Change Evolution')
    plt.colorbar(im, ax=ax, label='Relative strain change')

    # Plume centroid over time
    ax = axes[1]
    ax.plot(migration['timestamps'], migration['centroid_positions'], 'b-o', linewidth=2)
    ax.fill_between(
        migration['timestamps'],
        np.array(migration['centroid_positions']) - np.array(migration['plume_spread']),
        np.array(migration['centroid_positions']) + np.array(migration['plume_spread']),
        alpha=0.3
    )
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Position along fiber (m)')
    ax.set_title('CO2 Plume Migration')
    ax.grid(True, alpha=0.3)
    ax.legend(['Centroid', 'Plume extent'])

    plt.tight_layout()
    plt.savefig('output/demo4_evolution.png', dpi=150)
    print("Saved: output/demo4_evolution.png")
    plt.show()


def demo_preprocessing_techniques():
    """
    Demonstrate various preprocessing techniques.
    """
    print("\n" + "="*60)
    print("DEMO 5: Advanced Preprocessing Techniques")
    print("="*60 + "\n")

    # Generate noisy data
    loader = generate_synthetic_das_data(
        n_channels=300,
        n_samples=10000,
        sampling_rate=1000.0,
        n_events=5,
        noise_level=0.3,  # Higher noise
        seed=123
    )

    raw_data = loader.data

    # Apply different preprocessing pipelines
    preprocessor = DASPreprocessor(sampling_rate=loader.sampling_rate)

    # Pipeline 1: Basic filtering
    basic_filtered = (preprocessor
        .set_data(raw_data)
        .remove_mean()
        .bandpass_filter(5.0, 100.0)
        .get_data())

    # Pipeline 2: With SVD denoising
    svd_denoised = (preprocessor
        .set_data(raw_data)
        .remove_mean()
        .bandpass_filter(5.0, 100.0)
        .svd_denoise(n_components=15)
        .get_data())

    # Pipeline 3: Optimized for CO2 monitoring
    co2_optimized = preprocess_for_co2_monitoring(raw_data, sampling_rate=1000.0)

    # Compare
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    time = loader.time
    distance = loader.distance
    viz = DASVisualizer()

    viz.waterfall_plot(raw_data, time, distance,
                       title='Raw Data (High Noise)', ax=axes[0, 0], colorbar=False)
    viz.waterfall_plot(basic_filtered, time, distance,
                       title='Basic Bandpass Filter', ax=axes[0, 1], colorbar=False)
    viz.waterfall_plot(svd_denoised, time, distance,
                       title='SVD Denoised', ax=axes[1, 0], colorbar=False)
    viz.waterfall_plot(co2_optimized, time, distance,
                       title='CO2 Monitoring Optimized', ax=axes[1, 1])

    plt.tight_layout()
    plt.savefig('output/demo5_preprocessing.png', dpi=150)
    print("\nSaved: output/demo5_preprocessing.png")
    plt.show()


def main():
    """
    Main demonstration script for DAS CO2 monitoring project.
    """
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║  DISTRIBUTED ACOUSTIC SENSING FOR CO2 STORAGE MONITORING     ║
    ║                                                              ║
    ║  A demonstration of DAS data processing techniques for       ║
    ║  monitoring carbon capture and storage (CCS) operations.     ║
    ╚══════════════════════════════════════════════════════════════╝
    """)

    # Create output directory
    Path('output').mkdir(exist_ok=True)

    # Run demonstrations
    try:
        # Demo 1: Basic processing
        loader, processed_data = demo_basic_das_processing()

        # Demo 2: Event detection
        events = demo_event_detection(loader, processed_data)

        # Demo 3: F-K analysis
        demo_fk_analysis(loader, processed_data)

        # Demo 4: CO2 monitoring
        demo_co2_monitoring()

        # Demo 5: Preprocessing techniques
        demo_preprocessing_techniques()

        print("\n" + "="*60)
        print("ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\nOutput files saved to: output/")
        print("\nFor more information on real DAS datasets:")
        print_porotomo_info()

    except Exception as e:
        print(f"\nError during demonstration: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
