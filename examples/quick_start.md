# DAS Processing Example: Quick Start

This example shows how to quickly process DAS data for CO2 storage monitoring.

```python
import numpy as np
import matplotlib.pyplot as plt
from das_co2_monitoring import (
    DASDataLoader,
    DASPreprocessor,
    EventDetector,
    DASVisualizer,
    CO2Monitor
)
from das_co2_monitoring.data_loader import generate_synthetic_das_data

# ============================================
# 1. Load or Generate DAS Data
# ============================================

# Option A: Generate synthetic data for testing
loader = generate_synthetic_das_data(
    n_channels=500,      # Number of channels (spatial samples)
    n_samples=10000,     # Number of time samples
    sampling_rate=1000.0, # Hz
    channel_spacing=1.0,  # meters
    n_events=5,          # Microseismic events to simulate
    noise_level=0.1,     # Noise level
    seed=42              # For reproducibility
)

# Option B: Load real data from HDF5
# loader = DASDataLoader()
# loader.load_hdf5('path/to/data.h5')
# loader.set_parameters(sampling_rate=1000.0, channel_spacing=1.0)

print(loader.info())

# ============================================
# 2. Preprocess the Data
# ============================================

preprocessor = DASPreprocessor(sampling_rate=loader.sampling_rate)

processed_data = (preprocessor
    .set_data(loader.data)
    .remove_mean()                    # Remove DC offset
    .bandpass_filter(5.0, 100.0)      # 5-100 Hz bandpass
    .median_denoise(kernel_size=(1, 5))  # Remove spikes
    .normalize(method='std')          # Normalize amplitudes
    .get_data())

print(f"Processing steps: {preprocessor.get_history()}")

# ============================================
# 3. Detect Microseismic Events
# ============================================

detector = EventDetector(
    sampling_rate=loader.sampling_rate,
    channel_spacing=loader.channel_spacing
)

# STA/LTA detection
events = detector.sta_lta_detect(
    processed_data,
    sta_window=0.02,   # 20 ms short-term window
    lta_window=0.3,    # 300 ms long-term window
    trigger_on=2.5,    # Trigger threshold
    trigger_off=1.0,   # Off threshold
    min_channels=10    # Minimum channels to trigger
)

print(f"\nDetected {len(events)} events:")
for e in events[:5]:
    print(f"  t={e.time:.3f}s, channel={e.channel}, SNR={e.snr:.1f}")

# ============================================
# 4. Visualize Results
# ============================================

viz = DASVisualizer(figsize=(12, 8))

# Waterfall plot with detected events
fig = viz.waterfall_plot(
    processed_data,
    loader.time,
    loader.distance,
    title='DAS Data with Detected Microseismic Events',
    events=events
)
plt.show()

# F-K spectrum
fig = viz.fk_spectrum(
    processed_data,
    sampling_rate=loader.sampling_rate,
    channel_spacing=loader.channel_spacing,
    velocity_lines=[1500, 2000, 3000]
)
plt.show()

# Event catalog
fig = viz.event_catalog_plot(events, loader.distance)
plt.show()

# ============================================
# 5. CO2 Monitoring (Time-Lapse Analysis)
# ============================================

# Set up monitoring with baseline
monitor = CO2Monitor(
    sampling_rate=loader.sampling_rate,
    channel_spacing=loader.channel_spacing
)
monitor.set_baseline(processed_data)

# Simulate a repeat survey (in practice, load real repeat data)
from das_co2_monitoring.monitoring import simulate_co2_injection_effect

repeat_data = simulate_co2_injection_effect(
    processed_data,
    injection_channel=250,  # Where CO2 is injected
    effect_radius=50,       # Channels affected
    strain_increase=0.15,   # 15% strain increase
    velocity_decrease=0.03  # 3% velocity decrease
)

# Analyze changes
result = monitor.analyze_repeat(repeat_data, timestamp=30.0)  # 30 days after injection

print(f"\nMonitoring Result:")
print(f"  Quality: {result.quality_metric:.3f}")
print(f"  Anomalies detected: {len(result.anomaly_locations)}")

# Comparison visualization
fig = viz.comparison_plot(
    processed_data,
    repeat_data,
    loader.time,
    loader.distance,
    titles=('Baseline', '30 Days Post-Injection'),
    show_difference=True
)
plt.show()
```

## Real Data Sources

For real DAS data, consider these publicly available datasets:

1. **PoroTomo Brady Hot Springs**
   - URL: https://gdr.openei.org/submissions/980
   - 8.6 km fiber, geothermal site
   - Format: HDF5

2. **FORGE Utah**
   - URL: https://gdr.openei.org/submissions/1318
   - Geothermal DAS with microseismicity
   - Format: HDF5/SEGY

3. **IRIS PASSCAL DAS**
   - URL: https://ds.iris.edu/mda/4G
   - Various DAS experiments
   - Format: miniSEED
