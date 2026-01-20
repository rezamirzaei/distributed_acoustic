# Distributed Acoustic Sensing (DAS) for CO2 Storage Monitoring

A Python library for processing and analyzing Distributed Acoustic Sensing (DAS) data with applications to CO2 sequestration monitoring and Carbon Capture and Storage (CCS) operations.

## Features

- **Real-data-first samples**: small, realistic datasets under `data/real/` (auto-generated if missing)
- **Data Loading**: HDF5 and NumPy bundles
- **Preprocessing**: Bandpass filtering, SVD denoising, F-K filtering, AGC
- **Event Detection**: STA/LTA, amplitude threshold, template matching
- **Visualization**: Waterfall plots, F-K spectra, event catalogs
- **CO2 Monitoring**: Time-lapse analysis, plume migration tracking

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/distributed-acoustic.git
cd distributed-acoustic

# Install dependencies
pip install -e .
```

## Quick Start (Real data)

The easiest way to start is the PyCharm-friendly notebook:

- `notebooks/das_co2_monitoring_tutorial_pycharm.ipynb`

### Ensure the real sample datasets exist

The project ships (or regenerates) small real-parameter datasets.
If the `.npz` files are missing, they’ll be regenerated locally.

```bash
python data/real/download_data.py
```

### Run the real-data processing example

```bash
python examples/process_real_data.py
```

## Notes for PyCharm

- Open `notebooks/das_co2_monitoring_tutorial_pycharm.ipynb`
- Select the interpreter/kernel for this project
- Make sure the package is installed editable:

```bash
pip install -e .
```

## Quick Start

```python
from das_co2_monitoring import (
    DASDataLoader,
    DASPreprocessor,
    EventDetector,
    DASVisualizer,
    CO2Monitor
)
from das_co2_monitoring.data_loader import generate_synthetic_das_data

# Generate synthetic DAS data
loader = generate_synthetic_das_data(
    n_channels=500,
    n_samples=10000,
    sampling_rate=1000.0,
    n_events=5,
    seed=42
)

# Preprocess
preprocessor = DASPreprocessor(sampling_rate=loader.sampling_rate)
processed = (preprocessor
    .set_data(loader.data)
    .remove_mean()
    .bandpass_filter(5.0, 100.0)
    .normalize()
    .get_data())

# Detect events
detector = EventDetector(sampling_rate=loader.sampling_rate)
events = detector.sta_lta_detect(processed)

# Visualize
viz = DASVisualizer()
viz.waterfall_plot(processed, loader.time, loader.distance, events=events)
```

## Project Structure

```
distributed_acoustic/
├── src/
│   └── das_co2_monitoring/
│       ├── __init__.py
│       ├── data_loader.py      # Data loading and generation
│       ├── preprocessing.py    # Signal processing
│       ├── event_detection.py  # Microseismic detection
│       ├── visualization.py    # Plotting utilities
│       └── monitoring.py       # CO2 time-lapse analysis
├── notebooks/
│   └── das_co2_monitoring_tutorial.ipynb
├── examples/
│   └── quick_start.md
├── data/                       # Data directory
├── main.py                     # Demo script
├── pyproject.toml
└── README.md
```

## Examples

### Running the Demo

```bash
python main.py
```

This will run 5 demonstrations:
1. Basic DAS data processing
2. Microseismic event detection
3. F-K spectrum analysis
4. CO2 storage monitoring simulation
5. Advanced preprocessing techniques

### Jupyter Notebook

Open `notebooks/das_co2_monitoring_tutorial.ipynb` for an interactive tutorial.

## Publicly Available DAS Datasets

| Dataset | Source | Description |
|---------|--------|-------------|
| **PoroTomo Brady** | [GDR](https://gdr.openei.org/submissions/980) | 8.6 km fiber at geothermal site |
| **FORGE Utah** | [GDR](https://gdr.openei.org/submissions/1318) | Geothermal DAS with microseismicity |
| **IRIS PASSCAL** | [IRIS](https://ds.iris.edu/mda/4G) | Various DAS experiments |

## Key Processing Techniques

### Preprocessing Pipeline

```python
preprocessor = DASPreprocessor(sampling_rate=1000.0)
clean_data = (preprocessor
    .set_data(raw_data)
    .remove_mean()                    # Remove DC offset
    .remove_trend()                   # Remove linear trend
    .bandpass_filter(1.0, 100.0)      # Bandpass filter
    .svd_denoise(n_components=20)     # SVD denoising
    .fk_filter(velocity_min=100)      # F-K filter
    .automatic_gain_control()         # AGC for display
    .get_data())
```

### Event Detection

```python
detector = EventDetector(sampling_rate=1000.0)

# STA/LTA detection
events = detector.sta_lta_detect(
    data,
    sta_window=0.05,    # 50 ms short-term window
    lta_window=0.5,     # 500 ms long-term window
    trigger_on=3.0,     # Trigger threshold
    min_channels=10     # Minimum channels
)

# Pick arrival times
for event in events:
    event = detector.pick_arrivals(data, event)
```

### Time-Lapse Monitoring

```python
monitor = CO2Monitor(sampling_rate=1000.0)
monitor.set_baseline(baseline_data)

# Analyze repeat surveys
for repeat_data, timestamp in surveys:
    result = monitor.analyze_repeat(repeat_data, timestamp=timestamp)
    print(f"Quality: {result.quality_metric:.3f}")
    print(f"Anomalies: {len(result.anomaly_locations)}")

# Track plume migration
migration = monitor.detect_plume_migration()
print(f"Migration rate: {migration['migration_rate']:.2f} m/day")
```

## References

1. Parker, T., et al. (2014). Distributed Acoustic Sensing – a new tool for seismic applications. First Break.
2. Daley, T.M., et al. (2013). Field testing of fiber-optic distributed acoustic sensing for subsurface monitoring. The Leading Edge.
3. Lindsey, N.J., et al. (2019). Fiber-Optic Network Observations of Earthquake Wavefields. Geophysical Research Letters.

## Applications

- **CO2 Storage Monitoring**: Track CO2 plume migration and detect induced seismicity
- **Geothermal Monitoring**: Monitor reservoir changes and induced microseismicity
- **Pipeline Monitoring**: Detect leaks and third-party interference
- **Seismology**: Record earthquake waveforms along fiber optic cables

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License - see LICENSE file for details.
