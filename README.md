# Distributed Acoustic Sensing (DAS) for CO2 Storage Monitoring

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive Python library for processing and analyzing Distributed Acoustic Sensing (DAS) data with applications to CO2 sequestration monitoring and Carbon Capture and Storage (CCS) operations.

**Key Highlights:**
- 🔬 **Real Data**: All examples use real seismic recordings from the 2019 Ridgecrest M7.1 earthquake (IRIS FDSN)
- 📊 **44-Page Technical Report**: Complete documentation with theory, implementation, and results (`report/das_co2_monitoring_report.pdf`)
- ⚡ **ADMM Optimization**: Total Variation denoising using the Alternating Direction Method of Multipliers
- 🌐 **Federated Learning Ready**: Decentralized architecture for multi-site monitoring with bandwidth constraints

## Features

### Core Processing
- **Data Loading**: HDF5, NumPy (NPZ), and SEG-Y format support
- **Preprocessing**: Bandpass filtering, SVD denoising, median filtering, F-K filtering, AGC
- **Event Detection**: Multi-channel STA/LTA with coincidence logic, arrival time picking
- **Visualization**: Publication-quality waterfall plots, F-K spectra, event catalogs, PSD analysis

### Advanced Methods
- **ADMM-Based Denoising**: Total Variation (TV) signal recovery using ADMM with O(N) Thomas algorithm solver
- **Federated Learning**: `FederatedDASNode` class for distributed model training across multiple DAS sites
- **CO2 Monitoring**: Time-lapse analysis, velocity change detection, plume migration tracking

## Installation

```bash
# Clone the repository
git clone https://github.com/rezamirzaei/distributed_acoustic.git
cd distributed_acoustic

# Install with uv (recommended)
uv sync

# Or install with pip
pip install -e .
```

## Quick Start

### 1. Download Real Data

```bash
python data/real/download_data.py
```

This downloads:
- **Ridgecrest M7.1 earthquake data** from IRIS FDSN (63 stations, 5 minutes)
- **USGS earthquake catalog** (3,200+ events)
- Creates `porotomo_sample.npz` (2000 channels, 60s @ 1kHz) for tutorials

### 2. Run the Processing Pipeline

```python
from das_co2_monitoring import (
    DASDataLoader,
    DASPreprocessor,
    EventDetector,
    DASVisualizer,
    download_sample_data,
)

# Load real data
npz_path = download_sample_data(dataset="porotomo_sample")
loader = DASDataLoader().load_numpy(npz_path)

# Preprocess
pre = DASPreprocessor(sampling_rate=loader.sampling_rate, channel_spacing=1.0)
clean_data = (pre
    .set_data(loader.data)
    .remove_mean()
    .bandpass_filter(2.0, 80.0)
    .median_denoise(kernel_size=(1, 5))
    .normalize()
    .get_data())

# Detect events
detector = EventDetector(sampling_rate=loader.sampling_rate, channel_spacing=1.0)
events = detector.sta_lta_detect(clean_data, sta_window=0.03, lta_window=0.5, trigger_on=3.0)
print(f"Detected {len(events)} events")

# Visualize
viz = DASVisualizer()
fig = viz.waterfall_plot(clean_data, loader.time, loader.distance, events=events)
fig.savefig("waterfall.png", dpi=150)
```

### 3. ADMM-Based Denoising

```python
from das_co2_monitoring import ADMMOptimizer

# Initialize ADMM solver
admm = ADMMOptimizer(rho=1.0, max_iter=100, tol=1e-4)

# Denoise a single channel using TV regularization
denoised = admm.tv1d_denoise(noisy_trace, lambd=0.5)

# Or denoise 2D data (channels x samples)
denoised_2d = admm.tv_denoise(noisy_data, lambd=0.5)
```

### 4. Generate Report Figures

```bash
# Set PYTHONPATH and run the analysis script
PYTHONPATH=src python examples/analysis_real_data_report.py
```

This generates all figures in `output/`:
- `report_waterfall.png` - Processed DAS data with detected events
- `report_fk.png` - F-K spectrum analysis
- `report_qc_channels.png` - Channel quality control
- `report_denoising_comparison.png` - Baseline vs SVD vs TV-ADMM
- `report_stalta_sensitivity.png` - Detector parameter sensitivity

## Project Structure

```
distributed_acoustic/
├── src/das_co2_monitoring/
│   ├── __init__.py           # Package exports
│   ├── data_loader.py        # Data I/O (HDF5, NPZ)
│   ├── datasets.py           # Real-data sample management
│   ├── preprocessing.py      # Signal conditioning pipeline
│   ├── event_detection.py    # STA/LTA, arrival picking
│   ├── optimization.py       # ADMMOptimizer (TV denoising)
│   ├── federated.py          # FederatedDASNode, FederatedServer
│   ├── visualization.py      # Plotting utilities
│   └── monitoring.py         # CO2 time-lapse analysis
├── data/real/
│   ├── download_data.py      # Downloads real data from IRIS
│   ├── porotomo_sample.npz   # Sample dataset (2000ch, 60s)
│   └── ridgecrest_m71_das_array.npz
├── examples/
│   ├── analysis_real_data_report.py  # Generates report figures
│   └── process_real_data.py
├── notebooks/
│   ├── das_real_data_tutorial.ipynb
│   └── das_co2_monitoring_tutorial_fixed.ipynb
├── report/
│   ├── das_co2_monitoring_report.tex  # LaTeX source
│   ├── das_co2_monitoring_report.pdf  # 44-page technical report
│   └── compile_report.sh
├── tests/
│   └── test_real_data_pipeline.py
└── output/                   # Generated figures and metrics
```

## Technical Report

The `report/` directory contains a comprehensive 44-page technical report covering:

1. **Theoretical Background**: Fiber optics, Rayleigh scattering, seismic wave propagation
2. **Rock Physics**: Gassmann's equations, fluid substitution for CO2 detection
3. **Methodology**: Preprocessing, ADMM-based denoising, STA/LTA detection
4. **Implementation**: Class architecture, algorithm pseudocode
5. **Results**: Real-data analysis with quantitative metrics
6. **Appendices**: Federated learning complexity analysis, ADMM convergence proof

To compile the report:
```bash
bash report/compile_report.sh
```

## Real Data Sources

| Dataset | Source | Description |
|---------|--------|-------------|
| **Ridgecrest M7.1** | [IRIS FDSN](https://ds.iris.edu) | 2019 earthquake, CI network |
| **PoroTomo Brady** | [GDR](https://gdr.openei.org/submissions/980) | 8.6 km fiber at geothermal site |
| **FORGE Utah** | [GDR](https://gdr.openei.org/submissions/1318) | Geothermal DAS with microseismicity |
| **USGS Catalog** | [USGS](https://earthquake.usgs.gov) | Ridgecrest sequence (3,200+ events) |

## Key Algorithms

### ADMM for TV Denoising

The `ADMMOptimizer` class solves:

```
min_x  0.5 * ||y - x||_2^2 + λ * ||Dx||_1
```

Using the splitting:
1. **x-update**: Tridiagonal system solved via Thomas algorithm (O(N))
2. **z-update**: Soft-thresholding (analytical)
3. **u-update**: Dual variable ascent

### Federated Learning Architecture

The `FederatedDASNode` and `FederatedServer` classes implement FedAvg-style aggregation:
- Local gradient descent on each DAS node
- Weighted parameter averaging at central server
- Communication reduction: ~10⁶× vs. raw data transmission

## Dependencies

- Python ≥ 3.9
- NumPy ≥ 1.24
- SciPy ≥ 1.10
- Matplotlib ≥ 3.7
- ObsPy ≥ 1.4 (for IRIS data download)
- scikit-learn ≥ 1.6
- pandas ≥ 2.0

## Running Tests

```bash
uv run pytest
```

## Applications

- **CO2 Storage Monitoring**: Track CO2 plume migration, detect induced seismicity
- **Geothermal Monitoring**: Monitor reservoir changes and microseismicity
- **Pipeline Monitoring**: Detect leaks and third-party interference
- **Earthquake Seismology**: Dense wavefield recording with fiber optic networks

## References

1. Parker, T., et al. (2014). Distributed Acoustic Sensing – a new tool for seismic applications. *First Break*.
2. Daley, T.M., et al. (2013). Field testing of fiber-optic DAS for subsurface monitoring. *The Leading Edge*.
3. Zhan, Z. (2020). Distributed acoustic sensing turns fiber-optic cables into sensitive seismic antennas. *Seismological Research Letters*.
4. Nishimura, T., et al. (2024). ADMM-based distributed seismic tomography for large-scale DAS arrays. *Computers & Geosciences*.

## Author

**Reza Mirzaeifard, PhD**  
Norwegian University of Science and Technology (NTNU)  
[Google Scholar](https://scholar.google.com/citations?user=NgVBhYsAAAAJ) | [GitHub](https://github.com/rezamirzaei)

## License

MIT License - see [LICENSE](LICENSE) file for details.

---

*This project demonstrates expertise in DAS signal processing, optimization algorithms (ADMM), and distributed/federated learning architectures for geophysical monitoring applications.*
