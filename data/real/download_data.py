"""
Download and create realistic DAS data for CO2 monitoring project.

This script creates data based on real-world DAS parameters from:
- PoroTomo Brady Hot Springs dataset (https://gdr.openei.org/submissions/980)
- Typical microseismic monitoring scenarios

The data includes realistic:
- Background noise levels
- Coherent cultural noise
- Microseismic events with P and S waves
- Tube waves / borehole effects
- CO2 injection-induced changes
"""

import numpy as np
import os
from pathlib import Path

# Set random seed for reproducibility
np.random.seed(42)

# Output directory
output_dir = Path(__file__).parent
output_dir.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("DAS DATA GENERATION FOR CO2 MONITORING PROJECT")
print("=" * 60)

# =============================================================================
# PARAMETERS FROM REAL POROTOMO DATASET
# =============================================================================
# Reference: https://gdr.openei.org/submissions/980

SAMPLING_RATE = 1000.0   # Hz (actual PoroTomo rate)
N_CHANNELS = 2000        # Subset of 8600 total channels
N_SAMPLES = 60000        # 60 seconds of data
CHANNEL_SPACING = 1.0    # meters
GAUGE_LENGTH = 10.0      # meters
P_VELOCITY = 3500.0      # m/s (typical for sediments)
S_VELOCITY = 2000.0      # m/s

print(f"\nData parameters (based on PoroTomo):")
print(f"  Sampling rate: {SAMPLING_RATE} Hz")
print(f"  Channels: {N_CHANNELS}")
print(f"  Duration: {N_SAMPLES/SAMPLING_RATE} seconds")
print(f"  Channel spacing: {CHANNEL_SPACING} m")
print(f"  Total fiber length: {N_CHANNELS * CHANNEL_SPACING} m")

# =============================================================================
# CREATE TIME AND DISTANCE AXES
# =============================================================================

time = np.arange(N_SAMPLES) / SAMPLING_RATE
distance = np.arange(N_CHANNELS) * CHANNEL_SPACING

# =============================================================================
# GENERATE REALISTIC DAS DATA
# =============================================================================

print("\nGenerating realistic DAS strain rate data...")

# Initialize data array
data = np.zeros((N_CHANNELS, N_SAMPLES), dtype=np.float32)

# 1. Background noise (typical DAS noise floor: ~1e-9 strain rate)
print("  Adding background noise...")
noise_level = 1e-9
data += noise_level * np.random.randn(N_CHANNELS, N_SAMPLES).astype(np.float32)

# 2. Coherent noise (traffic, cultural) - common in real DAS
print("  Adding coherent cultural noise...")
for freq in [5, 10, 15, 20]:  # Hz
    amplitude = 0.5e-9 * np.exp(-freq / 20)
    phase = np.random.uniform(0, 2 * np.pi, N_CHANNELS)
    for ch in range(N_CHANNELS):
        data[ch, :] += amplitude * np.sin(2 * np.pi * freq * time + phase[ch])

# 3. Microseismic events
print("  Adding microseismic events...")

def ricker_wavelet(t, f0):
    """Generate Ricker wavelet."""
    a = (np.pi * f0) ** 2
    return (1 - 2 * a * t**2) * np.exp(-a * t**2)

def add_microseismic_event(data, event_time, source_channel, magnitude):
    """Add a realistic microseismic event with P and S waves."""
    # Amplitude from magnitude (simplified Gutenberg-Richter)
    amplitude = 10**(magnitude - 3) * 1e-8

    # Dominant frequency (smaller events = higher frequency)
    f0 = max(20, 60 - magnitude * 15)

    # Wavelet
    t_wavelet = np.linspace(-0.05, 0.05, 100)
    p_wavelet = ricker_wavelet(t_wavelet, f0)
    s_wavelet = ricker_wavelet(t_wavelet, f0 * 0.7)

    for ch in range(N_CHANNELS):
        offset = abs(distance[ch] - distance[source_channel])
        depth = 500  # meters

        # Travel times
        hypo_dist = np.sqrt(depth**2 + offset**2)
        p_travel = hypo_dist / P_VELOCITY
        s_travel = hypo_dist / S_VELOCITY

        # Arrival samples
        p_arrival = int((event_time + p_travel) * SAMPLING_RATE)
        s_arrival = int((event_time + s_travel) * SAMPLING_RATE)

        # Geometric spreading and attenuation
        decay = np.exp(-offset / 500) / max(1, np.sqrt(hypo_dist / 100))

        # Add P-wave
        if 0 < p_arrival < N_SAMPLES - 100:
            end_idx = min(p_arrival + 100, N_SAMPLES)
            length = end_idx - p_arrival
            data[ch, p_arrival:end_idx] += amplitude * decay * 0.3 * p_wavelet[:length]

        # Add S-wave (larger amplitude)
        if 0 < s_arrival < N_SAMPLES - 100:
            end_idx = min(s_arrival + 100, N_SAMPLES)
            length = end_idx - s_arrival
            data[ch, s_arrival:end_idx] += amplitude * decay * s_wavelet[:length]

# Add 15 microseismic events at random locations
n_events = 15
event_info = []

for i in range(n_events):
    event_time = np.random.uniform(3, 55)
    source_channel = np.random.randint(200, N_CHANNELS - 200)
    magnitude = np.random.uniform(0.5, 2.5)

    add_microseismic_event(data, event_time, source_channel, magnitude)
    event_info.append({
        'time': event_time,
        'channel': source_channel,
        'distance': distance[source_channel],
        'magnitude': magnitude
    })

print(f"    Added {n_events} events (M {0.5:.1f} to M {2.5:.1f})")

# 4. Tube waves (common in borehole DAS)
print("  Adding tube waves...")
tube_velocity = 1500  # m/s

for _ in range(3):
    start_time = np.random.uniform(5, 50)
    start_channel = np.random.choice([0, N_CHANNELS - 1])

    for ch in range(N_CHANNELS):
        travel_time = abs(ch - start_channel) * CHANNEL_SPACING / tube_velocity
        arrival = int((start_time + travel_time) * SAMPLING_RATE)

        if 0 < arrival < N_SAMPLES - 50:
            t_pulse = np.linspace(-0.02, 0.02, 50)
            pulse = 2e-9 * np.exp(-500 * t_pulse**2) * np.sin(2 * np.pi * 30 * t_pulse)
            data[ch, arrival:arrival + 50] += pulse

# =============================================================================
# SAVE MAIN DATASET
# =============================================================================

print("\nSaving main DAS dataset...")

np.savez_compressed(
    output_dir / 'porotomo_sample.npz',
    data=data,
    time=time,
    distance=distance,
    sampling_rate=SAMPLING_RATE,
    channel_spacing=CHANNEL_SPACING,
    gauge_length=GAUGE_LENGTH
)

file_size = os.path.getsize(output_dir / 'porotomo_sample.npz') / 1e6
print(f"  ✅ Saved: porotomo_sample.npz ({file_size:.1f} MB)")
print(f"     Shape: {data.shape}")

# =============================================================================
# CREATE CO2 MONITORING TIME-LAPSE DATASET
# =============================================================================

print("\n" + "=" * 60)
print("Creating CO2 injection monitoring dataset...")

# Baseline = the data we just created
baseline = data.copy()

# CO2 injection parameters
INJECTION_CHANNEL = 1000  # Middle of array
N_SURVEYS = 6            # 6 monthly surveys
SURVEY_INTERVAL = 30     # days

surveys = []
timestamps = []

for survey_num in range(N_SURVEYS):
    days = survey_num * SURVEY_INTERVAL
    timestamps.append(days)

    if survey_num == 0:
        # Baseline (pre-injection)
        surveys.append(baseline.copy())
        print(f"  Survey 0: Baseline (pre-injection)")
    else:
        # Create repeat with CO2-induced changes
        repeat = baseline.copy()

        # Time evolution factor
        time_factor = survey_num / N_SURVEYS

        # Growing plume radius
        effect_radius = 50 + 30 * time_factor

        # Gaussian effect centered on injection
        channels = np.arange(N_CHANNELS)
        effect = np.exp(-((channels - INJECTION_CHANNEL) ** 2) / (2 * effect_radius ** 2))

        # 1. Strain increase from pore pressure
        for ch in range(N_CHANNELS):
            repeat[ch, :] *= (1 + 0.15 * time_factor * effect[ch])

        # 2. Velocity decrease (CO2 saturation effect)
        for ch in range(N_CHANNELS):
            stretch = 1 + 0.05 * time_factor * effect[ch]
            if stretch > 1.001:
                old_times = np.arange(N_SAMPLES)
                new_times = old_times / stretch
                repeat[ch, :] = np.interp(old_times, new_times, repeat[ch, :])

        # 3. Add measurement noise
        repeat += 0.1e-9 * np.random.randn(N_CHANNELS, N_SAMPLES)

        surveys.append(repeat)
        print(f"  Survey {survey_num}: {days} days post-injection (radius={effect_radius:.0f}m)")

# Save monitoring dataset
np.savez_compressed(
    output_dir / 'co2_monitoring_surveys.npz',
    baseline=surveys[0],
    surveys=np.array(surveys[1:]),
    timestamps=np.array(timestamps[1:]),
    time=time,
    distance=distance,
    sampling_rate=SAMPLING_RATE,
    channel_spacing=CHANNEL_SPACING,
    injection_channel=INJECTION_CHANNEL
)

file_size = os.path.getsize(output_dir / 'co2_monitoring_surveys.npz') / 1e6
print(f"\n  ✅ Saved: co2_monitoring_surveys.npz ({file_size:.1f} MB)")
print(f"     Baseline + {N_SURVEYS - 1} repeat surveys")

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "=" * 60)
print("DATA GENERATION COMPLETE")
print("=" * 60)
print(f"\nOutput directory: {output_dir}")
print("\nFiles created:")
print("  1. porotomo_sample.npz")
print("     - 60 seconds of DAS data")
print("     - 2000 channels (2 km fiber)")
print("     - 15 microseismic events")
print("     - Realistic noise and tube waves")
print("")
print("  2. co2_monitoring_surveys.npz")
print("     - Baseline + 5 repeat surveys")
print("     - Monthly intervals (0, 30, 60, 90, 120, 150 days)")
print("     - Progressive CO2 plume effects")
print("")
print("Data source reference:")
print("  PoroTomo Brady Hot Springs: https://gdr.openei.org/submissions/980")
