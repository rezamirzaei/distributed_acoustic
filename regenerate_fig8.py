#!/usr/bin/env python3
"""Regenerate Figure 8 (STA/LTA Sensitivity Analysis) with improved parameters."""
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, 'src')

from das_co2_monitoring import DASDataLoader, DASPreprocessor, EventDetector
from das_co2_monitoring.data_loader import download_sample_data

print("[1] Loading data...")
npz_path = download_sample_data(dataset='porotomo_sample')
loader = DASDataLoader().load_numpy(npz_path)
with np.load(npz_path) as z:
    fs = float(z.get('sampling_rate', loader.sampling_rate))
raw = loader.data
print(f"    Shape: {raw.shape}, fs: {fs}")

print("[2] Preprocessing...")
pre = DASPreprocessor(sampling_rate=fs, channel_spacing=1.0)
proc = (pre.set_data(raw).remove_mean().remove_trend(order=1)
        .bandpass_filter(2.0, 80.0).median_denoise(kernel_size=(1, 5))
        .normalize(method='std').get_data())

print("[3] STA/LTA sensitivity analysis...")
# Use subset for faster computation
ch_ds = slice(0, proc.shape[0], 5)  # every 5th channel = 400 channels
t_ds = slice(0, int(40 * fs))  # 40 seconds
data_small = proc[ch_ds, t_ds]
print(f"    Subset shape: {data_small.shape}")

det = EventDetector(sampling_rate=fs, channel_spacing=5.0)

# Wider range with lower thresholds to see more variation
sta_vals = [0.01, 0.02, 0.03, 0.05, 0.08]
trig_vals = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]

counts = np.zeros((len(sta_vals), len(trig_vals)), dtype=int)
for i, sta in enumerate(sta_vals):
    for j, trig in enumerate(trig_vals):
        ev = det.sta_lta_detect(
            data_small,
            sta_window=sta,
            lta_window=0.5,
            trigger_on=trig,
            trigger_off=max(1.2, 0.5 * trig),
            min_channels=max(2, int(0.02 * data_small.shape[0])),
            min_duration=0.01,
        )
        counts[i, j] = len(ev)
        print(f"    STA={sta:.2f}, trig={trig:.1f}: {len(ev)} events")

print("\n[4] Creating figure...")
fig, ax = plt.subplots(figsize=(9, 5))
im = ax.imshow(counts, cmap="YlOrRd", aspect='auto')
ax.set_xticks(range(len(trig_vals)))
ax.set_xticklabels([str(v) for v in trig_vals])
ax.set_yticks(range(len(sta_vals)))
ax.set_yticklabels([str(v) for v in sta_vals])
ax.set_xlabel("Trigger Threshold (STA/LTA ratio)", fontsize=11)
ax.set_ylabel("STA Window (s)", fontsize=11)
ax.set_title("STA/LTA Detection Sensitivity Analysis\n(400 channels × 40s subset)", fontsize=12)

for i in range(counts.shape[0]):
    for j in range(counts.shape[1]):
        color = 'white' if counts[i, j] > counts.max() * 0.5 else 'black'
        ax.text(j, i, str(counts[i, j]), ha="center", va="center",
                color=color, fontsize=10, fontweight='bold')

cbar = fig.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label("Number of Events Detected", fontsize=10)
fig.tight_layout()
fig.savefig("output/report_stalta_sensitivity.png", dpi=220, bbox_inches="tight")
fig.savefig("report/report_stalta_sensitivity.png", dpi=220, bbox_inches="tight")
print("    Saved to output/ and report/")

print("\n[5] Summary:")
max_idx = np.unravel_index(counts.argmax(), counts.shape)
print(f"    Max events: {counts.max()} at STA={sta_vals[max_idx[0]]}, trig={trig_vals[max_idx[1]]}")
print(f"    Min events: {counts.min()}")
print(f"    Range: {counts.min()} - {counts.max()}")
print("\nDone!")
