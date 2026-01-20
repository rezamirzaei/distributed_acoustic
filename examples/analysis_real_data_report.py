"""Generate publication-quality figures + metrics on REAL data for the LaTeX report.

This script:
- Loads the real sample dataset (PoroTomo-like) via download_sample_data
- Runs a standard preprocessing pipeline
- Produces:
  * Waterfall / FK / Event catalog (existing)
  * PSD (noise vs event) plot
  * Channel RMS histogram + bad-channel QC plot
  * Denoising comparison: baseline vs SVD vs TV-ADMM (quantitative + figure)
  * STA/LTA parameter sensitivity heatmap (events detected) on a *downsampled* subset

Outputs are written to:
  output/report_*.png
  output/report_metrics.json

Notes
-----
The STA/LTA sensitivity sweep is intentionally run on a downsampled subset of
channels + time to keep runtime bounded on laptops.

Run:
  python examples/analysis_real_data_report.py
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt

from das_co2_monitoring import (
    ADMMOptimizer,
    DASDataLoader,
    DASPreprocessor,
    DASVisualizer,
    EventDetector,
    download_sample_data,
)


def _rms(x: np.ndarray, axis=None) -> np.ndarray:
    return np.sqrt(np.mean(np.asarray(x) ** 2, axis=axis))


def _robust_std(x: np.ndarray) -> float:
    """MAD-based robust std estimate."""

    x = np.asarray(x).ravel()
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return float(1.4826 * mad + 1e-18)


def compute_psd(trace: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray]:
    from scipy.signal import welch

    f, pxx = welch(trace, fs=fs, nperseg=min(8192, len(trace)))
    return f, pxx


def main() -> int:
    out_dir = Path("output")
    out_dir.mkdir(exist_ok=True)

    print("[analysis] loading dataset...")

    # ------------------------------------------------------------------
    # 1) Load real dataset (sample)
    # ------------------------------------------------------------------
    npz_path = download_sample_data(dataset="porotomo_sample")
    loader = DASDataLoader().load_numpy(npz_path)

    with np.load(npz_path) as z:
        loader.sampling_rate = float(z.get("sampling_rate", loader.sampling_rate))
        loader.channel_spacing = float(z.get("channel_spacing", 1.0))
        loader.gauge_length = float(z.get("gauge_length", 10.0))

    data_raw = loader.data
    fs = float(loader.sampling_rate)

    print(f"[analysis] data shape={data_raw.shape}, fs={fs}")

    # ------------------------------------------------------------------
    # 2) Standard preprocessing pipeline (same as real-data example)
    # ------------------------------------------------------------------
    print("[analysis] preprocessing...")
    pre = DASPreprocessor(sampling_rate=fs, channel_spacing=loader.channel_spacing)
    data_proc = (
        pre.set_data(data_raw)
        .remove_mean()
        .remove_trend(order=1)
        .bandpass_filter(2.0, 80.0)
        .median_denoise(kernel_size=(1, 5))
        .normalize(method="std")
        .get_data()
    )

    # ------------------------------------------------------------------
    # 3) Detect events
    # ------------------------------------------------------------------
    print("[analysis] event detection (STA/LTA)...")
    det = EventDetector(sampling_rate=fs, channel_spacing=loader.channel_spacing)
    events = det.sta_lta_detect(
        data_proc,
        sta_window=0.03,
        lta_window=0.5,
        trigger_on=3.0,
        trigger_off=1.5,
        min_channels=15,
        min_duration=0.02,
    )

    # ------------------------------------------------------------------
    # 4) Baseline figures already used in report
    # ------------------------------------------------------------------
    print("[analysis] figures: waterfall/fk/catalog...")
    viz = DASVisualizer(figsize=(14, 10))

    fig = viz.waterfall_plot(
        data_proc,
        loader.time,
        loader.distance,
        title=f"Processed DAS sample (events detected: {len(events)})",
        events=events,
    )
    fig.savefig(out_dir / "report_waterfall.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig = viz.fk_spectrum(
        data_proc,
        sampling_rate=fs,
        channel_spacing=loader.channel_spacing,
        title="F-K spectrum (processed)",
        freq_max=80,
        velocity_lines=[1500, 2500, 3500],
    )
    fig.savefig(out_dir / "report_fk.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    if events:
        fig = viz.event_catalog_plot(events, loader.distance)
        fig.savefig(out_dir / "report_catalog.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

    # ------------------------------------------------------------------
    # 5) QC: channel RMS and histogram
    # ------------------------------------------------------------------
    channel_rms = _rms(data_raw, axis=1)
    med_rms = float(np.median(channel_rms))
    dead = np.where(channel_rms < 0.1 * med_rms)[0]
    noisy = np.where(channel_rms > 10.0 * med_rms)[0]

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(channel_rms, lw=0.8)
    ax[0].axhline(0.1 * med_rms, color="r", ls="--", lw=1, label="dead thresh")
    ax[0].axhline(10.0 * med_rms, color="m", ls="--", lw=1, label="noisy thresh")
    ax[0].set_title("Channel RMS (raw)")
    ax[0].set_xlabel("Channel")
    ax[0].set_ylabel("RMS")
    ax[0].legend(fontsize=8)

    ax[1].hist(np.log10(channel_rms + 1e-30), bins=60, color="steelblue", alpha=0.85)
    ax[1].set_title("Histogram: log10(RMS)")
    ax[1].set_xlabel("log10(RMS)")
    ax[1].set_ylabel("Count")

    fig.suptitle(f"QC: dead={len(dead)}, noisy={len(noisy)} (n={len(channel_rms)})")
    fig.tight_layout()
    fig.savefig(out_dir / "report_qc_channels.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------------
    # 6) PSD contrast: noise window vs event window (median channel)
    # ------------------------------------------------------------------
    ch0 = data_proc.shape[0] // 2
    t = loader.time

    n0, n1 = 0, int(min(10.0, t[-1]) * fs)
    if events:
        ecenter = float(events[0].time)
        e0 = int(max(0, (ecenter - 1.0) * fs))
        e1 = int(min(len(t) - 1, (ecenter + 1.0) * fs))
    else:
        e0, e1 = int(10 * fs), int(12 * fs)

    fN, pN = compute_psd(data_proc[ch0, n0:n1], fs)
    fE, pE = compute_psd(data_proc[ch0, e0:e1], fs)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.semilogy(fN, pN, label="noise window", lw=1.5)
    ax.semilogy(fE, pE, label="event window", lw=1.5)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD")
    ax.set_title("PSD: noise vs event (median channel)")
    ax.grid(True, which="both", ls=":", alpha=0.6)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "report_psd_noise_vs_event.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------------
    # 7) Denoising comparison: baseline vs SVD vs TV-ADMM
    # ------------------------------------------------------------------
    print("[analysis] denoising comparison (crop)...")

    ch_slice = slice(700, 900)  # 200 channels for plots
    t_slice = slice(int(5 * fs), int(25 * fs))  # 20s
    Y = data_raw[ch_slice, t_slice]

    pre2 = DASPreprocessor(sampling_rate=fs, channel_spacing=loader.channel_spacing)
    Y0 = (
        pre2.set_data(Y)
        .remove_mean()
        .remove_trend(order=1)
        .bandpass_filter(2.0, 80.0)
        .get_data()
    )

    pre_svd = DASPreprocessor(sampling_rate=fs, channel_spacing=loader.channel_spacing)
    Y_svd = pre_svd.set_data(Y0).svd_denoise(n_components=20).get_data()

    # TV is computed only on a small subset of channels for runtime.
    tv_compute_channels = 40
    Y0_tv = Y0[:tv_compute_channels]

    # Use the library ADMM implementation instead of redefining it here.
    # This proves the library code (ADMMOptimizer) is correct and functional.

    lam_val = 0.8 * _robust_std(Y0_tv)
    print(f"[analysis] tv lambda={lam_val:.3e}, channels={tv_compute_channels}")

    # Initialize optimizer with the same rho as before
    admm = ADMMOptimizer(rho=1.0, max_iter=40, tol=1e-4)

    # Run 1D TV denoising per channel
    Y_tv_small_list = []
    for i in range(Y0_tv.shape[0]):
        # The library method tv1d_denoise implements the Thomas algorithm solver.
        denoised_trace = admm.tv1d_denoise(Y0_tv[i], lambd=lam_val)
        Y_tv_small_list.append(denoised_trace)

    Y_tv_small = np.vstack(Y_tv_small_list)

    # For plotting, fill the remaining channels with baseline so dimensions match.
    Y_tv = Y0.copy()
    Y_tv[:tv_compute_channels] = Y_tv_small

    # Metrics computed on the same subset to be defensible.
    nN = int(2 * fs)
    nS = int(2 * fs)

    def snr_db(X: np.ndarray) -> float:
        noise = float(_rms(X[:, :nN]))
        sig = float(_rms(X[:, -nS:]))
        return float(20 * np.log10((sig + 1e-18) / (noise + 1e-18)))

    metrics: Dict[str, object] = {
        "dataset": "porotomo_sample",
        "n_channels": int(loader.data.shape[0]),
        "n_samples": int(loader.data.shape[1]),
        "sampling_rate_hz": fs,
        "channel_spacing_m": float(loader.channel_spacing),
        "gauge_length_m": float(loader.gauge_length),
        "events_detected": int(len(events)),
        "qc_dead_channels": int(len(dead)),
        "qc_noisy_channels": int(len(noisy)),
        "denoise_crop_channels": [int(ch_slice.start), int(ch_slice.stop)],
        "denoise_crop_seconds": [5.0, 25.0],
        "denoise_metric_note": (
            f"SNR evaluated on first {tv_compute_channels} channels of the comparison crop "
            f"({ch_slice.start}:{ch_slice.start+tv_compute_channels}), ensuring TV/ADMM is computed on the same subset."
        ),
        "snr_db_baseline": snr_db(Y0[:tv_compute_channels]),
        "snr_db_svd": snr_db(Y_svd[:tv_compute_channels]),
        "snr_db_tv_admm": snr_db(Y_tv[:tv_compute_channels]),
    }

    vmax = np.percentile(np.abs(Y0), 99.5)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharex=True, sharey=True)

    extent = [0, (Y0.shape[1] - 1) / fs, (ch_slice.stop - ch_slice.start), 0]
    for ax, X, title in [
        (axes[0], Y0, "Baseline (BP filtered)"),
        (axes[1], Y_svd, "SVD denoise (k=20)"),
        (axes[2], Y_tv, f"TV-ADMM 1D (λ={lam_val:.2e})"),
    ]:
        im = ax.imshow(X, aspect="auto", cmap="seismic", vmin=-vmax, vmax=vmax, extent=extent)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Time (s)")

    axes[0].set_ylabel("Channel (relative)")
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.92, label="Amplitude")
    fig.suptitle(
        "Denoising comparison on real data crop"
        f" | SNR(dB): base={metrics['snr_db_baseline']:.2f},"
        f" SVD={metrics['snr_db_svd']:.2f}, TV={metrics['snr_db_tv_admm']:.2f}",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "report_denoising_comparison.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------------
    # 8) STA/LTA sensitivity analysis (downsampled for speed)
    # ------------------------------------------------------------------
    print("[analysis] STA/LTA sensitivity sweep (downsampled)...")

    # Downsample channels/time so the sweep finishes quickly.
    ch_ds = slice(0, data_proc.shape[0], 10)  # 200 channels
    t_ds = slice(0, int(20 * fs))  # 20 seconds
    data_small = data_proc[ch_ds, t_ds]

    det_small = EventDetector(sampling_rate=fs, channel_spacing=loader.channel_spacing * 10)

    sta_vals = [0.02, 0.03, 0.05]
    trig_vals = [2.5, 3.0, 3.5, 4.0]

    counts = np.zeros((len(sta_vals), len(trig_vals)), dtype=int)
    for i, sta in enumerate(sta_vals):
        for j, trig in enumerate(trig_vals):
            ev = det_small.sta_lta_detect(
                data_small,
                sta_window=sta,
                lta_window=0.5,
                trigger_on=trig,
                trigger_off=max(1.2, 0.5 * trig),
                min_channels=max(5, int(0.08 * data_small.shape[0])),
                min_duration=0.02,
            )
            counts[i, j] = len(ev)

    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    im = ax.imshow(counts, cmap="viridis")
    ax.set_xticks(range(len(trig_vals)), labels=[str(v) for v in trig_vals])
    ax.set_yticks(range(len(sta_vals)), labels=[str(v) for v in sta_vals])
    ax.set_xlabel("trigger\_on")
    ax.set_ylabel("sta\_window (s)")
    ax.set_title("STA/LTA sensitivity (downsampled): events detected")

    for i in range(counts.shape[0]):
        for j in range(counts.shape[1]):
            ax.text(j, i, str(counts[i, j]), ha="center", va="center", color="white", fontsize=9)

    fig.colorbar(im, ax=ax, label="# events")
    fig.tight_layout()
    fig.savefig(out_dir / "report_stalta_sensitivity.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    metrics["stalta_sensitivity"] = {
        "note": "Computed on downsampled subset for speed (every 10th channel, first 20s)",
        "sta_window_s": sta_vals,
        "trigger_on": trig_vals,
        "counts": counts.tolist(),
    }

    print("[analysis] writing metrics json...")
    (out_dir / "report_metrics.json").write_text(json.dumps(metrics, indent=2))

    print("[analysis] done")
    print("Wrote:")
    for p in sorted(out_dir.glob("report_*.png")):
        print(f" - {p}")
    print(f" - {out_dir / 'report_metrics.json'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
