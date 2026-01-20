import numpy as np

try:
    from das_co2_monitoring import DASDataLoader, DASPreprocessor, EventDetector
    from das_co2_monitoring.data_loader import download_sample_data
except ModuleNotFoundError:  # pragma: no cover
    # Fallback for IDE configurations where the package isn't installed yet.
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

    from das_co2_monitoring import DASDataLoader, DASPreprocessor, EventDetector
    from das_co2_monitoring.data_loader import download_sample_data


def test_porotomo_sample_pipeline_smoke():
    npz_path = download_sample_data(dataset="porotomo_sample")

    loader = DASDataLoader().load_numpy(npz_path)
    with np.load(npz_path) as z:
        if "sampling_rate" in z:
            loader.sampling_rate = float(z["sampling_rate"])
        if "channel_spacing" in z:
            loader.channel_spacing = float(z["channel_spacing"])

    assert loader.data is not None
    assert loader.data.ndim == 2
    assert loader.time is not None
    assert loader.distance is not None

    # keep test fast: subset
    data = loader.data[:256, :4000]

    pre = DASPreprocessor(sampling_rate=loader.sampling_rate, channel_spacing=loader.channel_spacing)
    processed = (
        pre.set_data(data)
        .remove_mean()
        .remove_trend(order=1)
        .bandpass_filter(2.0, 80.0)
        .normalize(method="std")
        .get_data()
    )

    assert processed.shape == data.shape
    assert np.isfinite(processed).all()

    detector = EventDetector(sampling_rate=loader.sampling_rate, channel_spacing=loader.channel_spacing)
    events = detector.sta_lta_detect(
        processed,
        sta_window=0.03,
        lta_window=0.5,
        trigger_on=3.0,
        min_channels=10,
        min_duration=0.02,
    )

    # Not asserting >0 (depends on random seed / subset); just ensure it runs.
    assert isinstance(events, list)
