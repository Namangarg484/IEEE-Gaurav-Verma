"""
Integrity tests for the PPG-glucose pipeline.

Run:  ~/.venvs/ppgfusion/bin/python -m pytest tests -q

They guard against the failure modes that invalidated the first draft:
label/PPG cross-joins, subject leakage across folds, mis-scaled timestamps.
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from ppg_data import FS, WIN, MENDELEY_ROOT, BIGIDEAS_DIR, load_mendeley, make_views, handcrafted_features  # noqa: E402


@pytest.fixture(scope="module")
def mendeley():
    return load_mendeley()


def test_mendeley_counts(mendeley):
    assert len(np.unique(mendeley.record)) == 67
    assert len(np.unique(mendeley.subject)) == 23
    assert mendeley.X.shape == (len(mendeley), WIN)


def test_one_label_per_recording_matches_label_file(mendeley):
    for rec in np.unique(mendeley.record):
        ys = np.unique(mendeley.y[mendeley.record == rec])
        assert len(ys) == 1, f"{rec} has several labels"
        lab = pd.read_csv(os.path.join(MENDELEY_ROOT, "Labels", f"label_{rec}.csv")).iloc[0]
        assert float(lab["Glucose"]) == ys[0]


def test_windows_are_zscored_and_pulsatile(mendeley):
    assert np.allclose(mendeley.X.mean(1), 0, atol=1e-4)
    assert np.allclose(mendeley.X.std(1), 1, atol=1e-3)
    hr = handcrafted_features(mendeley.X)[:, 0]
    assert np.median(hr) > 45 and np.median(hr) < 120, "dominant frequency is not a plausible heart rate"


def test_within_subject_variation_exists(mendeley):
    """The corrupted CSV collapsed labels to per-subject means; the raw data must not."""
    spread = mendeley.meta.groupby("subject").glucose.agg(lambda g: g.max() - g.min())
    assert (spread > 0).sum() >= 20


def test_loso_folds_never_leak_subjects(mendeley):
    from sklearn.model_selection import LeaveOneGroupOut
    for tr, te in LeaveOneGroupOut().split(mendeley.X, mendeley.y, mendeley.subject):
        assert not set(mendeley.subject[tr]) & set(mendeley.subject[te])
        assert not set(mendeley.record[tr]) & set(mendeley.record[te])


def test_view_shapes(mendeley):
    v = make_views(mendeley.X[:3])
    assert tuple(v["x1"].shape) == (3, 1, WIN)
    assert tuple(v["x2"].shape) == (3, 1, 64, 64)
    assert tuple(v["x3"].shape) == (3, 3, 256)
    assert all(np.isfinite(t.numpy()).all() for t in v.values())


@pytest.mark.skipif(not glob.glob(os.path.join(BIGIDEAS_DIR, "sub_*.npz")), reason="BIG IDEAs not extracted")
def test_bigideas_timestamps_are_seconds():
    for f in sorted(glob.glob(os.path.join(BIGIDEAS_DIR, "sub_*.npz"))):
        d = np.load(f)
        t = d["egv_time"]
        assert abs(np.median(np.diff(t)) - 300) < 5, f"{f}: CGM spacing is not 300 s (timestamp units wrong?)"
        assert (t[-1] - t[0]) > 86400, f"{f}: recording shorter than a day"
        assert d["bvp"].shape[1] == 120 * FS
        assert np.isfinite(d["bvp"]).all(1).mean() >= 0.2, f"{f}: too few gap-free BVP windows"


CLEAN_CSV = os.path.join(ROOT, "Archive_Dataset", "clean-dataset.csv")


@pytest.mark.skipif(not os.path.exists(CLEAN_CSV), reason="clean-dataset.csv absent")
def test_clean_dataset_csv_is_a_cross_join():
    """Documents why clean-dataset.csv must never be used: within a patient the
    glucose label cycles row by row through the patient's label list, so every
    window's mean label collapses to the patient mean."""
    df = pd.read_csv(CLEAN_CSV, usecols=["Patient_Id", "Glucose_level"])          # whole file (844,946 rows)
    checked = 0
    for pid, g in df.groupby("Patient_Id"):
        lab = g.Glucose_level.to_numpy()
        k = len(np.unique(lab))
        if k < 2:
            continue                                                             # a single label cannot cycle
        frac = (lab[k:] == lab[:-k]).mean()
        assert frac > 0.98, f"patient {pid}: labels do not cycle with period {k} (match fraction {frac:.3f})"
        assert len(np.unique(lab[:k])) == k, f"patient {pid}: first {k} rows should carry all {k} labels"
        checked += 1
    assert checked >= 15, f"only {checked} patients had >1 label; expected most of the 22"
