"""
run_foundation_probes.py
========================
Frozen-encoder ridge probes of three open pretrained PPG foundation models,
evaluated under exactly the benchmark's protocols:

    PaPaGei-S   (Pillai et al., ICLR 2025; Nokia Bell Labs, BSD-3-Clause code;
                 weights Zenodo 13983110)        125 Hz, 10-s segments, 512-d
    Pulse-PPG   (Saha et al., IMWUT 2025; MIT; weights Zenodo 17345536)
                                                  50 Hz, any length (temporal max-pool), 512-d
    AnyPPG      (Nie et al., KDD 2026; checkpoint shipped in its GitHub repo)
                                                  125 Hz, 10-s segments, 512-d

The downloads live in data.nosync/encoders/ (repositories) and
data.nosync/encoders/weights/ (checkpoints).  Only the model-definition files
are imported (each was read before use: they import torch/numpy/sklearn/tqdm/
matplotlib and define nn.Modules only), and every checkpoint is loaded with
``torch.load(..., weights_only=True)``, which refuses arbitrary pickled objects.

Inputs per cohort (native signal -> encoder input)
--------------------------------------------------
* Kermani-Esmaeili: the raw 10-s, 2175-Hz recording (one segment per record).
* STUBA ePPG: the 30-s, 100-Hz IR1 record, cut into three 10-s segments
  (Pulse-PPG: the whole 30 s).
* BIG IDEAs: the 64-Hz BVP in the 60 s preceding each CGM reading; the two
  most recent gap-free 10-s segments (Pulse-PPG: the most recent gap-free 20 s).
Each segment is band-passed 0.5-12 Hz (4th-order Chebyshev type II, 20 dB,
zero phase, as in PaPaGei's pyPPG preprocessing) at the native rate,
resampled to the encoder's rate and z-scored; segment embeddings are averaged
per reference measurement.

Probe: RidgeCV (alphas 1e-2..1e4) on embeddings standardised with training
statistics only.  Protocols: leave-one-subject-out (Kermani-Esmaeili, STUBA),
chronological split with 1-h guard gap (BIG IDEAs; absolute and residual
targets).  Results are written in the same files and formats as the other
benchmark rows, so bootstrap_ci.py / stats_extra.py / clinical_metrics.py /
make_results_macros.py pick them up unchanged.

Usage
-----
    python run_foundation_probes.py --encoders papagei anyppg pulseppg --datasets mendeley stuba bigideas
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import logging
import os
import sys
import time
from copy import copy

import numpy as np
import pandas as pd
import torch
from scipy import signal as sps
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from ppg_data import MENDELEY_ROOT, STUBA_ROOT, WindowSet, load_bigideas, load_mendeley, load_stuba  # noqa: E402
from run_bigideas_cv import chronological_split, subject_mean_baseline, subject_train_means  # noqa: E402
from run_subject_cv import OUT_DIR, merge_json, metrics  # noqa: E402

log = logging.getLogger("fm_probes")
ENC = os.path.join(ROOT, "data.nosync", "encoders")
SEG_S = 10.0


# --------------------------------------------------------------------------- #
# Encoders (model files imported by path; weights loaded weights_only=True)
# --------------------------------------------------------------------------- #
def _import(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _strip(sd: dict) -> dict:
    return {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}


def load_encoder(name: str) -> tuple[torch.nn.Module, int, str]:
    """Return (model in eval mode, input rate Hz, short licence note)."""
    if name == "papagei":
        m = _import(os.path.join(ENC, "papagei-foundation-model", "models", "resnet.py"), "papagei_resnet")
        net = m.ResNet1DMoE(in_channels=1, base_filters=32, kernel_size=3, stride=2, groups=1, n_block=18,
                            n_classes=512, n_experts=3)
        sd = torch.load(os.path.join(ENC, "weights", "papagei", "papagei_s.pt"), map_location="cpu", weights_only=True)
        net.load_state_dict(_strip(sd))
        return net.eval(), 125, "code BSD-3-Clause; weights Zenodo 13983110 (no licence stated)"
    if name == "anyppg":
        m = _import(os.path.join(ENC, "AnyPPG", "load_anyppg", "resnet1d.py"), "anyppg_resnet1d")
        net = m.Net1D(in_channels=1, base_filters=64, ratio=1.0, filter_list=[64, 160, 160, 400, 400, 512],
                      m_blocks_list=[2, 2, 2, 3, 3, 1], kernel_size=3, stride=2, groups_width=16,
                      use_bn=True, use_do=True, verbose=False)
        sd = torch.load(os.path.join(ENC, "AnyPPG", "load_anyppg", "anyppg_ckpt.pth"), map_location="cpu", weights_only=True)
        net.load_state_dict(sd)
        return net.eval(), 125, "checkpoint from GitHub PKUDigitalHealth/AnyPPG (no licence file)"
    if name == "pulseppg":
        m = _import(os.path.join(ENC, "pulseppg", "pulseppg", "nets", "ResNet1D", "ResNet1D_Net.py"), "pulseppg_resnet")
        net = m.Net(in_channels=1, base_filters=128, kernel_size=11, stride=2, groups=1, n_block=12, finalpool="max")
        ck = torch.load(os.path.join(ENC, "weights", "pulseppg", "pulseppg", "experiments", "out", "pulseppg",
                                     "checkpoint_best.pkl"), map_location="cpu", weights_only=True)
        net.load_state_dict(_strip(ck["net"]))
        return net.eval(), 50, "MIT (code and weights, Zenodo 17345536)"
    raise ValueError(name)


def embed(net: torch.nn.Module, name: str, X: np.ndarray, batch: int = 64) -> np.ndarray:
    """X: (n, L) preprocessed segments at the encoder's rate -> (n, 512)."""
    out = []
    with torch.inference_mode():
        for i in range(0, len(X), batch):
            x = torch.from_numpy(X[i:i + batch, None, :].astype(np.float32))
            y = net(x)
            y = y[0] if name == "papagei" else y          # PaPaGei returns (embedding, moe1, moe2, pooled); index 0 per its code
            out.append(y.cpu().numpy())
    return np.concatenate(out)


# --------------------------------------------------------------------------- #
# Preprocessing
# --------------------------------------------------------------------------- #
def prep(x: np.ndarray, fs_in: float, fs_out: int) -> np.ndarray:
    """Cheby-II 0.5-12 Hz band-pass at native rate, resample, z-score."""
    sos = sps.cheby2(4, 20, [0.5, min(12.0, 0.45 * fs_in)], btype="bandpass", fs=fs_in, output="sos")
    y = sps.sosfiltfilt(sos, np.asarray(x, np.float64) - np.mean(x))
    n_out = int(round(len(y) * fs_out / fs_in))
    y = sps.resample(y, n_out)
    return ((y - y.mean()) / (y.std() + 1e-8)).astype(np.float32)


def segments(x: np.ndarray, fs: float, seg_s: float) -> list[np.ndarray]:
    n = int(round(seg_s * fs))
    return [x[i * n:(i + 1) * n] for i in range(len(x) // n)]


# --------------------------------------------------------------------------- #
# Per-record embeddings for each cohort
# --------------------------------------------------------------------------- #
def records_mendeley(net, name, fs_enc):
    ws = load_mendeley()
    rows, emb = [], []
    for _, r in ws.meta.iterrows():
        raw = pd.read_csv(os.path.join(MENDELEY_ROOT, "RawData", f"signal_{r.record}.csv"), header=None).to_numpy().ravel()
        seg = prep(raw, 2175, fs_enc)
        if name != "pulseppg":
            seg = seg[: int(SEG_S * fs_enc)]
        emb.append(embed(net, name, seg[None, :]).mean(0))
        rows.append(r)
    return pd.DataFrame(rows).reset_index(drop=True), np.stack(emb)


def records_stuba(net, name, fs_enc):
    import wfdb
    ws = load_stuba(channel="IR1")
    rec_paths = {f"{int(l.split('/')[0].split('_')[1]):02d}_{l.strip().split('/')[-1][-4:]}": l.strip()
                 for l in open(os.path.join(STUBA_ROOT, "RECORDS")).read().replace("\\", "/").splitlines() if l.strip()}
    rows, emb = [], []
    for _, r in ws.meta.iterrows():
        sig = wfdb.rdrecord(os.path.join(STUBA_ROOT, rec_paths[r.record]))
        raw = sig.p_signal[:, sig.sig_name.index("IR1")]
        x = prep(raw, 100, fs_enc)
        segs = [x] if name == "pulseppg" else segments(x, fs_enc, SEG_S)
        segs = [(s - s.mean()) / (s.std() + 1e-8) for s in segs]
        emb.append(embed(net, name, np.stack(segs)).mean(0))
        rows.append(r)
    return pd.DataFrame(rows).reset_index(drop=True), np.stack(emb)


def records_bigideas(net, name, fs_enc):
    ws = load_bigideas(max_windows_per_egv=2)
    keep = set(ws.meta.record)
    rows, emb, seg_owner, segs = [], [], [], []
    for f in sorted(glob.glob(os.path.join(ROOT, "data.nosync", "bigideas", "sub_*.npz"))):
        sid = os.path.basename(f)[4:7]
        d = np.load(f)
        bvp = d["bvp"]
        pre = bvp.shape[1] // 2
        for i in range(len(d["glucose"])):
            key = f"{sid}_{i:05d}"
            if key not in keep:
                continue
            seg_all = bvp[i, pre - 60 * 64:pre]
            n10 = 10 * 64
            parts = [seg_all[j * n10:(j + 1) * n10] for j in range(6)]
            ok = [p for p in parts if np.isfinite(p).all() and p.std() > 1e-3]
            if not ok:
                continue
            if name == "pulseppg":
                last2 = ok[-2:]
                chosen = [np.concatenate(last2)] if len(last2) == 2 else last2
            else:
                chosen = ok[-2:]
            for p in chosen:
                segs.append(prep(p, 64, fs_enc)); seg_owner.append(len(rows))
            rows.append(key)
    log.info("%s BIG IDEAs: %d readings, %d segments", name, len(rows), len(segs))
    E = np.zeros((len(rows), 0))
    lens = {len(s) for s in segs}
    chunks = []
    for L in sorted(lens):                                   # batch by length (Pulse-PPG may mix 10/20 s)
        idx = [k for k, s in enumerate(segs) if len(s) == L]
        chunks.append((idx, embed(net, name, np.stack([segs[k] for k in idx]))))
    D = chunks[0][1].shape[1]
    S = np.zeros((len(segs), D), np.float32)
    for idx, e in chunks:
        S[idx] = e
    owner = np.array(seg_owner)
    E = np.stack([S[owner == k].mean(0) for k in range(len(rows))])
    meta = ws.meta.set_index("record").loc[rows].reset_index()
    return meta, E


# --------------------------------------------------------------------------- #
# Probing under the benchmark protocols
# --------------------------------------------------------------------------- #
def ridge_fit_predict(E, y, tr, te):
    sc = StandardScaler().fit(E[tr])
    m = RidgeCV(alphas=np.logspace(-2, 4, 13)).fit(sc.transform(E[tr]), y[tr])
    return m.predict(sc.transform(E[te])).astype(np.float32)


def as_windowset(meta: pd.DataFrame, E: np.ndarray, source: str) -> WindowSet:
    return WindowSet(E.astype(np.float32), meta.glucose.to_numpy(np.float32), meta.subject.astype(str).to_numpy(),
                     meta.record.astype(str).to_numpy(), meta, source=source)


def run_loso(ws: WindowSet) -> pd.DataFrame:
    pred = np.full(len(ws), np.nan, np.float32)
    for tr, te in LeaveOneGroupOut().split(ws.X, ws.y, ws.subject):
        pred[te] = ridge_fit_predict(ws.X, ws.y, tr, te)
    return pd.DataFrame({"record": ws.record, "subject": ws.subject, "y": ws.y, "pred": pred, "n_windows": 1})


def run_chrono(ws: WindowSet, residual: bool) -> pd.DataFrame:
    tr, te = chronological_split(ws)
    y_fit = ws.y.copy()
    off = 0.0
    if residual:
        mu = subject_train_means(ws, tr)
        y_fit[tr] = ws.y[tr] - np.array([mu[s] for s in ws.subject[tr]], np.float32)
        off = np.array([mu.get(s, ws.y[tr].mean()) for s in ws.subject[te]], np.float32)
    pred = ridge_fit_predict(ws.X, y_fit, tr, te) + off
    return pd.DataFrame({"record": ws.record[te], "subject": ws.subject[te], "y": ws.y[te], "pred": pred, "n_windows": 1})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--encoders", nargs="+", default=["papagei", "anyppg", "pulseppg"])
    ap.add_argument("--datasets", nargs="+", default=["mendeley", "stuba", "bigideas"])
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))
    os.makedirs(OUT_DIR, exist_ok=True)
    info_path = os.path.join(OUT_DIR, "foundation_encoders.json")
    info = json.load(open(info_path)) if os.path.exists(info_path) else {}

    for name in args.encoders:
        t0 = time.time()
        net, fs_enc, lic = load_encoder(name)
        n_par = sum(p.numel() for p in net.parameters())
        info[name] = {"input_rate_hz": fs_enc, "params": int(n_par), "licence": lic,
                      "segment_s": "whole record / 20 s" if name == "pulseppg" else SEG_S}
        json.dump(info, open(info_path, "w"), indent=2)
        log.info("loaded %s (%d params, %d Hz) in %.0f s", name, n_par, fs_enc, time.time() - t0)
        for ds in args.datasets:
            t0 = time.time()
            meta, E = {"mendeley": records_mendeley, "stuba": records_stuba, "bigideas": records_bigideas}[ds](net, name, fs_enc)
            np.savez(os.path.join(OUT_DIR, f"embeddings_{name}_{ds}.npz"), E=E, record=meta.record.astype(str).to_numpy())
            ws = as_windowset(meta, E, f"{ds}/{name}")
            if ds == "bigideas":
                for residual in (False, True):
                    df = run_chrono(ws, residual)
                    tag = f"within_{name}{'_res' if residual else ''}_seed0"
                    df.to_csv(os.path.join(OUT_DIR, f"bigideas_{tag}.csv"), index=False)
                    m = metrics(df)
                    m.update(model=name, protocol="within", target="residual" if residual else "absolute", seed=0,
                             encoder_params=int(n_par), seconds=round(time.time() - t0, 1))
                    merge_json(os.path.join(OUT_DIR, "bigideas_summary.json"), tag, m)
                    log.info("RESULT bigideas %-24s MAE %.2f  r %.3f  within-r %s", tag, m["mae"], m["pearson_r"],
                             m.get("within_subject_r_median"))
            else:
                df = run_loso(ws)
                prefix, summary = ("", "subject_cv_summary.json") if ds == "mendeley" else ("stuba_IR1_", "stuba_summary.json")
                tag = f"{prefix}{name}_seed0"
                df.to_csv(os.path.join(OUT_DIR, f"subject_cv_{tag}.csv"), index=False)
                m = metrics(df)
                m.update(model=name, folds="loso", seed=0, encoder_params=int(n_par), seconds=round(time.time() - t0, 1))
                merge_json(os.path.join(OUT_DIR, summary), tag, m)
                log.info("RESULT %-9s %-22s MAE %.2f  r %.3f  zoneA %.1f", ds, tag, m["mae"], m["pearson_r"], m["zone_A_pct"])


if __name__ == "__main__":
    main()
