"""
freeze_folds.py
===============
Export the benchmark's split protocols as frozen fold files so that any future
model can be evaluated on exactly the same train/test indices, and so the
no-leak guarantee of each protocol is verified once and recorded.

    python scripts/freeze_folds.py                      # all datasets x primary protocols
    python scripts/freeze_folds.py --datasets mendeley --protocols subject_loso window_random

Writes results/folds/<dataset>__<protocol>__seed<k>.json with:
    protocol, level, gap_s/params, seed, fingerprint, n_folds, leak_report,
    folds: [{train_records: [...], test_records: [...]}]   (record ids, not window
    indices, so the file is stable across window hop/length choices)
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from ppgbench import load_dataset, make_folds  # noqa: E402
from ppgbench import protocols  # noqa: E402

DEFAULT = {
    "mendeley": ["subject_loso", "subject_kfold", "record_random", "window_random"],
    "stuba": ["subject_loso", "subject_kfold"],
    "bigideas": ["chronological", "chronological_nogap", "personal", "record_kfold", "subject_loso"],
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=list(DEFAULT))
    ap.add_argument("--protocols", nargs="*", default=None)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "folds"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for ds in args.datasets:
        kw = {"max_windows_per_egv": 2} if ds == "bigideas" else {}
        ws = load_dataset(ds, **kw)
        for proto in (args.protocols or DEFAULT[ds]):
            spec = protocols.resolve_protocol(proto)
            for seed in (args.seeds if spec.uses_seed else [0]):
                folds = make_folds(ws, proto, seed=seed, check=True)
                rep = protocols.leak_report(ws, folds, proto)
                entry = {
                    "dataset": ds, "protocol": proto, "level": spec.level, "seed": seed,
                    "fingerprint": protocols.fold_fingerprint(folds), "n_folds": len(folds),
                    "n_windows": int(len(ws)), "n_records": int(len(np.unique(ws.record))),
                    "n_subjects": int(len(np.unique(ws.subject))),
                    "leak_report": {k: v for k, v in rep.items() if k != "per_fold"},
                    "folds": [{"train_records": sorted(set(ws.record[tr].tolist())),
                               "test_records": sorted(set(ws.record[te].tolist()))} for tr, te in folds],
                }
                path = os.path.join(args.out, f"{ds}__{proto}__seed{seed}.json")
                json.dump(entry, open(path, "w"))
                print(f"{ds:9s} {proto:20s} seed {seed}  level={spec.level:8s} folds={len(folds):3d}  "
                      f"fingerprint={entry['fingerprint']}  -> {os.path.relpath(path, ROOT)}")


if __name__ == "__main__":
    main()
