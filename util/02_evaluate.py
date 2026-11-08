#!/usr/bin/env python
"""Score saved test predictions with one shared function. This is the frozen evaluation script.

Run from the project root:
    .venv/bin/python util/02_evaluate.py                 # score every finished run under results/
    .venv/bin/python util/02_evaluate.py --freeze        # record this file's hash (once, before scoring)

Every method, whether scGPT or a baseline, writes a run folder
    results/test_<batch>/<method>/budget_<B>/seed_<S>/
containing
    predictions_test.csv   columns cell_id, predicted; one row per test cell
    run.json               written last; needs "seed" and "budget", and the data folder as
                           "data_dir" or settings.data_dir (default data/processed)
Folders without run.json are treated as unfinished and skipped.

Scoring rules (decided by Amy, 2026-11-07):
    Macro F1 averages over every cell type present in the test set, however few cells it has.
    A prediction of a type absent from the test set counts against the recall of the true type
    but adds no class of its own to the average.
Test labels come from data/processed/obs.csv and are used here only.

Outputs:
    <run>/metrics.json                       Overall and per-type metrics for one run.
    results/test_<batch>/metrics_summary.csv One row per run.
    results/test_<batch>/metrics_per_type.csv One row per run and cell type.

Freeze: --freeze writes util/02_evaluate.sha256. After that, scoring stops if this file has
changed, so every result is scored by the same code. Changing it means deleting the hash file,
freezing again and rescoring everything.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

SCRIPT = Path(__file__).resolve()
HASH_FILE = SCRIPT.with_suffix(".sha256")
DEFAULT_DATA_DIR = Path("data/processed")
USD_PER_GPU_HOUR = 1.006  # g5.xlarge on-demand, us-east-1


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", type=Path, default=Path("results"))
    p.add_argument("--freeze", action="store_true", help="Record this script's hash and exit.")
    p.add_argument("--allow-unfrozen", action="store_true",
                   help="Score without a matching frozen hash, for testing only. Marked in every output.")
    return p.parse_args()


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def score(y_true, y_pred):
    """The one scoring function. Returns overall metrics and a per-type table."""
    y_true = np.asarray(y_true, dtype=str)
    y_pred = np.asarray(y_pred, dtype=str)
    present = sorted(set(y_true))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=present, zero_division=0)
    n_predicted = pd.Series(y_pred).value_counts()
    per_type = pd.DataFrame({
        "celltype": present, "n_test": support,
        "n_predicted": [int(n_predicted.get(t, 0)) for t in present],
        "precision": precision, "recall": recall, "f1": f1,
    })
    absent_pred = ~np.isin(y_pred, present)
    overall = {
        "macro_f1": float(f1_score(y_true, y_pred, labels=present, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=present, average="weighted", zero_division=0)),
        "n_test": int(len(y_true)),
        "n_types_scored": len(present),
        "n_predicted_absent_type": int(absent_pred.sum()),
        "predicted_absent_types": sorted(set(y_pred[absent_pred])),
    }
    return overall, per_type


def load_truth(run_dir, run, cache):
    """Return test labels in split order, checked against the run's split file."""
    data_dir = Path(run.get("data_dir") or run.get("settings", {}).get("data_dir") or DEFAULT_DATA_DIR)
    key = (data_dir, int(run["seed"]))
    if key not in cache:
        obs = pd.read_csv(data_dir / "obs.csv", index_col="cell_id")
        split = pd.read_csv(data_dir / "splits" / f"seed_{run['seed']}.csv")
        test = split.loc[split["split"] == "test"]
        cache[key] = (obs.loc[test["cell_id"], "celltype"].astype(str),
                      sorted(test["tech"].astype(str).unique()))
    return cache[key]


def check_predictions(pred, truth, run_dir):
    """Stop rather than score predictions that don't cover exactly the test cells."""
    assert not pred["cell_id"].duplicated().any(), f"{run_dir}: duplicate cell ids"
    missing = set(truth.index) - set(pred["cell_id"])
    extra = set(pred["cell_id"]) - set(truth.index)
    assert not missing and not extra, f"{run_dir}: {len(missing)} test cells missing, {len(extra)} extra cells"
    assert pred["predicted"].notna().all(), f"{run_dir}: empty predictions"


def find_runs(results):
    for path in sorted(results.glob("test_*/*/budget_*/seed_*/run.json")):
        if (path.parent / "predictions_test.csv").exists():
            yield path.parent
        else:
            print(f"Skipping {path.parent}: run.json but no predictions_test.csv")


def evaluate_run(run_dir, script_hash, frozen, cache):
    run = json.loads((run_dir / "run.json").read_text())
    truth, test_batches = load_truth(run_dir, run, cache)
    folder_batch = run_dir.parents[2].name.removeprefix("test_")
    assert folder_batch == "+".join(test_batches), f"{run_dir}: folder says {folder_batch}, split says {test_batches}"

    pred = pd.read_csv(run_dir / "predictions_test.csv", usecols=["cell_id", "predicted"])
    check_predictions(pred, truth, run_dir)
    y_pred = pred.set_index("cell_id").loc[truth.index, "predicted"].astype(str)
    overall, per_type = score(truth.to_numpy(), y_pred.to_numpy())

    train_s = run.get("train_seconds")
    infer_s = run.get("test_inference_seconds")
    gpu_hours = ((train_s or 0) + (infer_s or 0)) / 3600 if train_s is not None else None
    meta = {
        "test_batches": "+".join(test_batches),
        "method": run_dir.parents[1].name,
        "budget": str(run["budget"]),
        "seed": int(run["seed"]),
        "n_fit": run.get("n_fit"), "n_val": run.get("n_val"),
        "train_seconds": train_s, "test_inference_seconds": infer_s,
        "gpu_hours": gpu_hours,
        "cost_usd": gpu_hours * USD_PER_GPU_HOUR if gpu_hours is not None else None,
        "peak_gpu_gib": run.get("peak_gpu_gib"),
        "eval_script_sha256": script_hash, "eval_frozen": frozen,
    }
    record = {**meta, **overall, "per_type": per_type.to_dict(orient="records")}
    (run_dir / "metrics.json").write_text(json.dumps(record, indent=2) + "\n")
    return {**meta, **overall}, per_type.assign(**{k: meta[k] for k in ("method", "budget", "seed")})


def budget_order(budget):
    return float("inf") if budget == "all" else float(budget)


def main():
    args = parse_args()
    script_hash = file_sha256(SCRIPT)
    if args.freeze:
        if HASH_FILE.exists():
            raise SystemExit(f"Already frozen at {HASH_FILE.read_text().strip()}. Delete {HASH_FILE} to refreeze, "
                             "then rescore everything.")
        HASH_FILE.write_text(script_hash + "\n")
        print(f"Frozen: {script_hash}")
        return

    frozen = HASH_FILE.exists() and HASH_FILE.read_text().strip() == script_hash
    if not frozen and not args.allow_unfrozen:
        raise SystemExit("The evaluation script is not frozen, or has changed since it was frozen. "
                         "Run with --freeze first, or --allow-unfrozen for a test.")
    if not frozen:
        print("WARNING: scoring with an unfrozen evaluation script. Outputs are marked eval_frozen=false.")

    cache, by_batch = {}, {}
    for run_dir in find_runs(args.results):
        summary, per_type = evaluate_run(run_dir, script_hash, frozen, cache)
        by_batch.setdefault(summary["test_batches"], []).append((summary, per_type))
        print(f"{summary['method']:>18} budget {summary['budget']:>5} seed {summary['seed']}: "
              f"macro F1 {summary['macro_f1']:.3f}, accuracy {summary['accuracy']:.3f}")

    for batch, rows in by_batch.items():
        summary = pd.DataFrame([r[0] for r in rows])
        summary = summary.assign(_b=summary["budget"].map(budget_order)).sort_values(["method", "_b", "seed"])
        per_type = pd.concat([r[1] for r in rows])
        per_type = per_type.assign(_b=per_type["budget"].map(budget_order)).sort_values(
            ["method", "_b", "seed", "celltype"])
        out = args.results / f"test_{batch}"
        summary.drop(columns="_b").to_csv(out / "metrics_summary.csv", index=False)
        per_type.drop(columns="_b").to_csv(out / "metrics_per_type.csv", index=False)
        print(f"Wrote {out}/metrics_summary.csv and metrics_per_type.csv ({len(summary)} runs)")
    if not by_batch:
        print(f"No finished runs under {args.results}")


if __name__ == "__main__":
    main()
