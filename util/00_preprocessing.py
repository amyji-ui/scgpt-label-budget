#!/usr/bin/env python
"""Convert the pancreas .h5ad to plain files and build train/test and label budget splits.

Run from the project root:
    .venv/bin/python util/00_preprocessing.py

Outputs, under data/processed/ by default:
    X_lognorm.npz, counts.npz   Sparse CSR matrices, cells x genes. Load with scipy.sparse.load_npz.
    obs.csv, var.csv            Cell metadata and gene names, in matrix row and column order.
    splits/seed_<S>.csv         One row per cell, in matrix row order:
                                  cell_id, tech
                                  split              train or test
                                  budget_<N>, budget_all
                                                     fit, val or unused
    split_summary.csv           Cell type counts and fractions for each seed and budget.
    manifest.json               Settings, source file checksum and split sizes.

The test set is whole held-out batches, so no batch appears in both training and test, and
it is the same for every seed. Each seed draws its own budgets from the training batches.

Validation cells count toward the budget: budget_500 has 500 fit + val cells in total.

Budgets follow the training cell type fractions, except that every type gets at least
--min-per-type cells where the training pool has that many. This lifts rare types above their
natural share. Sampling uses training cell labels only.
"""

import argparse
import hashlib
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

LABEL_COL = "celltype"
BATCH_COL = "tech"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, default=Path("data/pancreas.h5ad"))
    p.add_argument("--outdir", type=Path, default=Path("data/processed"))
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    p.add_argument("--budgets", type=int, nargs="+", default=[500, 1000, 5000],
                   help="Label budgets. All training cells are always added as budget_all.")
    p.add_argument("--test-batches", nargs="+", default=["smartseq2"], help="Batches held out as the test set.")
    p.add_argument("--min-per-type", type=int, default=5,
                   help="Minimum cells per type in every budget. Must be at least the number of folds for "
                        "stratified CV.")
    p.add_argument("--val-frac", type=float, default=0.1, help="Share of each budget used for validation.")
    return p.parse_args()


def sha256(path, chunk_size=2**20):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def export_matrices(adata, outdir):
    """Write both expression matrices and the metadata in row and column order."""
    sp.save_npz(outdir / "X_lognorm.npz", sp.csr_matrix(adata.X, dtype=np.float32))
    sp.save_npz(outdir / "counts.npz", sp.csr_matrix(adata.layers["counts"], dtype=np.float32))
    adata.obs.rename_axis("cell_id").to_csv(outdir / "obs.csv")
    adata.var.rename_axis("gene").to_csv(outdir / "var.csv")


def split_test(batches, test_batches):
    """Return a boolean mask of test cells: every cell of the held-out batches."""
    unknown = set(test_batches) - set(batches)
    if unknown:
        raise ValueError(f"Unknown test batches: {sorted(unknown)}")
    is_test = batches.isin(test_batches).to_numpy()
    if is_test.all():
        raise ValueError("The test batches cover the whole dataset, leaving no training cells.")
    return is_test


def allocate(class_sizes, budgets, min_per_type):
    """Give each budget per-type counts close to the pool's fractions.

    Every type gets at least min_per_type cells, or all of its cells if it has fewer.
    Counts only grow as the budget grows, which is what allows the budgets to be nested.
    """
    counts = np.minimum(min_per_type, class_sizes)
    if budgets[0] < counts.sum():
        raise ValueError(f"Budget {budgets[0]} is smaller than the {counts.sum()} cells the minimums require.")
    if budgets[-1] > class_sizes.sum():
        raise ValueError(f"Budget {budgets[-1]} exceeds the {class_sizes.sum()} training cells.")

    fractions = class_sizes / class_sizes.sum()
    allocations = {}
    for budget in budgets:
        target = fractions * budget
        while counts.sum() < budget:
            deficit = target - counts
            deficit[counts >= class_sizes] = -np.inf
            counts[np.argmax(deficit)] += 1
        allocations[budget] = counts.copy()
    return allocations


def build_budgets(labels, is_test, budgets, min_per_type, val_frac, rng):
    """Return {budget name: array of fit, val or unused}, with nested budgets."""
    train_idx = np.flatnonzero(~is_test)
    cell_types, class_sizes = np.unique(labels[train_idx], return_counts=True)
    allocations = allocate(class_sizes, sorted(budgets) + [int(class_sizes.sum())], min_per_type)
    shuffled = [rng.permutation(train_idx[labels[train_idx] == t]) for t in cell_types]

    roles = {}
    for budget, counts in allocations.items():
        name = "all" if budget == class_sizes.sum() and budget not in budgets else str(budget)
        role = np.full(len(labels), "unused", dtype=object)
        for cells, n_cells in zip(shuffled, counts):
            # Every type with at least two cells appears in both fit and val.
            n_val = min(max(round(n_cells * val_frac), 1), n_cells - 1)
            role[cells[:n_cells]] = "fit"
            role[cells[:n_val]] = "val"
        roles[f"budget_{name}"] = role
    return roles


def check_split(table, labels, min_per_type):
    """Stop rather than save a split that breaks a rule the comparison depends on."""
    test = table["split"].eq("test").to_numpy()
    shared = set(table.loc[test, "tech"]) & set(table.loc[~test, "tech"])
    assert not shared, f"Batches in both training and test: {sorted(shared)}"

    train_types, train_sizes = np.unique(labels[~test], return_counts=True)
    required = dict(zip(train_types, np.minimum(min_per_type, train_sizes)))
    budget_cols = [c for c in table.columns if c.startswith("budget_")]
    previous = None
    for col in budget_cols:
        selected = table[col].ne("unused").to_numpy()
        assert not (selected & test).any(), f"{col} contains test cells"
        counts = pd.Series(labels[selected]).value_counts()
        for cell_type, n_required in required.items():
            assert counts.get(cell_type, 0) >= n_required, f"{col} has too few {cell_type} cells"
        if col != "budget_all":
            assert selected.sum() == int(col.removeprefix("budget_")), f"{col} has the wrong size"
        if previous is not None:
            assert not (previous & ~selected).any(), f"{col} does not contain the smaller budget"
        previous = selected
    assert previous.sum() == (~test).sum(), "budget_all is not all training cells"


def summarize(table, labels, seed):
    """Return per-type counts and fractions for one seed."""
    whole = pd.Series(labels).value_counts(normalize=True)
    rows = []
    groups = {"test": table["split"].eq("test").to_numpy()}
    groups |= {c: table[c].ne("unused").to_numpy() for c in table.columns if c.startswith("budget_")}
    for group, mask in groups.items():
        counts = pd.Series(labels[mask]).value_counts()
        for cell_type in whole.index:
            n_cells = int(counts.get(cell_type, 0))
            rows.append({
                "seed": seed, "group": group, "celltype": cell_type, "n_cells": n_cells,
                "fraction": n_cells / mask.sum(), "whole_dataset_fraction": whole[cell_type],
            })
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    (args.outdir / "splits").mkdir(parents=True, exist_ok=True)

    adata = ad.read_h5ad(args.input)
    export_matrices(adata, args.outdir)
    labels = adata.obs[LABEL_COL].astype(str).to_numpy()

    is_test = split_test(adata.obs[BATCH_COL].astype(str), args.test_batches)
    train_types, train_sizes = np.unique(labels[~is_test], return_counts=True)
    too_few = {str(t): int(n) for t, n in zip(train_types, train_sizes) if n < args.min_per_type}
    if too_few:
        print(f"WARNING: fewer than {args.min_per_type} training cells, so all are used: {too_few}")
    absent = sorted(set(labels) - set(train_types))
    if absent:
        print(f"WARNING: cell types only in the test set, which no method can learn: {absent}")
    absent = sorted(set(train_types) - set(labels[is_test]))
    if absent:
        print(f"Cell types in training but not in the test set: {absent}")

    summaries, sizes = [], {}
    for seed in args.seeds:
        rng = np.random.default_rng(seed)
        table = pd.DataFrame({
            "cell_id": adata.obs_names,
            "tech": adata.obs[BATCH_COL].astype(str).to_numpy(),
            "split": np.where(is_test, "test", "train"),
        })
        roles = build_budgets(labels, is_test, args.budgets, args.min_per_type, args.val_frac, rng)
        table = table.assign(**roles)
        check_split(table, labels, args.min_per_type)
        table.to_csv(args.outdir / "splits" / f"seed_{seed}.csv", index=False)
        summaries.append(summarize(table, labels, seed))

        sizes[seed] = {"test": int(is_test.sum()), "train": int((~is_test).sum())}
        for col in roles:
            sizes[seed][col] = {r: int((table[col] == r).sum()) for r in ("fit", "val")}
        print(f"seed {seed}: {sizes[seed]}")

    pd.concat(summaries).to_csv(args.outdir / "split_summary.csv", index=False)

    settings = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    manifest = {
        "settings": settings,
        "source_sha256": sha256(args.input),
        "shape": list(adata.shape),
        "split_sizes": sizes,
    }
    (args.outdir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {args.outdir}")


if __name__ == "__main__":
    main()
