#!/usr/bin/env python
"""Embed every cell with Harmony-corrected principal components (the classical baseline).

Run from the project root:
    .venv/bin/python util/05_harmony_embed.py
Then fit logistic regression on the embeddings:
    .venv/bin/python util/03_logreg.py --features harmony

Pipeline, all without labels:
    1. Normalization  The dataset's log-normalized matrix (X_lognorm.npz), the same input as the HVG
                      baseline. The counts layer is not used: it is non-integer in four batches.
    2. Genes          Genes in the scGPT vocabulary (as for every method), then the same 2,000 HVGs as
                      03_logreg.py --features hvg: scanpy flavor "seurat", batch_key tech, chosen on
                      training-pool cells only.
    3. Scaling        Each gene centred and scaled to unit variance over all cells, clipped at 10
                      (scanpy's standard sc.pp.scale(max_value=10)).
    4. PCA            50 components, fitted on all cells.
    5. Harmony        harmonypy with batch = tech, default settings (theta 2, about 100 clusters).

Steps 3-5 use every cell, test cells included, because Harmony corrects cells jointly and the test
batch has to be corrected toward the training batches. Only expression and batch names are used,
never labels (open decision 3: unlabeled test cells allowed for all methods).

No step depends on the label budget, so one file per seed serves every budget. The seed only sets
Harmony's clustering initialization.
    results/embeddings/<data dir name>/harmony/seed_<S>.npz    cell_ids, embeddings (Harmony),
                                                                pcs (uncorrected, for comparison)
    results/embeddings/<data dir name>/harmony/seed_<S>.json   settings, timing, variance explained
"""

import argparse
import importlib.util
import json
import time
from importlib.metadata import version
from pathlib import Path

import harmonypy
import numpy as np
import pandas as pd
import scanpy as sc

_spec = importlib.util.spec_from_file_location("logreg", Path(__file__).with_name("03_logreg.py"))
logreg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(logreg)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    p.add_argument("--checkpoint-dir", type=Path, default=Path("scGPT_human"), help="For the vocabulary.")
    p.add_argument("--outdir", type=Path, default=Path("results"))
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    p.add_argument("--n-hvg", type=int, default=2000)
    p.add_argument("--n-pcs", type=int, default=50)
    p.add_argument("--theta", type=float, default=2.0, help="Harmony's batch diversity penalty.")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = args.outdir / "embeddings" / args.data_dir.name / "harmony"
    out_dir.mkdir(parents=True, exist_ok=True)
    hvg = logreg.HVGFeatures(args.data_dir, args.checkpoint_dir, args.n_hvg)
    obs = hvg.obs

    cached = None
    for seed in args.seeds:
        path = out_dir / f"seed_{seed}.npz"
        if path.exists() and not args.overwrite:
            print(f"Skipping {path}: already done")
            continue
        split = pd.read_csv(args.data_dir / "splits" / f"seed_{seed}.csv")
        assert (split["cell_id"].to_numpy() == obs.index.to_numpy()).all(), "split and obs rows differ"
        train_mask = split["split"].eq("train").to_numpy()

        started = time.time()
        key = train_mask.tobytes()
        if cached is None or cached[0] != key:
            # Steps 2-4 depend only on the training pool, which is the same for every seed.
            cols = hvg.hvgs(train_mask)
            x = hvg.matrix[:, cols].toarray().astype(np.float32)
            x -= x.mean(axis=0)
            std = x.std(axis=0)
            x /= np.where(std > 0, std, 1)
            np.clip(x, None, 10, out=x)
            pca = sc.pp.pca(x, n_comps=args.n_pcs, zero_center=True, random_state=0, return_info=True)
            pcs, variance_ratio = pca[0].astype(np.float32), pca[2]
            cached = (key, cols, pcs, variance_ratio)
        _, cols, pcs, variance_ratio = cached
        pca_seconds = time.time() - started

        started = time.time()
        meta = obs[["tech"]].astype(str).reset_index(drop=True)
        result = harmonypy.run_harmony(pcs, meta, "tech", theta=args.theta, random_state=seed, verbose=False)
        corrected = np.asarray(result.Z_corr)
        if corrected.shape != pcs.shape:
            corrected = corrected.T
        assert corrected.shape == pcs.shape and np.isfinite(corrected).all(), "unexpected Harmony output"
        harmony_seconds = time.time() - started

        np.savez_compressed(path, cell_ids=obs.index.to_numpy().astype("U"),
                            embeddings=corrected.astype(np.float32), pcs=pcs)
        info = {
            "seed": seed, "n_cells": int(len(obs)), "n_hvg": int(len(cols)), "n_pcs": args.n_pcs,
            "pca_variance_explained": float(variance_ratio.sum()),
            "harmony_iterations": int(getattr(result, "max_iter_harmony", -1)),
            "pca_seconds": pca_seconds, "harmony_seconds": harmony_seconds,
            "settings": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "versions": {p: version(p) for p in ("harmonypy", "scanpy", "numpy")},
        }
        path.with_suffix(".json").write_text(json.dumps(info, indent=2) + "\n")
        print(f"seed {seed}: PCA {pca_seconds:.0f}s ({variance_ratio.sum():.1%} variance in {args.n_pcs} PCs), "
              f"Harmony {harmony_seconds:.0f}s, saved {path}", flush=True)


if __name__ == "__main__":
    main()
