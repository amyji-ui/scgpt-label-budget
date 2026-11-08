#!/usr/bin/env python
"""Embed every cell with scVI trained on raw counts (the deep baseline).

Run from the project root:
    .venv/bin/python util/06_scvi_embed.py                              # main run
    .venv/bin/python util/06_scvi_embed.py --exclude-batches smarter    # sensitivity run
Then fit logistic regression on the embeddings:
    .venv/bin/python util/03_logreg.py --features scvi
    .venv/bin/python util/03_logreg.py --features scvi_no_smarter

Counts (open decision 1, option A+, Amy, 2026-11-08):
    inDrop1-4, smartseq2   Integer counts, used as they are.
    celseq, celseq2        UMI counts stored after a saturation correction x = -256 ln(1 - k/256).
                           Inverted exactly to the integer UMI count k.
    fluidigmc1             Estimated read counts at 0.01 resolution. Rounded.
    smarter                Looks like RPKM, not counts. Rounded and kept, so every method sees the
                           same cells. Per-batch dispersion stops its inflated scale from making
                           scVI over-trust it, and the --exclude-batches run measures its effect.

Inputs, all without labels:
    Genes   The same 2,000 HVGs as the HVG and Harmony baselines (03_logreg.py), chosen on
            training-pool cells only.
    Cells   All cells, test batch included, with batch = tech (open decision 3). With
            --exclude-batches, those batches are left out entirely.

Model (scvi-tools 0.20.3; settings used for multi-batch integration benchmarks):
    n_latent 30, n_layers 2, n_hidden 128, gene_likelihood "nb", dispersion "gene-batch".
    Default epoch count (400 for this many cells), early stopping on the validation ELBO of a
    held-out 10% of cells. The seed sets initialization and the held-out cells.

Outputs, one file per seed, used for every budget:
    results/embeddings/<data dir name>/<scvi or scvi_no_<batches>>/seed_<S>.npz    cell_ids, embeddings
    .../seed_<S>.json          settings, count fixes, epochs, timing
    .../seed_<S>_history.csv   training and validation loss per epoch
"""

import argparse
import importlib.util
import json
import time
from importlib.metadata import version
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import scvi
import torch

_spec = importlib.util.spec_from_file_location("logreg", Path(__file__).with_name("03_logreg.py"))
logreg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(logreg)

UMI_CORRECTED = {"celseq": 256, "celseq2": 256}  # batch -> number of distinct UMIs


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    p.add_argument("--checkpoint-dir", type=Path, default=Path("scGPT_human"), help="For the vocabulary.")
    p.add_argument("--outdir", type=Path, default=Path("results"))
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    p.add_argument("--n-hvg", type=int, default=2000)
    p.add_argument("--exclude-batches", nargs="*", default=[])
    p.add_argument("--n-latent", type=int, default=30)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--n-hidden", type=int, default=128)
    p.add_argument("--gene-likelihood", default="nb")
    p.add_argument("--dispersion", default="gene-batch")
    p.add_argument("--max-epochs", type=int, default=None, help="Default: scvi-tools' own rule.")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def fix_counts(counts, batches):
    """Return integer counts per decision 1, and a per-batch report of what changed."""
    counts = counts.tocsr().astype(np.float64)
    report = {}
    for batch in np.unique(batches):
        rows = np.flatnonzero(batches == batch)
        start, end = counts.indptr[rows[0]], counts.indptr[rows[-1] + 1]
        assert (np.diff(rows) == 1).all(), f"{batch} rows are not contiguous"
        values = counts.data[start:end]
        if batch in UMI_CORRECTED:
            n = UMI_CORRECTED[batch]
            values = n * (1 - np.exp(-values / n))
            # The source caps saturated genes (all n UMIs seen) at k = n - 0.5; a raw UMI count shows n.
            saturated = values > n - 0.5 - 1e-6
            values[saturated] = n
            off = np.abs(values - np.round(values)).max()
            assert off < 1e-2, f"{batch}: UMI inversion is off by up to {off}"
            how = (f"inverted UMI saturation correction (N={n}), max distance to integer {off:.1e}; "
                   f"{saturated.sum()} saturated values set to {n}")
        else:
            non_integer = np.mean(np.abs(values - np.round(values)) > 1e-6)
            how = "unchanged" if non_integer == 0 else f"rounded ({non_integer:.1%} of nonzero values were non-integer)"
        rounded = np.round(values)
        report[str(batch)] = {"method": how, "nonzero_values_set_to_zero": float(np.mean(rounded == 0))}
        counts.data[start:end] = rounded
    counts.eliminate_zeros()
    assert (counts.data >= 1).all() and (counts.data == np.round(counts.data)).all()
    return counts.astype(np.float32), report


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("A GPU is required.")
    name = "scvi" + "".join(f"_no_{b}" for b in sorted(args.exclude_batches))
    out_dir = args.outdir / "embeddings" / args.data_dir.name / name
    out_dir.mkdir(parents=True, exist_ok=True)

    hvg = logreg.HVGFeatures(args.data_dir, args.checkpoint_dir, args.n_hvg)
    obs = hvg.obs
    batches = obs["tech"].astype(str).to_numpy()
    unknown = set(args.exclude_batches) - set(batches)
    assert not unknown, f"unknown batches {unknown}"
    order = np.argsort(batches, kind="stable")  # contiguous rows per batch for fix_counts
    counts, report = fix_counts(sp.load_npz(args.data_dir / "counts.npz")[order], batches[order])
    counts = counts[np.argsort(order)]
    for batch, r in report.items():
        print(f"  {batch}: {r['method']}; {r['nonzero_values_set_to_zero']:.1%} of nonzeros became 0")

    for seed in args.seeds:
        path = out_dir / f"seed_{seed}.npz"
        if path.exists() and not args.overwrite:
            print(f"Skipping {path}: already done")
            continue
        split = pd.read_csv(args.data_dir / "splits" / f"seed_{seed}.csv")
        assert (split["cell_id"].to_numpy() == obs.index.to_numpy()).all(), "split and obs rows differ"
        assert not set(args.exclude_batches) & set(split.loc[split["split"] == "test", "tech"]), \
            "the test batch cannot be excluded"
        genes = hvg.in_vocab[hvg.hvgs(split["split"].eq("train").to_numpy())]
        keep = ~np.isin(batches, args.exclude_batches)

        adata = ad.AnnData(counts[np.flatnonzero(keep)][:, genes].tocsr(),
                           obs=obs.loc[keep, ["tech"]].astype(str).astype("category"))
        adata.var_names = pd.read_csv(args.data_dir / "var.csv")["gene"].astype(str).to_numpy()[genes]
        empty = int((np.asarray(adata.X.sum(1)).ravel() == 0).sum())
        assert empty == 0, f"{empty} cells have no counts in the HVGs"

        scvi.settings.seed = seed
        scvi.model.SCVI.setup_anndata(adata, batch_key="tech")
        model = scvi.model.SCVI(adata, n_latent=args.n_latent, n_layers=args.n_layers, n_hidden=args.n_hidden,
                                gene_likelihood=args.gene_likelihood, dispersion=args.dispersion)
        started = time.time()
        model.train(max_epochs=args.max_epochs, use_gpu=True, train_size=0.9, early_stopping=True,
                    enable_progress_bar=False)
        train_seconds = time.time() - started
        latent = model.get_latent_representation().astype(np.float32)
        assert np.isfinite(latent).all(), "non-finite latent values"

        history = pd.concat([model.history[k] for k in ("elbo_train", "elbo_validation",
                                                        "reconstruction_loss_validation")], axis=1)
        history.to_csv(out_dir / f"seed_{seed}_history.csv")
        np.savez_compressed(path, cell_ids=obs.index.to_numpy()[keep].astype("U"), embeddings=latent)
        info = {
            "seed": seed, "n_cells": int(keep.sum()), "n_genes": int(len(genes)),
            "excluded_batches": sorted(args.exclude_batches), "count_fixes": report,
            "epochs_run": int(len(history)), "max_epochs": model.trainer.max_epochs if hasattr(model, "trainer") else None,
            "final_elbo_validation": float(history["elbo_validation"].dropna().iloc[-1]),
            "train_seconds": train_seconds,
            "settings": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "versions": {p: version(p) for p in ("scvi-tools", "torch", "anndata")},
        }
        path.with_suffix(".json").write_text(json.dumps(info, indent=2) + "\n")
        print(f"seed {seed}: {info['epochs_run']} epochs in {train_seconds / 60:.1f} min, "
              f"validation ELBO {info['final_elbo_validation']:.1f}, saved {path}", flush=True)


if __name__ == "__main__":
    main()
