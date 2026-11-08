#!/usr/bin/env python
"""Logistic regression on fixed features, with the same fixed settings at every budget.

Run from the project root:
    .venv/bin/python util/03_logreg.py --features hvg
    .venv/bin/python util/03_logreg.py --features scgpt_finetuned
    .venv/bin/python util/03_logreg.py --features scgpt_frozen      # after util/04_scgpt_embed.py
    .venv/bin/python util/03_logreg.py --features harmony           # after util/05_harmony_embed.py
    .venv/bin/python util/03_logreg.py --features scvi              # after util/06_scvi_embed.py

Feature sources:
    hvg               Log-normalized expression of highly variable genes (the "simple floor").
                      Genes are limited to the scGPT vocabulary, and HVGs are chosen per batch
                      (scanpy, flavor "seurat", batch_key tech) on training-pool cells only.
                      No labels and no test cells are used to choose them.
    scgpt_finetuned   The cell embeddings saved by 01_finetuning.py for the same budget and seed.
                      This is the secondary check for fine-tuned scGPT (open decision 4).
    scgpt_frozen      Pretrained scGPT embeddings with no training, from 04_scgpt_embed.py, one
                      file per seed. Logistic regression is the only part that sees the budget.
    harmony           Harmony-corrected PCs of the same HVGs, from 05_harmony_embed.py, one file
                      per seed. Uses unlabeled test cells (open decision 3).
    scvi              scVI latent means from raw counts of the same HVGs, from 06_scvi_embed.py.
    scvi_no_smarter   Sensitivity run: scVI trained without smarter. smarter's budget cells have no
                      embedding, so they are dropped and that run uses fewer labels than its budget.

Settings (no tuning, decided by Amy, 2026-11-08):
    C = 0.01   Inverse regularization strength, the same at every budget. Chosen from the middle
               of the plateau in a pilot sweep (2026-11-08), not tuned per budget, so no budget
               borrows information from another budget's labels.
    alpha = 0  Prior correction. The predicted class is argmax_c [log p(c|x) - alpha * log prior(c)],
               where prior is the class share in the training cells. alpha = 0 is plain argmax;
               alpha = 1 removes the class prior, which favours rare types (higher recall, lower
               precision). Kept as an option, off by default.
    With nothing to tune, the model trains on all budget cells (fit + val).

Outputs, in the same run folder format as 01_finetuning.py so 02_evaluate.py scores them:
    results/test_<batch>/<features>_logreg/budget_<B>/seed_<S>/
        predictions_test.csv   cell_id, predicted, prob_<type> (unadjusted probabilities).
        run.json               Written last.
"""

import argparse
import importlib.util
import json
import time
import warnings
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

_spec = importlib.util.spec_from_file_location("evaluate", Path(__file__).with_name("02_evaluate.py"))
evaluate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(evaluate)



def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--features", choices=["hvg", "scgpt_finetuned", "scgpt_frozen", "harmony", "scvi", "scvi_no_smarter"], required=True)
    p.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    p.add_argument("--checkpoint-dir", type=Path, default=Path("scGPT_human"), help="For the vocabulary.")
    p.add_argument("--outdir", type=Path, default=Path("results"))
    p.add_argument("--budgets", nargs="+", default=["500", "1000", "5000", "all"])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    p.add_argument("--n-hvg", type=int, default=2000)
    p.add_argument("--C", type=float, default=0.01)
    p.add_argument("--alpha", type=float, default=0.0)
    p.add_argument("--max-iter", type=int, default=3000)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


class HVGFeatures:
    """Log-normalized HVG expression, with HVGs chosen once per training pool."""

    def __init__(self, data_dir, checkpoint_dir, n_hvg):
        vocab = set(json.loads((checkpoint_dir / "vocab.json").read_text()))
        genes = pd.read_csv(data_dir / "var.csv")["gene"].astype(str).to_numpy()
        self.in_vocab = np.flatnonzero([g in vocab for g in genes])
        self.genes = genes[self.in_vocab]
        self.matrix = sp.load_npz(data_dir / "X_lognorm.npz").tocsr()[:, self.in_vocab].tocsr()
        self.obs = pd.read_csv(data_dir / "obs.csv", index_col="cell_id")
        self.n_hvg = n_hvg
        self.cache = {}

    def hvgs(self, train_mask):
        key = train_mask.tobytes()
        if key not in self.cache:
            pool = ad.AnnData(self.matrix[train_mask], obs=self.obs.iloc[np.flatnonzero(train_mask)][["tech"]].copy())
            pool.var_names = self.genes
            sc.pp.highly_variable_genes(pool, n_top_genes=self.n_hvg, flavor="seurat", batch_key="tech")
            self.cache[key] = np.flatnonzero(pool.var["highly_variable"].to_numpy())
        return self.cache[key]

    def get(self, split, budget, seed, outdir):
        train_mask = split["split"].eq("train").to_numpy()
        cols = self.hvgs(train_mask)
        x = self.matrix[:, cols].toarray().astype(np.float32)
        return x, split["cell_id"].to_numpy(), {"n_genes_in_vocab": int(len(self.genes)), "n_hvg": int(len(cols))}


class FinetunedEmbeddings:
    """Cell embeddings saved by 01_finetuning.py for the same budget and seed."""

    def get(self, split, budget, seed, outdir):
        batch = "+".join(sorted(split.loc[split["split"] == "test", "tech"].astype(str).unique()))
        path = outdir / f"test_{batch}" / "scgpt_finetuned" / f"budget_{budget}" / f"seed_{seed}" / "embeddings.npz"
        data = np.load(path)
        ids, roles = data["cell_ids"].astype(str), data["roles"].astype(str)
        expected = split.set_index("cell_id").loc[ids]
        expected = np.where(expected["split"].eq("test"), "test", expected[f"budget_{budget}"])
        assert (roles == expected).all(), f"{path}: roles differ from the split file"
        return data["embeddings"].astype(np.float32), ids, {"embeddings": str(path)}


class SavedEmbeddings:
    """Label-free embeddings saved once per seed: scgpt_frozen (04), harmony (05), scvi (06).

    Every cell must be present, except for runs whose .json lists excluded_batches.
    """

    def __init__(self, data_dir, name):
        self.data_dir, self.name = data_dir, name

    def get(self, split, budget, seed, outdir):
        path = outdir / "embeddings" / self.data_dir.name / self.name / f"seed_{seed}.npz"
        data = np.load(path)
        ids = data["cell_ids"].astype(str)
        meta = path.with_suffix(".json")
        excluded = json.loads(meta.read_text()).get("excluded_batches", []) if meta.exists() else []
        expected = split.loc[~split["tech"].isin(excluded), "cell_id"].astype(str).to_numpy()
        assert (ids == expected).all(), f"{path}: cells differ from the split"
        return data["embeddings"].astype(np.float32), ids, {"embeddings": str(path), "excluded_batches": excluded}


def adjusted_predict(log_proba, log_prior, alpha):
    return np.argmax(log_proba - alpha * log_prior, axis=1)


def fit_lr(x, y, C, max_iter):
    model = LogisticRegression(C=C, max_iter=max_iter, solver="lbfgs")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(x, y)
    converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
    return model, converged


def log_prior(y, n_classes):
    counts = np.bincount(y, minlength=n_classes).astype(float)
    return np.log(counts / counts.sum())


def run_one(features, split, labels, budget, seed, args, run_dir):
    role = split[f"budget_{budget}"].to_numpy()
    is_test = split["split"].eq("test").to_numpy()
    x_all, ids, info = features.get(split, budget, seed, args.outdir)
    index = pd.Series(np.arange(len(ids)), index=ids)
    pick = lambda mask: index.loc[split.loc[mask, "cell_id"]].to_numpy()
    has = split["cell_id"].isin(ids).to_numpy()
    assert has[is_test].all(), "some test cells have no features"
    in_train = (role != "unused") & has
    n_missing = int(((role != "unused") & ~has).sum())
    train_rows, test_rows = pick(in_train), pick(is_test)

    # Labels for training cells only. Test labels are never read here.
    classes = np.array(sorted(set(labels[in_train])))
    code = {c: i for i, c in enumerate(classes)}
    y_train = np.array([code[c] for c in labels[in_train]])
    print(f"{args.features}, budget {budget}, seed {seed}: {len(train_rows)} training cells (fit + val), "
          f"{len(test_rows)} test, {x_all.shape[1]} features"
          + (f", {n_missing} budget cells dropped for lack of features" if n_missing else ""))

    t0 = time.time()
    x_train = x_all[train_rows]
    scaler = StandardScaler().fit(x_train)
    model, converged = fit_lr(scaler.transform(x_train), y_train, args.C, args.max_iter)
    train_seconds = time.time() - t0

    t0 = time.time()
    lp = model.predict_log_proba(scaler.transform(x_all[test_rows]))
    pred = classes[adjusted_predict(lp, log_prior(y_train, len(classes)), args.alpha)]
    infer_seconds = time.time() - t0

    run_dir.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame({"cell_id": ids[test_rows], "predicted": pred})
    out = pd.concat([out, pd.DataFrame(np.exp(lp), columns=[f"prob_{c}" for c in classes])], axis=1)
    out.to_csv(run_dir / "predictions_test.csv", index=False)
    run = {
        "method": f"{args.features}_logreg", "budget": str(budget), "seed": seed,
        "data_dir": str(args.data_dir),
        "test_batches": sorted(split.loc[is_test, "tech"].astype(str).unique()),
        "n_fit": int(((role == "fit") & has).sum()), "n_val": int(((role == "val") & has).sum()),
        "n_test": int(len(test_rows)), "n_budget_cells_dropped": n_missing,
        "classes": classes.tolist(), "n_features": int(x_all.shape[1]), "features": info,
        "C": args.C, "alpha": args.alpha, "trained_on": "fit + val", "converged": converged,
        "train_seconds": train_seconds, "test_inference_seconds": infer_seconds,
        "settings": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "versions": {"sklearn": sklearn.__version__, "scanpy": sc.__version__},
    }
    (run_dir / "run.json").write_text(json.dumps(run, indent=2) + "\n")
    print(f"  C {args.C}, alpha {args.alpha}, converged {converged}, {train_seconds:.0f}s")


def main():
    args = parse_args()
    obs = pd.read_csv(args.data_dir / "obs.csv", index_col="cell_id")
    features = {"hvg": lambda: HVGFeatures(args.data_dir, args.checkpoint_dir, args.n_hvg),
                "scgpt_finetuned": FinetunedEmbeddings,
                "scgpt_frozen": lambda: SavedEmbeddings(args.data_dir, "scgpt_frozen"),
                "harmony": lambda: SavedEmbeddings(args.data_dir, "harmony"),
                "scvi": lambda: SavedEmbeddings(args.data_dir, "scvi"),
                "scvi_no_smarter": lambda: SavedEmbeddings(args.data_dir, "scvi_no_smarter")}[args.features]()
    for seed in args.seeds:
        split = pd.read_csv(args.data_dir / "splits" / f"seed_{seed}.csv")
        assert (split["cell_id"].to_numpy() == obs.index.to_numpy()).all(), "split and obs rows differ"
        labels = obs["celltype"].astype(str).to_numpy()
        batch = "+".join(sorted(split.loc[split["split"] == "test", "tech"].astype(str).unique()))
        for budget in args.budgets:
            run_dir = args.outdir / f"test_{batch}" / f"{args.features}_logreg" / f"budget_{budget}" / f"seed_{seed}"
            if (run_dir / "run.json").exists() and not args.overwrite:
                print(f"Skipping {run_dir}: already done")
                continue
            run_one(features, split, labels, budget, seed, args, run_dir)


if __name__ == "__main__":
    main()
