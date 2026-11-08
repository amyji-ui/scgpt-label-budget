#!/usr/bin/env python
"""Embed every cell with the pretrained scGPT checkpoint, without any training (frozen scGPT).

Run from the project root:
    .venv/bin/python util/04_scgpt_embed.py
Then fit logistic regression on the embeddings:
    .venv/bin/python util/03_logreg.py --features scgpt_frozen

The input is exactly what 01_finetuning.py feeds the model for the same seed: the same binning and
the same random sample of up to 3,000 expressed genes per cell. So the only difference from
fine-tuned scGPT is the training. The embedding is the <cls> token output, L2-normalized per cell
as in the authors' scgpt.tasks.embed_data.

No labels are read. Every cell is embedded, test cells included, as fine-tuned scGPT also does.
The embeddings depend only on the data and the seed, not on the budget, so one file per seed
serves every budget:
    results/embeddings/<data dir name>/scgpt_frozen/seed_<S>.npz   cell_ids, embeddings
    results/embeddings/<data dir name>/scgpt_frozen/seed_<S>.json  settings and timing
"""

import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scgpt.tokenizer.gene_tokenizer import GeneVocab

_spec = importlib.util.spec_from_file_location("finetuning", Path(__file__).with_name("01_finetuning.py"))
ft = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ft)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    p.add_argument("--checkpoint-dir", type=Path, default=Path("scGPT_human"))
    p.add_argument("--outdir", type=Path, default=Path("results"))
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-seq-len", type=int, default=3001, help="Same as 01_finetuning.py.")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("A GPU is required.")
    config = json.loads((args.checkpoint_dir / "args.json").read_text())
    vocab = GeneVocab.from_file(args.checkpoint_dir / "vocab.json")
    matrix, gene_ids, obs = ft.load_expression(args.data_dir, vocab)
    setup = {
        "device": torch.device("cuda"),
        "pad_value": config["pad_value"],
        "tokens": {"pad": vocab[ft.PAD_TOKEN], "cls": vocab[ft.CLS_TOKEN]},
    }
    out_dir = args.outdir / "embeddings" / args.data_dir.name / "scgpt_frozen"
    out_dir.mkdir(parents=True, exist_ok=True)

    # The classifier head is unused; its size does not affect the embedding.
    model, n_loaded = ft.build_model(config, vocab, 1, "pretrained", args.checkpoint_dir)
    model = model.to(setup["device"]).eval()
    rows = np.arange(matrix.shape[0])

    for seed in args.seeds:
        path = out_dir / f"seed_{seed}.npz"
        if path.exists() and not args.overwrite:
            print(f"Skipping {path}: already done")
            continue
        np.random.seed(seed)  # Same binning tie-breaks and gene sample as 01_finetuning.py.
        binned = ft.bin_cells(matrix, config["n_bins"])
        cells_matrix = ft.sample_genes(binned, args.max_seq_len - 1, np.random.default_rng([seed, 2]))

        torch.cuda.reset_peak_memory_stats()
        started = time.time()
        _, embeddings = ft.predict(model, cells_matrix, gene_ids, rows, args, setup)
        seconds = time.time() - started
        embeddings = embeddings.numpy().astype(np.float32)
        embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
        assert np.isfinite(embeddings).all(), "non-finite embeddings"

        np.savez_compressed(path, cell_ids=obs["cell_id"].to_numpy().astype("U"), embeddings=embeddings)
        info = {
            "seed": seed, "n_cells": int(len(rows)), "dim": int(embeddings.shape[1]),
            "pretrained_weights_loaded": n_loaded, "embed_seconds": seconds,
            "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2**30,
            "settings": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        }
        path.with_suffix(".json").write_text(json.dumps(info, indent=2) + "\n")
        print(f"seed {seed}: embedded {len(rows)} cells in {seconds / 60:.1f} min, saved {path}", flush=True)


if __name__ == "__main__":
    main()
