#!/usr/bin/env python
"""Fine tune scGPT for cell type annotation at each label budget and seed.

Run from the project root, after util/00_preprocessing.py:
    .venv/bin/python util/01_finetuning.py                                  # 8 fine tuning runs
    .venv/bin/python util/01_finetuning.py --init random --budgets 500 all  # random init baseline

Settings follow the authors' annotation tutorial (scGPT tutorials/Tutorial_Annotation.ipynb): cross
entropy only, no batch labels, no HVG filter, up to 3,000 expressed genes per cell, 10 epochs, the
lowest validation loss picks the best epoch, Adam with eps=1e-4, and fp16 with a gradient scaler.
The known differences are PyTorch attention instead of flash-attn, which is not installed, and
reshuffling the fit cells every epoch.

Each budget trains on its `fit` cells, and its `val` cells choose the best epoch. The test batch is
only used for predictions at the end: this script does not read test labels or score the test set,
so that every method is scored by the same evaluation script.

To use another test batch, build its splits in their own folder and point --data-dir at it:
    .venv/bin/python util/00_preprocessing.py --test-batches inDrop1 --outdir data/processed_inDrop1
    .venv/bin/python util/01_finetuning.py --data-dir data/processed_inDrop1

Outputs, under results/test_<batch>/scgpt_<finetuned|random>/budget_<B>/seed_<S>/, where <batch>
is read from the split file's test cells, so runs on different test batches never overwrite
each other:
    predictions_test.csv    cell_id, predicted cell type and one probability per type.
    embeddings.npz          cell_ids, roles (fit, val or test) and cell embeddings from the best epoch.
    history.csv             Loss, validation macro F1 and time for each epoch.
    run.json                Settings and measurements. A run is complete once this file exists,
                            and complete runs are skipped unless --overwrite is set.
    best_model.pt           Only with --save-checkpoint (about 205 MB per run).
"""

import argparse
import copy
import json
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from scgpt.model import TransformerModel
from scgpt.preprocess import binning
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt.utils import load_pretrained
from sklearn.metrics import f1_score

PAD_TOKEN = "<pad>"
CLS_TOKEN = "<cls>"
HEAD_PREFIX = "cls_decoder."


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    p.add_argument("--checkpoint-dir", type=Path, default=Path("scGPT_human"))
    p.add_argument("--outdir", type=Path, default=Path("results"))
    p.add_argument("--init", choices=["pretrained", "random"], default="pretrained")
    p.add_argument("--budgets", nargs="+", default=["500", "1000", "5000", "all"])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lr-decay", type=float, default=0.9, help="Learning rate multiplier per epoch.")
    p.add_argument("--adam-eps", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--max-seq-len", type=int, default=3001, help="Sequence length including the <cls> token.")
    p.add_argument("--save-checkpoint", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def load_expression(data_dir, vocab):
    """Return log-normalized expression limited to vocabulary genes, their vocabulary ids and metadata."""
    matrix = sp.load_npz(data_dir / "X_lognorm.npz").tocsr()
    genes = pd.read_csv(data_dir / "var.csv")["gene"].astype(str).to_numpy()
    obs = pd.read_csv(data_dir / "obs.csv")
    in_vocab = np.array([g in vocab for g in genes])
    gene_ids = np.array(vocab(list(genes[in_vocab])), dtype=np.int64)
    print(f"Using {in_vocab.sum()} of {len(genes)} genes found in the scGPT vocabulary")
    return matrix[:, in_vocab].tocsr(), gene_ids, obs


def bin_cells(matrix, n_bins):
    """Bin each cell's expressed genes with scGPT's own method, once per seed."""
    binned = matrix.copy()
    binned.data = binned.data.astype(np.float32)
    for i in range(binned.shape[0]):
        start, end = binned.indptr[i], binned.indptr[i + 1]
        if end > start:
            binned.data[start:end] = binning(binned.data[start:end], n_bins)
    return binned


def sample_genes(binned, max_genes, rng):
    """Keep up to max_genes random expressed genes per cell. As in the tutorial, this happens once."""
    indices, data, indptr = [], [], [0]
    for i in range(binned.shape[0]):
        start, end = binned.indptr[i], binned.indptr[i + 1]
        keep = np.arange(start, end)
        if len(keep) > max_genes:
            keep = np.sort(rng.choice(keep, max_genes, replace=False))
        indices.append(binned.indices[keep])
        data.append(binned.data[keep])
        indptr.append(indptr[-1] + len(keep))
    return sp.csr_matrix((np.concatenate(data), np.concatenate(indices), indptr), shape=binned.shape)


def encode_batch(cells_matrix, gene_ids, rows, tokens, pad_value, device):
    """Build one batch of a <cls> token plus each cell's sampled genes, padded to the longest cell."""
    cells = []
    for row in rows:
        start, end = cells_matrix.indptr[row], cells_matrix.indptr[row + 1]
        cells.append((gene_ids[cells_matrix.indices[start:end]], cells_matrix.data[start:end]))

    length = max(len(ids) for ids, _ in cells) + 1
    src = np.full((len(cells), length), tokens["pad"], dtype=np.int64)
    val = np.full((len(cells), length), pad_value, dtype=np.float32)
    for i, (ids, values) in enumerate(cells):
        src[i, 0], val[i, 0] = tokens["cls"], 0
        src[i, 1 : len(ids) + 1], val[i, 1 : len(ids) + 1] = ids, values
    return torch.from_numpy(src).to(device), torch.from_numpy(val).to(device)


def build_model(config, vocab, n_classes, init, checkpoint_dir):
    """Build the model, and load pretrained weights with a check that they all arrived."""
    model = TransformerModel(
        ntoken=len(vocab),
        d_model=config["embsize"],
        nhead=config["nheads"],
        d_hid=config["d_hid"],
        nlayers=config["nlayers"],
        nlayers_cls=config["n_layers_cls"],
        n_cls=n_classes,
        vocab=vocab,
        dropout=config["dropout"],
        pad_token=PAD_TOKEN,
        pad_value=config["pad_value"],
        do_mvc=False,
        input_emb_style=config["input_emb_style"],
        n_input_bins=config["n_bins"],
        cell_emb_style="cls",
        use_fast_transformer=False,
    )
    if init == "random":
        return model, 0

    params = torch.load(checkpoint_dir / "best_model.pt", map_location="cpu", weights_only=True)
    load_pretrained(model, params, verbose=False)

    # load_pretrained silently skips mismatched weights, so check every weight except the new head.
    expected = {k.replace("Wqkv.", "in_proj_"): v for k, v in params.items()}
    state = model.state_dict()
    bad = [k for k, v in state.items() if not k.startswith(HEAD_PREFIX) and not (k in expected and torch.equal(v, expected[k]))]
    if bad:
        raise RuntimeError(f"{len(bad)} weights were not loaded from the checkpoint, such as {bad[:5]}")
    return model, sum(not k.startswith(HEAD_PREFIX) for k in state)


@torch.no_grad()
def predict(model, cells_matrix, gene_ids, rows, args, setup):
    """Return logits and cell embeddings."""
    model.eval()
    logits, embeddings = [], []
    for start in range(0, len(rows), args.batch_size):
        batch = rows[start : start + args.batch_size]
        src, val = encode_batch(cells_matrix, gene_ids, batch, setup["tokens"], setup["pad_value"], setup["device"])
        with torch.autocast("cuda", dtype=torch.float16):
            out = model(src, val, src_key_padding_mask=src.eq(setup["tokens"]["pad"]), CLS=True)
        logits.append(out["cls_output"].float().cpu())
        embeddings.append(out["cell_emb"].float().cpu())
    return torch.cat(logits), torch.cat(embeddings)


def train(model, cells_matrix, gene_ids, y, fit_rows, val_rows, args, setup, seed):
    """Train on fit cells and return the epoch history, best epoch and best weights."""
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=args.adam_eps)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=args.lr_decay)
    scaler = torch.cuda.amp.GradScaler()
    loss_fn = torch.nn.CrossEntropyLoss()
    rng = np.random.default_rng([seed, 1])
    n_classes = setup["n_classes"]

    history, best_state, best_loss, best_epoch = [], None, float("inf"), 0
    for epoch in range(1, args.epochs + 1):
        started = time.time()
        model.train()
        order = rng.permutation(fit_rows)
        total_loss = 0.0
        for start in range(0, len(order), args.batch_size):
            batch = order[start : start + args.batch_size]
            src, val = encode_batch(cells_matrix, gene_ids, batch, setup["tokens"], setup["pad_value"], setup["device"])
            target = torch.from_numpy(y[batch]).to(setup["device"])
            with torch.autocast("cuda", dtype=torch.float16):
                out = model(src, val, src_key_padding_mask=src.eq(setup["tokens"]["pad"]), CLS=True)
                loss = loss_fn(out["cls_output"], target)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item() * len(batch)
        scheduler.step()

        logits, _ = predict(model, cells_matrix, gene_ids, val_rows, args, setup)
        val_loss = loss_fn(logits, torch.from_numpy(y[val_rows])).item()
        val_f1 = f1_score(y[val_rows], logits.argmax(1).numpy(), average="macro", labels=range(n_classes), zero_division=0)
        history.append({
            "epoch": epoch, "train_loss": total_loss / len(order), "val_loss": val_loss,
            "val_macro_f1": val_f1, "seconds": time.time() - started,
        })
        print(f"  epoch {epoch}: train loss {history[-1]['train_loss']:.3f}, val loss {val_loss:.3f}, "
              f"val macro F1 {val_f1:.3f}, {history[-1]['seconds']:.0f}s", flush=True)

        if val_loss < best_loss:
            best_loss, best_epoch = val_loss, epoch
            best_state = copy.deepcopy(model.state_dict())
    return history, best_epoch, best_state


def run_one(budget, seed, split, cells_matrix, gene_ids, labels, config, vocab, args, setup, run_dir):
    """Run one fine tuning job and save its outputs."""
    column = f"budget_{budget}"
    if column not in split:
        raise ValueError(f"{column} is not in the split file")
    fit_rows = np.flatnonzero(split[column] == "fit")
    val_rows = np.flatnonzero(split[column] == "val")
    test_rows = np.flatnonzero(split["split"] == "test")

    # Labels are encoded for training cells only, so test labels never enter this script's data.
    train_mask = split["split"].eq("train").to_numpy()
    classes = sorted(set(labels[train_mask]))
    to_index = {c: i for i, c in enumerate(classes)}
    y = np.full(len(labels), -1, dtype=np.int64)
    y[train_mask] = [to_index[c] for c in labels[train_mask]]
    setup = {**setup, "n_classes": len(classes)}

    torch.manual_seed(seed)
    torch.cuda.reset_peak_memory_stats()
    model, n_loaded = build_model(config, vocab, len(classes), args.init, args.checkpoint_dir)
    model = model.to(setup["device"])

    print(f"{args.init}, budget {budget}, seed {seed}: {len(fit_rows)} fit, {len(val_rows)} val, {len(test_rows)} test", flush=True)
    started = time.time()
    history, best_epoch, best_state = train(model, cells_matrix, gene_ids, y, fit_rows, val_rows, args, setup, seed)
    train_seconds = time.time() - started
    model.load_state_dict(best_state)
    best = history[best_epoch - 1]

    started = time.time()
    logits, test_embeddings = predict(model, cells_matrix, gene_ids, test_rows, args, setup)
    inference_seconds = time.time() - started
    probabilities = torch.softmax(logits, dim=1).numpy()

    run_dir.mkdir(parents=True, exist_ok=True)
    predictions = pd.DataFrame({
        "cell_id": split["cell_id"].to_numpy()[test_rows],
        "predicted": np.array(classes)[probabilities.argmax(1)],
    })
    for i, cell_type in enumerate(classes):
        predictions[f"prob_{cell_type}"] = probabilities[:, i]
    predictions.to_csv(run_dir / "predictions_test.csv", index=False)

    budget_rows = np.concatenate([fit_rows, val_rows])
    _, budget_embeddings = predict(model, cells_matrix, gene_ids, budget_rows, args, setup)
    np.savez_compressed(
        run_dir / "embeddings.npz",
        cell_ids=split["cell_id"].to_numpy()[np.concatenate([budget_rows, test_rows])].astype(str),
        roles=np.array(["fit"] * len(fit_rows) + ["val"] * len(val_rows) + ["test"] * len(test_rows)),
        embeddings=torch.cat([budget_embeddings, test_embeddings]).numpy().astype(np.float32),
    )
    pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
    if args.save_checkpoint:
        torch.save(best_state, run_dir / "best_model.pt")

    settings = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    summary = {
        "init": args.init, "budget": budget, "seed": seed,
        "test_batches": sorted(split.loc[test_rows, "tech"].astype(str).unique()),
        "n_fit": len(fit_rows), "n_val": len(val_rows), "n_test": len(test_rows),
        "classes": classes, "best_epoch": best_epoch, "best_val_loss": best["val_loss"],
        "val_macro_f1_at_best_epoch": best["val_macro_f1"],
        "train_seconds": train_seconds, "test_inference_seconds": inference_seconds,
        "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2**30, "pretrained_weights_loaded": n_loaded,
        "settings": settings,
        "versions": {p: version(p) for p in ("torch", "scgpt", "numpy")},
    }
    (run_dir / "run.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"  best epoch {best_epoch}, val loss {best['val_loss']:.3f}, {train_seconds / 60:.1f} min. Saved to {run_dir}", flush=True)

    del model, best_state
    torch.cuda.empty_cache()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("A GPU is required.")

    config = json.loads((args.checkpoint_dir / "args.json").read_text())
    vocab = GeneVocab.from_file(args.checkpoint_dir / "vocab.json")
    matrix, gene_ids, obs = load_expression(args.data_dir, vocab)
    labels = obs["celltype"].astype(str).to_numpy()
    setup = {
        "device": torch.device("cuda"),
        "pad_value": config["pad_value"],
        "tokens": {"pad": vocab[PAD_TOKEN], "cls": vocab[CLS_TOKEN]},
    }
    name = "scgpt_finetuned" if args.init == "pretrained" else "scgpt_random"

    for seed in args.seeds:
        split = pd.read_csv(args.data_dir / "splits" / f"seed_{seed}.csv")
        if not split["cell_id"].astype(str).equals(obs["cell_id"].astype(str)):
            raise ValueError(f"seed_{seed}.csv is not in the same cell order as obs.csv")
        test_batches = sorted(split.loc[split["split"] == "test", "tech"].astype(str).unique())
        method_dir = args.outdir / f"test_{'+'.join(test_batches)}" / name

        pending = [b for b in args.budgets
                   if args.overwrite or not (method_dir / f"budget_{b}" / f"seed_{seed}" / "run.json").exists()]
        if len(pending) < len(args.budgets):
            print(f"Seed {seed}: skipping complete budgets {sorted(set(args.budgets) - set(pending))} in {method_dir}")
        if not pending:
            continue

        np.random.seed(seed)  # scGPT's binning breaks ties with numpy's global generator.
        binned = bin_cells(matrix, config["n_bins"])
        cells_matrix = sample_genes(binned, args.max_seq_len - 1, np.random.default_rng([seed, 2]))
        for budget in pending:
            run_dir = method_dir / f"budget_{budget}" / f"seed_{seed}"
            run_one(budget, seed, split, cells_matrix, gene_ids, labels, config, vocab, args, setup, run_dir)


if __name__ == "__main__":
    main()
