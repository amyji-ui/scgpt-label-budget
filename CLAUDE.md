# How Many Labels Does scGPT Need?

CMU course project by Amy Ji and Cindy Mai. We fine tune scGPT for cell type annotation at several label budgets and compare it with simpler methods under one evaluation pipeline.

**Deliverable:** a label budget curve of macro F1 against labeled cells, with one line per method. Following instructor feedback, we report the budget ranges where methods are statistically indistinguishable, not one universal crossover. "scGPT never overtakes" is a valid result.

Sources: the proposal plus instructor feedback. Where they conflict, the feedback wins. Items marked **[F]** come from the feedback.

## Status (2026-11-05)

`util/00_preprocessing.py` exports the data and builds the splits (`data/processed/`). `util/01_finetuning.py` runs scGPT fine tuning and random init and saves test predictions without scoring them, under `results/test_<batch>/...`, with the batch name read from the split file. For a second test batch, build its splits in a separate folder (`00_preprocessing.py --test-batches inDrop1 --outdir data/processed_inDrop1`) and pass that folder to `01_finetuning.py` with `--data-dir`. `util/02_evaluate.py` is the evaluation script, **frozen 2026-11-07** (sha256 `74200e16…`, in `util/02_evaluate.sha256`). It scores every finished run folder (`predictions_test.csv` with `cell_id`, `predicted`, plus `run.json` with `seed` and `budget`) and writes `metrics.json` per run and `metrics_summary.csv` and `metrics_per_type.csv` per test batch. It refuses to score if the file has changed since freezing; changing it means refreezing and rescoring everything. Baselines must write the same run folder format. Fine-tuned scGPT, seed 0 all budgets (with `best_model.pt`) and seed 1 budgets 500–5,000, are scored. `util/03_logreg.py` runs logistic regression on HVG features or fine-tuned scGPT embeddings, with fixed C = 0.01 and alpha = 0 at every budget (no tuning, Amy, 2026-11-08), trained on all budget cells (fit + val). `util/04_scgpt_embed.py` embeds every cell with the pretrained checkpoint (same input as fine tuning, 2.7 min per seed on GPU) into `results/embeddings/processed/scgpt_frozen/`; `03_logreg.py --features scgpt_frozen` uses them. `util/05_harmony_embed.py` runs log-normalized → same 2,000 HVGs → scaling → 50 PCs → Harmony (batch = `tech`, theta 2) on all cells, one file per seed in `results/embeddings/processed/harmony/` (with uncorrected PCs); `03_logreg.py --features harmony` uses them. `util/06_scvi_embed.py` fixes the counts per decision 1 (celseq/celseq2 saturated values, capped at k = 255.5 by the source, are set to 256) and trains scVI (30 latent, 2 layers, nb, gene-batch dispersion, early stopping; about 5 GPU min per seed) on the same HVGs into `results/embeddings/processed/scvi/`; `--exclude-batches smarter` writes `scvi_no_smarter/`. `03_logreg.py --features scvi|scvi_no_smarter` uses them; the no-smarter run drops smarter's budget cells. All baselines (HVG, Harmony, scVI, frozen scGPT) and fine-tuned scGPT are scored for seeds 0 and 1 (56 runs). The directory is not a git repository.

All methods use the saved `fit`/`val` split, not CV (Amy, 2026-11-05).

scGPT settings follow the authors' annotation tutorial, not the integration example (Amy, 2026-11-05): cross entropy only, no batch labels, no HVG filter, up to 3,000 expressed genes per cell, 10 epochs, best epoch by lowest validation loss, Adam `eps=1e-4`, fp16 with a gradient scaler. Keep to these settings unless Amy decides otherwise.

| Item | State |
|---|---|
| Data, `data/pancreas.h5ad` | Present |
| Checkpoint, `scGPT_human/` | Present |
| GPU | Working, bf16 supported |
| Environment, `.venv` | Working, verified on real data (below) |

## Environment

There is one venv, managed with uv on Python 3.11. `requirements.txt` is an exact lock, and its header explains why each constraint exists. Rebuild with:

```
uv venv --python 3.11 .venv && uv pip sync requirements.txt --python .venv/bin/python
```

Key versions: torch 2.3.1 (CUDA 12.1), torchtext 0.18.0, scgpt 0.2.4, scvi-tools 0.20.3, numpy 1.26.4, anndata 0.10.9.

**Do not upgrade torch, numpy, anndata, jax, pyarrow or setuptools casually.** `uv pip check` passes even on broken combinations because `torchtext` declares no torch limit, so test imports instead.

- **Tested:** all imports; reading the data; a 30-step bf16 fine-tuning run on GPU from the checkpoint; scVI training on GPU; Harmony; HVG selection; logistic regression.
- **Attention path:** `flash-attn` is not installed, so scGPT uses PyTorch attention. The `Wqkv` weights are renamed and loaded correctly: all 48 attention parameters match the checkpoint.
- **Checkpoint weights not loaded:** `flag_encoder` and `mvc_decoder` (4 keys), which were off in the test config. `cls_decoder` starts random, as expected.
- **`load_pretrained` drops mismatched keys silently,** so assert the loaded key count in real training code.
- **scvi-tools API:** 0.20.3 is old and takes `use_gpu=True`, not `accelerator=`. Cite this version in the report.
- **JAX:** runs on CPU only, which doesn't matter because the methods here run scVI on PyTorch.

## Study design

### Methods

| Method | Role | Input | Budget affects |
|---|---|---|---|
| scGPT, fine tuned | Under test | Binned expression | Fine tuning and classifier |
| scGPT, frozen **[F]** | Shows what fine tuning adds | Binned expression | Logistic regression only |
| Same architecture, random init **[F]** | Shows what pretraining adds; selected budgets only | Binned expression | All training |
| scVI | Deep baseline | Raw counts **[F]** | Logistic regression only |
| Harmony on PCs | Classical baseline | Log-normalized | Logistic regression only |
| HVG + logistic regression | Simple floor | Log-normalized | Logistic regression only |

### Rules

- **Budgets:** about 500, 1,000, 5,000 and all training cells. Validation labels count toward the budget **[F]**, including any used to tune logistic regression.
- **Repeats:** repeat label-subset sampling **[F]**, and held-out batches if feasible **[F]**. The proposal's two seeds are not enough for this. Use the same subsets across methods so comparisons can be paired.
- **Test split:** whole held-out batches only (Amy, 2026-11-05). No batch may appear in both training and test. `util/00_preprocessing.py` enforces this.
- **Rare types:** every budget has at least 5 cells of each type, where the training pool has that many (Amy, 2026-11-05), sampled using training labels only **[F]**. This lifts rare types above their natural share, so report that alongside the budget fractions. It supports stratified CV of up to 5 folds.
- **Test cells:** every method gets the same access to unlabeled test cells **[F]**. Test labels are never used for fitting or tuning.
- **Comparability:** the same cells and genes for all methods, with each method's appropriate input **[F]**.
- **Transfer claims:** check whether the pancreas studies were in scGPT's pretraining data before attributing gains to transfer **[F]**.
- **Reporting:** macro F1 is primary. Also report accuracy, per cell type results at the smallest and largest budgets, and wall clock time and cost per run.
- **Freeze:** the preprocessed data object and evaluation script are frozen before any real scGPT run. Every method loads the same object and uses the same scoring function. Changing either later means rescoring everything.

### Out of scope

Batch integration and perturbation prediction. The stretch goal is a second dataset, not a second model.

## Data facts (measured from `data/pancreas.h5ad`)

The file has 16,382 cells × 19,093 genes, stored as dense float32, with no NaNs or negative values and no duplicate names. `obs` holds `tech`, `celltype` and `size_factors`. `X` appears to be log-normalized (max about 13). There are 14 cell types.

**Raw counts:** the `counts` layer is not raw counts in every batch, which affects scVI.

| Batch (`tech`) | Cells | Non-integer share of nonzero values | Smallest nonzero | Types present | Types with <5 cells |
|---|---|---|---|---|---|
| inDrop3 | 3,605 | 0% | 1 | 14 | 3 |
| smartseq2 | 2,394 | 0% | 1 | 13 | 1 |
| celseq2 | 2,285 | 100% | 1.002 | 13 | 2 |
| inDrop1 | 1,937 | 0% | 1 | 14 | 1 |
| inDrop2 | 1,724 | 0% | 1 | 14 | 3 |
| smarter | 1,492 | 99.99% | near 0 | 4 | 0 |
| inDrop4 | 1,303 | 0% | 1 | 14 | 5 |
| celseq | 1,004 | 100% | 1.002 | 13 | 5 |
| fluidigmc1 | 638 | 57% | 0.01 | 13 | 4 |

So 5,419 cells (33%) are non-integer. Values below 1 in smarter and fluidigmc1 suggest they are not counts at all, so check them against the source studies.

What the non-integer values are (measured 2026-11-08):
- **celseq, celseq2:** UMI-saturation-corrected counts, x = −256·ln(1 − k/256). Inverting gives an exact integer UMI count k for 100% of nonzero values, so true counts can be recovered exactly.
- **fluidigmc1:** 96.6% of values are multiples of 0.01 and only 2.4% are below 1, consistent with estimated read counts (for example RSEM expected counts). Rounding is reasonable.
- **smarter:** continuous values with 21% of nonzeros below 1 and per-cell sums around 435,000, consistent with length-normalized values such as RPKM, not counts. Rounding turns 14% of nonzeros into 0. Confirm against Xin et al. 2016.
- **smartseq2:** integer read counts (not UMIs), per-cell sums around 290,000.

**Cell types:** alpha 5,493; beta 4,169; ductal 2,142; acinar 1,669; delta 1,055; gamma 699; activated_stellate 464; endothelial 313; quiescent_stellate 193; macrophage 79; mast 42; epsilon 32; schwann 25; t_cell 7.

**Split consequences:**
- `t_cell` exists only in inDrop1-4 (7 cells in total).
- `smarter` has only alpha, beta, delta and gamma, so it is a poor test batch for macro F1.
- Many batch and type pairs have only 1-5 cells, so per-type F1 and macro F1 can swing on a single cell.

**Vocabulary:** 18,160 of 19,093 genes (95.1%) are in the scGPT vocabulary; 933 are not.

**Sequence length:** in a sample of 2,000 cells, the median cell has 2,428 expressed vocabulary genes and 92% exceed the checkpoint's `max_seq_len` of 1200. So the choice of which genes reach scGPT matters, for example HVG filtering versus random sampling per cell.

## Checkpoint facts (`scGPT_human/args.json`)

12 layers, 8 heads, embedding size 512, `max_seq_len` 1200, 51 bins, continuous input embedding, vocabulary of 60,697. Training source is CELLxGENE Census, with a run directory dated May 2023. The checkpoint is about 205 MB.

## Open decisions

These must be settled before the week 2 freeze. The recommendations are proposals, not agreed choices.

1. **Non-integer counts for scVI.** Decided (Amy, 2026-11-08), option A+: celseq and celseq2 are inverted exactly to integer UMI counts; fluidigmc1 and smarter are rounded; all cells stay in; scVI uses per-batch dispersion (`dispersion="gene-batch"`) so smarter's inflated scale doesn't make scVI over-trust it; one sensitivity run trains scVI without smarter. No conversion of RPKM back to counts (needs gene lengths, not worth the time).
2. **Held-out batches.** Recommend smartseq2 as primary (different platform, and all 7 `t_cell` cells stay in training), then celseq2 and inDrop1 if budget allows. Holding out one inDrop batch is an easier test, since its siblings stay in training. Caution on celseq2: CELLxGENE has listed Muraro et al. (CEL-seq2, 2,126 primary cells) since 2021, before the May 2023 checkpoint, so that batch may have been in pretraining. Confirm against the Census release used, and avoid it as a test batch until then.
3. **Test-cell access.** Decided (Amy, 2026-11-08): unlabeled test cells are allowed for all methods, because Harmony corrects cells jointly and real annotation has those cells available. Document which methods actually use them: Harmony uses them for scaling, PCA and correction; HVG selection uses training cells only; scGPT and HVG + logistic regression see test cells only at prediction time.
4. **Final classifier for fine-tuned scGPT.** The proposal says logistic regression for all methods, but fine tuning trains a classifier head. Recommend the head as primary for fine-tuned and random-init runs, with logistic regression on embeddings as a secondary check.
5. **Macro F1 class set.** Decided (Amy, 2026-11-07): average over every cell type present in the test set, however few cells it has. That is 13 types for `smartseq2`, which has no `t_cell`. A wrong prediction of a type absent from the test set still costs points, through the recall of the true type.
6. **Gene set.** Recommend restricting all methods to the 18,160 shared genes, and selecting HVGs without test labels.
7. **Repeat counts and random-init budgets.** Recommend at least 5 label subsets, with random init at the smallest and largest budgets. Set these after timing one real run.
8. **Statistical test for "indistinguishable."** Recommend paired differences across shared label subsets with bootstrap confidence intervals.
9. **Member A and Member B.** The proposal does not assign them to names.
10. **Genes passed to scGPT.** Decided: the authors' limit of 3,000 randomly sampled expressed genes, no HVG filter. 43% of all cells and 93% of `smartseq2` exceed it, and it drops 28% of measured genes overall.

## Team and timeline

**Member A:** scGPT environment, gene vocabulary mapping, all fine tuning, methods and results.
**Member B:** preprocessing, split, baselines, evaluation script, plots, background and discussion.
Each reviews the other's code. Member B freezes data and evaluation before Member A's real sweep.

| Week | Member A | Member B | Gate |
|---|---|---|---|
| 1-2 | Install scGPT, one run end to end | Data, baselines, evaluation frozen | **Hard gate** |
| 3 | Two smallest budgets | Baseline results, all budgets | |
| 4 | Two largest budgets | First draft of curve | Midterm report |
| 5-6 | Reruns, timings, methods and results | Variance, background, discussion | |
| 7 | Second dataset if time allows | Final plots, proofreading | Final report |

**Fallback:** if scGPT is not working by the end of week 2, drop it and run the sweep with scVI.

## Compute

- **Instance:** g5.xlarge with A10G (23 GB reported), 4 vCPUs and 15 GB RAM. The dense matrices are about 1.25 GB each, so avoid unnecessary copies.
- **Disk:** 80 GB EBS volume (grown from 60 GB on 2026-11-05), with a 77 GB root filesystem and about 47 GB free. The venv is 6.3 GB and each checkpoint about 205 MB. EBS volumes can't be shrunk.
- **Temporary disk:** 233 GB local NVMe at `/opt/dlami/nvme`, included in the instance price. **Its data is lost when the instance stops or terminates,** so use it only for files you can regenerate, and keep code, frozen data, predictions, metrics and final checkpoints on the root disk or S3. Create directories at the start of each job and check the mount exists, because it may need setting up again after a stop.
- **Budget:** 40 GPU hours, about $40, plus about 25 hours if the second dataset is reached.
- **Measured speed:** 0.485 s per step at batch size 32 and full 1200-gene length, with bf16 and all layers trainable. That is about 66 cells/s, or about 4.1 minutes per epoch on 16,382 cells, using 8.2 GiB of GPU memory. It excludes data loading and evaluation, and it is a 30-step test on pre-encoded batches, not a full run.
- **Run count:** the proposal assumed 8 runs. As an example, 5 subsets × 4 budgets plus random init at 2 budgets is 30 runs per held-out batch. Each epoch costs about 4 minutes per 16,000 training cells, so epoch count decides the total. Time one real run with validation before committing.
- **Bridges2:** its migration was scheduled for early November 2026, so don't count on it as a backup.

Run Python through `.venv/bin/python`. Stop the instance when idle, since it bills by the hour.

**Scratchpad quota:** Claude's scratchpad under `/tmp` has a small disk quota, too small for an environment build. Use somewhere under `$HOME` for large temporary files.

## Proposal fixes for the reports

- **Related work [F]:** narrow "all of that work evaluates the models frozen." CellBench-LS includes task-head adaptation and SCMBench evaluates fine tuning. Emphasize our label-budget and held-out-batch setting.
- **Citations [F]:** replace the bracketed keys (`baron2016`, `grun2016`, and so on) with real references. Also verify the study list against Luecken et al., since `fluidigmc1` may map to a study not listed there.
- **Compute:** the abstract says V100 and 20 GB; Section 6 says A10G and 100 GB; the instance has 60 GB.
- **Midterm:** Section 5 says the two smaller budgets by midterm, but Table 2 has all four by week 4.
