# handwritten-to-data

Ukrainian handwriting recognition pipeline for the Kaggle [Handwritten to Data](https://www.kaggle.com/competitions/handwritten-to-data) competition.

The composite metric is `0.15·DetF1 + 0.05·ClassAcc + 0.30·(1-RegionCER) + 0.50·(1-PageCER)` — PageCER is 50% of the score, so optimisation is end-to-end transcription, not detection.

**Compute model:** Kaggle Notebooks only. No paid GPUs, no remote vLLM endpoint, no API keys. The library imports cleanly on a Mac (no torch on the import path); GPU dependencies are lazy-loaded inside `htr.backend`.

## Layout

```
src/htr/                    # pipeline code; safe to import on CPU
  metric.py                 # official kaggle_metric.score (vendored verbatim)
  pipeline.py               # sync detect → transcribe
  prompts.py                # Qwen-VL prompts
  schemas.py                # JSON schemas + TypedDicts
  backend.py                # Unsloth FastVisionModel wrapper (lazy GPU imports)
  image_ops.py              # encode / deskew / crop helpers
  postproc.py               # NMS
  data.py                   # HF dataset loader, CV splits
  silver.py                 # silver hallucination filters

configs/                    # pipeline YAMLs (Phase 0 baseline, Phase 1 toggles)
scripts/                    # CLI entrypoints
  build_cv_split.py         # → data/cv/{train,val}.jsonl
  clean_silver.py           # → data/cv/{train_clean,silver_clean}.jsonl
  mine_fewshot.py           # → configs/_fewshot.yaml
  prepare_lora_data.py      # → data/lora/{detect,transcribe}/{train,val}.jsonl
  run_inference.py          # writes a submission CSV
  score_cv.py               # composite metric on a prediction CSV
notebooks/
  01_train_lora.ipynb       # Kaggle: train LoRA → push adapter to HF Hub
  02_inference.ipynb        # Kaggle: install → load → predict → submission.csv → auto-submit
data/cv/                    # committed CV manifests (small JSONL)
research/                   # reference baselines + EDA notebooks
```

## Workflow

Two Kaggle notebooks, each self-contained (clones this repo on Kaggle, downloads HF data, no local prep needed). Linked via HuggingFace Hub — the train notebook pushes a LoRA adapter, the inference notebook pulls it.

### 01. `notebooks/01_train_lora.ipynb` — train LoRA

Uses Unsloth's `FastVisionModel` + TRL's `SFTTrainer` with `UnslothVisionDataCollator`. Trains the transcribe head (LoRA r=16, language layers only — vision encoder stays frozen) on gold + cleaned-silver crops (~184k examples with `--include-silver`), then `model.push_to_hub(...)` to your HF adapter repo.

The full set is ~11.5k steps (~38h) — too long for one Kaggle session — so the run is capped at `max_steps=2000` (~7h) with checkpoints every 250 steps. To train on more data, rerun with `trainer.train(resume_from_checkpoint=True)` across successive Commits.

Note: Unsloth (open source) is single-GPU, so the second T4 on a `GPU T4 ×2` runtime sits idle — the cap, not the GPU count, is the wall-clock lever.

Setup: Kaggle Secret `HF_TOKEN` (HF write token) attached to the notebook, GPU T4 ×2, Internet On. Edit the two constants in cell 2 if you fork. Runtime ~7h.

### 02. `notebooks/02_inference.ipynb` — predict + submit

Loads Qwen2.5-VL-7B-4bit + the LoRA adapter from HF Hub, runs sync detect → transcribe over the competition test set, writes `/kaggle/working/submission.csv`, then submits it straight to the competition via the Kaggle API.

Attach: `handwritten-to-data` (competition data). Secrets: `KAGGLE_USERNAME` + `KAGGLE_KEY` (from kaggle.com → Settings → API → _Create New Token_) for the auto-submit step; `HF_TOKEN` only for private adapter repos. Set `HF_ADAPTER_REPO = None` in cell 2 to skip the adapter and submit base-model predictions, or `SUBMIT = False` to only write the CSV. Runtime ~4–6h.

Phase-1 toggles live in `configs/pipeline_p1.yaml`:
- `postproc.nms_iou: 0.5` — drops duplicate line bboxes
- `image.deskew: true` — Hough-based skew correction
- `transcribe.crop_margins` — source-aware crop tightness
- `transcribe.few_shot_examples` — per-rtype in-context examples (run `mine_fewshot.py` first)

Switch `CONFIG` in the inference notebook to `configs/pipeline_p1.yaml` to enable.

## Local (Mac) workflow — optional

Only needed if you want to re-build the CV split, clean silver, or score predictions offline. Pure CPU.

```bash
pip install -e .
python scripts/build_cv_split.py           # → data/cv/{train,val}.jsonl
python scripts/clean_silver.py             # → data/cv/{train_clean,silver_clean}.jsonl
python scripts/mine_fewshot.py             # → configs/_fewshot.yaml
python scripts/score_cv.py --pred ...      # composite metric breakdown
```

The CV-split + silver-clean JSONLs are committed to the repo, so the training notebook works out-of-the-box without running these locally.

## Phase 3 — polish

Stubs under `scripts/`:
- `dictation_pseudolabels.py` — aligns Ukrainian National Dictation canonical text to detected lines on dictation pages (closes silver's 0-dictation gap).
- `self_distill.py` — re-labels silver with the fine-tuned model, keeps high-confidence regions.
- `ensemble.py` — 2-seed vote via line-level edit-distance consensus.
