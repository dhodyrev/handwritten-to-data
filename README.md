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
  01_inference.ipynb        # Kaggle: install → load → predict → submission.csv
  02_train_lora.ipynb       # Kaggle: Unsloth QLoRA on prepared data
data/cv/                    # committed CV manifests (small JSONL)
research/                   # reference baselines + EDA notebooks
```

## Local (Mac) workflow

Used for CV-split construction, silver cleaning, few-shot mining, and offline scoring. No GPU needed.

```bash
pip install -e .

python scripts/build_cv_split.py           # data/cv/{train,val}.jsonl
python scripts/clean_silver.py             # data/cv/{train_clean,silver_clean}.jsonl
python scripts/mine_fewshot.py             # configs/_fewshot.yaml
python scripts/prepare_lora_data.py --task transcribe --include-silver
                                           # data/lora/transcribe/{train,val}.jsonl + crops
```

Upload `data/lora/transcribe/` (the JSONL **and** the `_crops_*` directories — the JSONL references crop paths) as a Kaggle dataset for `02_train_lora.ipynb`. Upload `src/`, `scripts/`, and `configs/` as a separate Kaggle dataset (e.g. `htr-source`) for both notebooks.

## Kaggle inference (`notebooks/01_inference.ipynb`)

Attach datasets:
- `handwritten-to-data` (competition data, mounted at `/kaggle/input/handwritten-to-data/`)
- `htr-source` (this repo's `src/` + `scripts/` + `configs/`)
- optional: `htr-lora` (LoRA adapter from `02_train_lora.ipynb`)

The notebook installs Unsloth + Qwen2.5-VL-7B 4-bit, runs `scripts/run_inference.py` against the test manifest, and writes `/kaggle/working/submission.csv` ready for submission.

Phase-1 toggles live in `configs/pipeline_p1.yaml`:
- `postproc.nms_iou: 0.5` — drops duplicate line bboxes
- `image.deskew: true` — Hough-based skew correction
- `transcribe.crop_margins` — source-aware crop tightness
- `transcribe.few_shot_examples` — per-rtype in-context examples (run `mine_fewshot.py` first)

Re-run scoring after each toggle and lock in what moves the composite.

## Kaggle LoRA fine-tune (`notebooks/02_train_lora.ipynb`)

Uses Unsloth's `FastVisionModel` + TRL's `SFTTrainer` with the `UnslothVisionDataCollator`. The notebook trains the transcribe head (LoRA r=16, language layers only — vision encoder stays frozen) on the prepared crops and saves an adapter to `/kaggle/working/lora_adapter`.

Download the adapter, re-upload as a dataset (`htr-lora`), and point `01_inference.ipynb`'s `ADAPTER` variable at it.

## Phase 3 — polish

Stubs under `scripts/`:
- `dictation_pseudolabels.py` — aligns Ukrainian National Dictation canonical text to detected lines on dictation pages (closes silver's 0-dictation gap).
- `self_distill.py` — re-labels silver with the fine-tuned model, keeps high-confidence regions.
- `ensemble.py` — 2-seed vote via line-level edit-distance consensus.
