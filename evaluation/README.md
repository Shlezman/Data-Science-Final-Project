# SentiSense — Evaluation Harness

This directory contains the tools to evaluate the SentiSense multi-agent
pipeline against a manually-labelled golden dataset and compare multiple
LLM models.

---

## Directory Structure

```
evaluation/
├── golden_dataset.csv      ← Golden dataset (headline + 6 gold relevance scores)
├── metrics.py              ← Pure metric functions (MAE, Within-1/2, Pearson r)
├── evaluate.py             ← Main evaluation script (runs pipeline, saves results)
├── report.py               ← Leaderboard + per-model breakdown generator
├── results/                ← Created automatically by evaluate.py
│   ├── qwen2.5_14b_predictions.csv
│   ├── qwen2.5_14b_metrics.json
│   └── leaderboard.md      ← Created by report.py
└── README.md               ← This file
```

---

## What Is Evaluated

The pipeline produces 7 outputs per headline. Only the 6 relevance scores
are evaluated against the golden dataset:

| Output | Evaluated? |
|---|---|
| `relevance_category_1` — Politics & Government | ✅ Yes |
| `relevance_category_2` — Economy & Finance | ✅ Yes |
| `relevance_category_3` — Security & Military | ✅ Yes |
| `relevance_category_4` — Health & Medicine | ✅ Yes |
| `relevance_category_5` — Science & Climate | ✅ Yes |
| `relevance_category_6` — Technology | ✅ Yes |
| `global_sentiment` | ❌ No (saved but not evaluated) |

---

## Prerequisites

1. **Ollama running locally** with the model you want to evaluate:
   ```bash
   ollama serve
   ollama pull qwen2.5:14b   # or whichever model you are testing
   ```

2. **Python environment** with the `processing_engine` package installed:
   ```bash
   cd processing_engine
   uv sync          # or: pip install -e .
   ```

3. **Golden dataset** at `evaluation/golden_dataset.csv`.
   The sample file contains 20 labelled headlines. Replace it with your
   full manually-labelled dataset before running a real evaluation.

---

## Golden Dataset Format

The CSV must have these columns (in any order):

```
headline, gold_cat_1, gold_cat_2, gold_cat_3, gold_cat_4, gold_cat_5, gold_cat_6
```

Optional columns (used as pipeline metadata, not evaluated):
```
date, source, hour, popularity
```

Gold scores must be integers in the range **0–10** following the rubric:

| Score | Meaning |
|---|---|
| 0 | Completely unrelated |
| 1–3 | Tangentially related |
| 4–6 | Moderately related |
| 7–9 | Strongly related |
| 10 | Quintessential example |

---

## Step 1 — Validate the Golden Dataset (dry run)

Before running the full pipeline, check that your CSV is correctly formatted:

```bash
python -m evaluation.evaluate \
    --golden evaluation/golden_dataset.csv \
    --dry-run
```

This loads and validates the CSV without calling the LLM.

---

## Step 2 — Run the Evaluation

### Option A — Single model

```bash
python -m evaluation.evaluate \
    --golden  evaluation/golden_dataset.csv \
    --models  qwen2.5:14b \
    --output  evaluation/results/
```

This will:
1. Run every headline through the full 7-agent pipeline.
2. Save predictions to `results/qwen2.5_14b_predictions.csv`.
3. Compute MAE, Within-1/2 Accuracy, and Pearson r per category.
4. Save metrics to `results/qwen2.5_14b_metrics.json`.
5. Print a summary table to stdout.

### Option B — Multiple models in one command

```bash
python -m evaluation.evaluate \
    --golden  evaluation/golden_dataset.csv \
    --models  qwen2.5:14b llama3.1:8b mistral:7b \
    --output  evaluation/results/
```

This evaluates each model sequentially and automatically prints a
ranked leaderboard at the end comparing all models. It also saves
`results/leaderboard.md`.

> **Note:** Make sure each model is pulled in Ollama before running:
> `ollama pull llama3.1:8b`

---

## Step 3 — Generate or Refresh the Leaderboard

If you ran models separately on different days, you can regenerate
the leaderboard at any time from the saved `*_metrics.json` files:

```bash
python -m evaluation.report \
    --results evaluation/results/ \
    --output  evaluation/results/leaderboard.md
```

This reads all `*_metrics.json` files in the results directory and produces:
- A ranked leaderboard printed to stdout.
- A Markdown file at `results/leaderboard.md` ready to paste into
  `EVALUATION_REPORT.md`.

---

## Metrics Reference

| Metric | Formula | Interpretation |
|---|---|---|
| **MAE** | mean(\|predicted − gold\|) | Average error in score points. Lower is better. |
| **Within-1 Accuracy** | % where \|predicted − gold\| ≤ 1 | Primary ranking metric. Accounts for human subjectivity. |
| **Within-2 Accuracy** | % where \|predicted − gold\| ≤ 2 | Looser tolerance. |
| **Pearson r** | correlation(predicted, gold) | Ranking agreement. 1.0 = perfect. |
| **Composite Score** | avg Within-1 across 6 categories | Single number for model ranking. |

---

## Reproducibility Checklist

Before each model run, confirm:

- [ ] `ollama serve` is running
- [ ] The correct model is loaded (`ollama run <model>` to warm up)
- [ ] The same `golden_dataset.csv` is used (check row count)
- [ ] The same `processing_engine/` code version is used (`git log -1`)
- [ ] `SENTISENSE_OLLAMA_TEMPERATURE=0.1` (default — do not change between runs)
- [ ] `SENTISENSE_OLLAMA_NUM_CTX=8192` (default — do not change between runs)
- [ ] Results are saved before running the next model

---

## Environment Variables

All pipeline settings can be overridden via environment variables:

| Variable | Default | Description |
|---|---|---|
| `SENTISENSE_OLLAMA_MODEL` | `qwen2.5:14b` | Model to use (overridden by `--model`) |
| `SENTISENSE_OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `SENTISENSE_OLLAMA_TEMPERATURE` | `0.1` | LLM temperature (keep fixed across runs) |
| `SENTISENSE_OLLAMA_NUM_CTX` | `8192` | Context window size (keep fixed across runs) |
| `SENTISENSE_OLLAMA_TIMEOUT` | `120` | Request timeout in seconds |
| `SENTISENSE_AGENT_RECURSION_LIMIT` | `10` | Max ReAct loop steps per agent |

---

## Adding Your Real Golden Dataset

1. Replace `evaluation/golden_dataset.csv` with your full labelled dataset.
2. Keep the same column names: `headline, gold_cat_1, …, gold_cat_6`.
3. Run the dry-run validation first to catch any formatting issues.
4. Run `evaluate.py` for each model.
5. Run `report.py` to generate the leaderboard.
6. Paste the leaderboard Markdown into `EVALUATION_REPORT.md` Sections 5 and 6.
