# self_ranking_platform

Streamlit UI for manually annotating Hebrew news headlines. Used to build the ground-truth dataset that the `processing_engine` evaluation harness is measured against.

## What it does

- Upload a CSV of headlines (output from `mivzakim_scraper`)
- Score each headline across 6 relevance categories (0–10) and a global sentiment axis (−10 to +10)
- Headlines are sampled randomly without replacement so multiple annotators can work on the same file without overlap
- Export completed annotations as a CSV compatible with the `processing_engine/evaluation/golden_dataset.csv` schema

## Setup

```bash
pip install -r requirements.txt
streamlit run ranking_script.py --server.headless true
```

## Annotation Schema

| Field | Range | Description |
|-------|-------|-------------|
| `politics_government` | 0–10 | Relevance to politics or government |
| `economy_finance` | 0–10 | Relevance to economy or finance |
| `security_military` | 0–10 | Relevance to security or military affairs |
| `health_medicine` | 0–10 | Relevance to health or medicine |
| `science_climate` | 0–10 | Relevance to science or climate |
| `technology` | 0–10 | Relevance to technology |
| `global_sentiment` | −10–+10 | Overall tone (negative ↔ positive) |

## Input CSV format

The uploaded CSV must contain at minimum a `headline` column. Additional columns (`date`, `source`, `hour`, `popularity`) are passed through to the export.

## Output

The exported CSV matches the golden dataset schema expected by:
```bash
python -m processing_engine.evaluation.evaluate --golden <exported_file>.csv
```
