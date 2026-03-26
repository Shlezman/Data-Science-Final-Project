# mivzakim_scraper

Playwright-based scraper for Hebrew news headlines from [mivzakim.net](https://mivzakim.net). Supports two scraping modes: by date range and by keyword search.

## Output

Appends rows to `headlines.csv` (created in the working directory):

| Column | Description |
|--------|-------------|
| `date` | Publication date (YYYY-MM-DD) |
| `source` | News source name |
| `hour` | Publication time (HH:MM) |
| `popularity` | Importance/priority class from the site |
| `headline` | Hebrew headline text |

## Setup

```bash
pip install -e .
playwright install firefox
```

## Usage

### Scrape by date range (default)

```bash
python main.py
```

Scrapes backwards from the configured `resume_date` for `days` days, in batches of `batch_size` concurrent browser sessions. Edit `main.py` to adjust the date range and batch size.

### Scrape by keyword (programmatic)

```python
from scrape import get_search_data

df = get_search_data(keywords={"קורונה", "חיסון"}, num_pages=3)
df.to_csv("search_results.csv", index=False)
```

## Module layout

| File | Purpose |
|------|---------|
| `mivzakim_scraper.py` | `Scraper` class — paginates through a single date's archive page |
| `mivzakim_search_scraper.py` | `SearchScraper` class — searches by keyword |
| `scrape.py` | Batch orchestration: concurrency, retries, deduplication, CSV append |
| `main.py` | CLI entry point — configure date range here and run |
| `utils.py` | Session/cookie persistence, random mouse movements (anti-detection) |

## Anti-detection

- Random user agents and viewport sizes per browser context
- Random mouse movements before page reads
- Session and cookie state persisted across requests
- Headless Firefox (lower fingerprint than Chromium)
