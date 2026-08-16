# Figures guide — filling every placeholder in `Final_Project_book.md`

Two kinds of placeholders exist: **diagrams (Figures 2–5)** — already generated
as SVGs in this directory — and **dashboard screenshots (Figures 6–11)** which
you capture from the live site.

## 0. Setup (once)

1. Open the dashboard in a desktop browser, maximized:
   - `https://sentisens.cs.colman.ac.il` once the DNS record is live, otherwise
     `https://193.106.55.109` (accept the name-mismatch warning) or
     `http://193.106.55.109:3000`.
2. Log in with the site password.
3. **Switch to the light theme** (sun/moon button, top-right) — light reads far
   better in a printed/PDF book.
4. macOS capture: `⇧⌘4`, drag the exact card you need. Save as PNG straight
   into this `figures/` directory with the filenames below.
5. Keep one consistent browser width for all screenshots (e.g. 1440 px) so the
   figures look uniform in the book.

## 1. Diagrams (already made — just convert)

| Figure | File | Convert |
|---|---|---|
| Figure 2 — system architecture | `fig2_system_architecture.svg` | open the SVG in a browser → screenshot, or `rsvg-convert -w 1960 -o fig2.png fig2_system_architecture.svg` |
| Figure 3 — two-host topology | `fig3_deployment_topology.svg` | same |
| Figure 4 — chronological split | `fig4_chronological_split.svg` | same |
| Figure 5 — registry lifecycle | `fig5_registry_lifecycle.svg` | same |

If the book stays Markdown, reference the SVGs directly — no conversion needed.

## 2. Screenshots (Figures 6–11)

| Figure | Where | What to set up first | Crop | Save as |
|---|---|---|---|---|
| **Figure 6** — hero + model performance | **Dashboard** tab, top | Nothing — but shoot on a trading day *after* the nightly run (≈18:45 IL) so the hero shows today's call | Hero card ("Next-day forecast", TA-125 UP/DOWN) **plus** the "Model performance" card below it (Core metrics / Classification metrics / Sample coverage eval-vs-live bar) | `fig6_hero_performance.png` |
| **Figure 7** — EDA panels | **Dashboard** tab | Expand the collapsible "Exploratory data analysis" card | The whole EDA card: volume, mean sentiment, sentiment distribution, highest-category relevance, category correlation, validation quality + KPI row (two screenshots stacked if too tall) | `fig7_eda_panels.png` |
| **Figure 8** — all-days 3-D centroids | "**3D centroids**" button in the global header → drawer opens, "All days" view | Drag to a rotation where the 8 labeled cluster-center diamonds (K0–K7) are visible; note the caption caveat below | The drawer plot incl. the time slider and legend | `fig8_centroids_all_days.png` |
| **Figure 9** — single-day headline cloud | Same drawer → click any point (or "Single day"), pick a news-heavy weekday | Choose axes (X/Y/Z selectors) that spread the cloud; hover one headline so a Hebrew tooltip shows | The single-day plot with day centroid + cluster centers | `fig9_day_headline_cloud.png` |
| **Figure 10** — persona votes | **Simulator** tab → "Who says what?" card | Pick a **settled** date in the date selector so the VerdictRow shows "Model says …" *and* "Actually happened" | The persona grid + the verdict row | `fig10_persona_votes.png` |
| **Figure 11** — Models panel / registry leaderboard | Hidden operator view: press the invisible "Serving …" area top-right, or add `#models` to the URL | **Activate the intended champion first** (see note) — the active row is highlighted | The leaderboard table (version, ROC-AUC + CI, MCC, accuracy, n, Activate column) with the active champion visible | `fig11_models_leaderboard.png` |

**Figure 11 note — do this before shooting:** the registry currently has
`tcn-20260702-1351` active (manual activation). The book's headline result is
`patchtst-20260702-1351` (OOS accuracy 0.5780). One click on *Activate* next to
the PatchTST row makes the screenshot match the book — and it is also the
better model on every OOS metric.

**Figure 8 caption caveat:** the book caption says "colored by KMeans cluster",
but the all-days view renders points in a single color (cluster shown on hover
only; only the 8 center diamonds are cluster-colored). Either soften the
caption ("with the eight KMeans cluster centers as labeled markers") or ask for
a small UI change to color points by nearest cluster before capturing.

## 3. Replacing the placeholders in the book

For each figure, delete the block-quoted placeholder and insert an image line +
italic caption. Example for Figure 6 — replace:

```
> **[Figure 6 placeholder: dashboard screenshot - hero + model performance.]**
```

with:

```
![Figure 6](figures/fig6_hero_performance.png)
*Figure 6: Dashboard - prediction hero and the active champion's
model-performance panel (eval-seeded cumulative accuracy with the
eval/live split shown).*
```

Then in the **List of Figures** near the top, remove the `*(placeholder)*` /
`*(screenshot placeholder)*` suffixes.

All eleven replacement targets, by book line (search for `placeholder`):
Figure 2 (§3.1), Figure 3 (§3.5 topology), Figure 4 (§3.3), Figure 5 (§3.4),
Figures 6–10 (§3.5, one block of five), Table/Figure 11 (§4.2.8 — two spots:
the registry-leaderboard export note after Table 12, and the Figure 11
screenshot placeholder at the end of §4.2.8).

For the §4.2.8 *Table/Figure 11* "export the full registry leaderboard" spot you
can either paste the Models-panel screenshot or export the table as text with:

```bash
PGPASSWORD=<pw> psql -h localhost -U postgres -d sentisense -c \
 "SELECT version, model_type, round(oos_roc_auc::numeric,4) roc_auc,
         round(oos_accuracy::numeric,4) acc, round(oos_mcc::numeric,4) mcc, oos_n
  FROM model_registry ORDER BY oos_accuracy DESC NULLS LAST;"
```
