# Tech Debt Cleanup — Completed

Generated: 2026-02-28

## What Was Done

### Deleted (Category A — dead code)
- [x] **18 root JSON exports** (`ufc-predictor-2026-*.json`) — stale prediction snapshots
- [x] **7 Python scraping scripts** — `scrape_ufc_stats_library.py`, `scrape_ufc_stats_unparsed_data.py`, `run_scraper.py`, `analyze_fights.py`, `compare_models.py`, `ufc_predictor.py`, `scrape_ufc_stats_config.yaml`
- [x] **4 Jupyter notebooks** — `scrape_ufc_stats_*.ipynb` (historical scraping phase)
- [x] **2 dead Python support files** — `test_predictions.py` (broken imports), `requirements.txt` (for dead pipeline)
- [x] **2 misc dead files** — `prediction.log`, `debug.html`
- [x] **`__pycache__/`** — compiled Python cache

### Deleted (Category B — confirmed by Mike)
- [x] **`fantasy_mma/`** — separate full-stack app (FastAPI + React), not part of active workflow
- [x] **`docs/`** — research document, not referenced by code

### Deleted (Dead directories)
- [x] **`mma-scraper/`** — old Selenium FanDuel scraper + error screenshots
- [x] **`ufc_project/`** subdirectory — abandoned Railway config + empty .git
- [x] **`ufc_stats/`** — CSV data output from dead scraping pipeline

### Reorganized (Category C)
- [x] **8 analysis scripts moved to `scripts/`** with paths adjusted to use `path.join(__dirname, '..')`:
  - `scripts/backtest.js` — scores v1/v2 predictions against results
  - `scripts/backtest-v3.js` — compares v1 vs v3 engine on same fights
  - `scripts/compare-versions.js` — v2 vs v4 side-by-side for specific events
  - `scripts/error-analysis.js` — deep error categorization
  - `scripts/inspect-dump.js` — data quality audit of db-dump.json
  - `scripts/round-by-wc.js` — round distribution by weight class
  - `scripts/wc-method-analysis.js` — finish method probabilities by weight class
  - `scripts/auto-verifier.js` — Puppeteer-based result verification

### Kept (confirmed active by Mike)
- [x] **`chrome-extension/`** — actively used to collect fighter data for the app

---

## Current Project Structure

```
ufc_project/
├── start_app.bat          # Entry point
├── save_server.py         # Flask server (:5555)
├── .env                   # API keys (ODDS_API_KEY, GEMINI_KEY)
├── index.html             # Main web app
├── styles.css             # Stylesheet
├── storage.js             # IndexedDB persistence
├── fight-card-fetcher.js  # Fight card discovery
├── fighter-data-fetcher.js# Fighter stats fetching
├── prediction-engine.js   # 3-layer prediction model (v5)
├── confidence-ranker.js   # Prediction ranking
├── accuracy-tracker.js    # Accuracy tracking
├── ai-analyzer.js         # Model performance analysis
├── chrome-ai.js           # Chrome on-device AI
├── ui-components.js       # DOM helpers
├── app.js                 # Main controller
├── package.json           # Node dependencies
├── package-lock.json      # Locked deps
├── .gitignore             # Git exclusions
├── LICENSE                # MIT license
├── README.md              # Project docs
├── chrome-extension/      # Data collection extension
├── results/               # Saved predictions & imports
├── scripts/               # Offline analysis tools (8 files)
└── tasks/                 # This file
```

---

## Remaining Action Items

- [ ] **Security: Move ODDS_API_KEY out of save_server.py** — hardcoded at line 32, should load from `.env` via `python-dotenv`. Key is in git history — consider rotating.
- [ ] **Security: Fix .gitignore** — has `**` as last line which may negate other rules. Review.
- [ ] **Cleanup: Prune `results/`** — 29 JSON files including 12 old extension-import batches from Feb 6. Consider keeping only recent exports.
- [ ] **Cleanup: Review `package.json` deps** — `puppeteer`, `yargs` only used by `scripts/auto-verifier.js`. `csv-parser` appears unused. `axios`, `cheerio` may only be needed by scripts, not the browser app.
