# Entrepreneurial sensegiving about DSS-backed decisions under uncertainty and sustainability: Evidence from Twitter

This repository provides the open-source code for the paper:

**Entrepreneurial sensegiving about DSS-backed decisions under uncertainty and sustainability: Evidence from Twitter**
Submitted to *Technology in Society*.

The code implements an end-to-end data processing pipeline that transforms raw tweet-level data into an analysis-ready structured dataset with variables used for empirical modeling.

> **Important note on authorship and tooling:**
> The codebase **was refactored, modularized, and documented with the assistance of generative AI**. The overall pipeline design, variable definitions, and research decisions follow the authors’ study design; the implementation was packaged into a reusable, open-source script with AI support.

---

## What this repo does

The main script (`data_clean.py`) runs the following steps:

1. **Load** raw tweet-level data from an Excel file
2. **Text cleaning** (URL/mentions/hashtags removal, normalization)
3. **Language detection** and filtering (keeps **English** only)
4. **Length filter** (minimum cleaned text length)
5. **Engagement variables** (counts and log/rate transforms)
6. **Topic modeling** using **LDA** (dominant topic assignment per tweet)
7. **Sustainability / ESG tagging** (Baier lexicons):

   * hits, binary flags, term density
8. **Uncertainty tagging** (Loughran–McDonald uncertainty lexicon):

   * hits, binary flag, term density
9. **Sentiment** using **VADER**
10. **Export** a single **analysis-ready structured CSV**

---

## Repository structure (recommended)

```text
.
├── data_clean.py
├── LICENSE                      # MIT
├── README.md
└── Twitter_data/
    ├── Twitter_clean.xlsx
    ├── baier_env_lexicon_clean.csv
    ├── baier_social_lexicon_clean.csv
    ├── baier_governance_lexicon_clean.csv
    └── lm_uncertainty_lexicon_clean.csv
```

Outputs will be written to:

```text
Twitter_data/analysis_data/
└── dss_tweets_analysis_structured.csv
```

---

## Installation

Python 3.9+ is recommended.

Install dependencies:

```bash
pip install pandas numpy openpyxl langdetect scikit-learn vaderSentiment
```

---

## Quick start

From the repository root:

```bash
python data_clean.py --base-dir ./Twitter_data --input Twitter_clean.xlsx
```

This will create:

```text
Twitter_data/analysis_data/dss_tweets_analysis_structured.csv
```

---

## Optional flags

### Save intermediate files (debugging)

```bash
python data_clean.py --base-dir ./Twitter_data --save-intermediate
```

### Save LDA topic top-words report

```bash
python data_clean.py --base-dir ./Twitter_data --save-topic-report
```

### Adjust filters / LDA hyperparameters

```bash
python data_clean.py \
  --base-dir ./Twitter_data \
  --min-text-len 20 \
  --n-topics 15 \
  --cv-min-df 10 \
  --cv-max-df 0.95 \
  --n-top-words 15
```

---

## Input data requirements

### Required column

Your input Excel file must include a column named:

* `text` (tweet text)

### Expected columns (recommended)

If present, the script will use these fields:

* `user_name`, `user_id`, `date`, `detail_url`, `comment_id`
* engagement: `comment`, `good`, `share`, `view`

If engagement columns are missing, the script will raise an error (because engagement features are part of the analysis dataset).

---

## Lexicons

This pipeline expects cleaned lexicons (CSV) with a term column (default name: `term`):

* Baier lexicons:

  * `baier_env_lexicon_clean.csv`
  * `baier_social_lexicon_clean.csv`
  * `baier_governance_lexicon_clean.csv`
* Loughran–McDonald (LM) uncertainty lexicon:

  * `lm_uncertainty_lexicon_clean.csv`

If your term column is not named `term`, pass:

```bash
python data_clean.py --base-dir ./Twitter_data --lexicon-term-col YOUR_COLUMN_NAME
```

---

## Output: structured dataset

The exported CSV contains (among others):

* Cleaned text and lengths (`tweet_text_clean`, token/char lengths, logs)
* Engagement metrics (counts, rates, log transforms, offset term)
* ESG framing (E/S/G binaries, hits, densities, combined patterns)
* Uncertainty framing (binary, hits, density)
* Sentiment (VADER compound/neg/neu/pos)
* Topic assignment (`dss_topic`) from LDA

---

## Reproducibility notes

* Language detection uses a fixed seed for deterministic behavior.
* LDA uses a fixed random seed (`random_state=42`) by default.
* Topic modeling outcomes can still vary across environments due to upstream library changes; for strict reproducibility, pin dependency versions.

---

## License

This project is released under the **MIT License**. See `LICENSE` for details.

---

## Citation

If you use this code, please cite the paper (once published) and/or cite this repository.
