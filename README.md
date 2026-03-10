# mhealth-nlp-study[README.md](https://github.com/user-attachments/files/25860207/README.md)
# Tracking Subjective Symptom Improvement from Patient Narratives in Mobile Health

This repository contains the analysis workflow for the manuscript:

**Tracking subjective symptom improvement from patient narratives in Mobile Health: an observational Natural Language Processing study**

## Overview

This project examines whether patient-generated review narratives on a Chinese mHealth platform contain linguistic signals associated with:

- patient satisfaction
- perceived symptom improvement

The workflow includes:

- text preprocessing
- sentiment extraction
- keyword-based improvement indicator generation
- TF-IDF feature extraction
- regression analysis for satisfaction ratings
- classification analysis using Logistic Regression and Random Forest

## Repository structure

```text
mhealth-nlp-github-repo/
├── README.md
├── requirements.txt
├── notebooks/
│   └── analysis_pipeline.ipynb
├── scripts/
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── modeling.py
│   └── utils.py
└── data/
    └── sample_data_format.csv
```

## Data availability

The original dataset consists of publicly available user reviews collected from an online mHealth platform. Because the data include user-generated content gathered under the source platform's terms of use, the raw dataset is **not redistributed in this repository**.

A sample file format is provided in `data/sample_data_format.csv` to illustrate the expected input structure.

## Expected input data structure

The analysis expects a CSV file with the following columns:

- `review_id`: unique review identifier
- `review_text`: original review text
- `clean_text`: preprocessed review text
- `satisfaction_rating`: numerical satisfaction rating
- `improvement_label`: binary keyword-derived perceived improvement indicator (0/1)
- `sentiment_score`: sentiment polarity score

## Analytical workflow

### 1. Text preprocessing

The preprocessing pipeline performs:

- text cleaning
- punctuation removal
- whitespace normalization
- optional stopword removal
- token preparation for vectorization

### 2. Feature engineering

The following text-derived variables are generated:

- **sentiment polarity**: emotional value signal extracted from the review text
- **keyword-derived perceived improvement indicator**: binary indicator based on predefined symptom-improvement expressions
- **TF-IDF features**: weighted lexical features capturing word importance across the corpus

### 3. Modeling

Two main modeling tasks are included:

- **linear regression** predicting satisfaction ratings from sentiment polarity
- **classification models** predicting the keyword-derived perceived improvement indicator from TF-IDF features

Classification models included:

- Logistic Regression
- Random Forest

## How to run

### Install dependencies

```bash
pip install -r requirements.txt
```

### Open the notebook

```bash
jupyter notebook notebooks/analysis_pipeline.ipynb
```

### Or run scripts directly

The scripts in `scripts/` can be imported into your own notebook or executed as modules after adapting file paths.

## Notes on interpretation

The perceived improvement variable in this workflow is a **keyword-derived exploratory proxy** rather than a clinically validated outcome label. Accordingly, model outputs should be interpreted as exploratory evidence of narrative signal detection, not as externally validated clinical prediction.

## Citation

If you use or adapt this workflow, please cite the associated manuscript.
