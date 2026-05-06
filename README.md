# Disaster Tweets Classification

This project solves a binary tweet classification task: predict whether a tweet reports a real disaster (`target = 1`) or uses disaster-related language figuratively (`target = 0`). The repository includes exploratory data analysis, preprocessing, feature engineering, and a PyTorch/transformer training pipeline for the Kaggle **Natural Language Processing with Disaster Tweets** dataset.

## Key Results

**Data:** 7,613 labeled training tweets and 3,263 test tweets.
**Class balance:** 4,342 non-disaster tweets and 3,271 disaster tweets; the positive class share is approximately 43%.
**EDA findings:**

- missing values in `keyword` are rare: about 0.8% in both train and test;
- `location` is much noisier: about 33% missing values;
- train/test missing-value patterns are similar, which suggests the splits are comparable;
- engineered text meta-features include word count, unique word count, stop-word count, character count, punctuation count, hashtag count, and mention count.
  **Data cleaning:** missing `keyword`/`location` values are filled, unstable characters are removed from categorical fields, and several known contradictory duplicate labels are corrected manually.
  **Baseline model:** logistic regression with TF-IDF in the notebook achieved mean `F1 ≈ 0.522` on 5-fold cross-validation
  **Main model:** a fast single TF-IDF word/character n-gram logistic-regression pipeline trains on cleaned keyword, location, and tweet text, tunes the decision threshold on validation F1, and refuses to write a single-class submission. This remains the default submission path because it produced the stronger public score (`0.80416`) than the weighted TF-IDF ensemble experiment (`0.80140`).
  **Ensemble option:** `build_tfidf_ensemble_model()` remains available in `src/train.py` for offline experiments, but it is not used by the default run path.
  **Transformer option:** `run_transformer()` in `src/train.py` keeps the earlier `distilbert-base-uncased` fine-tuning path for optional experiments when transformer dependencies are installed.

## Repository Structure

```text
.
├── data/
│   ├── train.csv              # training set with target labels
│   ├── test.csv               # test set without target labels
│   └── sample_submission.csv  # example Kaggle submission
├── src/
│   ├── config.py              # training hyperparameters
│   ├── preprocessing.py       # pure data cleaning and feature engineering helpers
│   ├── eda.py                 # data loading entry points
│   ├── train.py               # TF-IDF training, threshold tuning, submission generation, and optional transformer path
│   ├── main.py                # training entry point
│   └── experiments.ipynb      # experiments
├── pyproject.toml             # project dependencies
├── tests/                     # preprocessing unit tests
└── uv.lock                    # uv lock file
```

## Quick Start

### 1. Install uv

If `uv` is not installed yet:

```bash
pip install uv
```

You can also use the official `uv` installer if it is preferred in your environment.

### 2. Clone the repository

```bash
git clone <repo-url>
cd disaster
```

### 3. Install dependencies

The project is defined in `pyproject.toml`, with versions pinned in `uv.lock`:

```bash
uv sync
```

### 4. Run training

```bash
uv run python src/main.py
```

The script runs the full pipeline: it loads the data, applies pure preprocessing helpers from `src/preprocessing.py`, trains the single TF-IDF logistic-regression model from `src/train.py`, tunes the decision threshold by validation F1, writes Kaggle test predictions to `submission.csv`, validates that the submission contains both classes, and prints `classification_report`, `F1`, and `ROC-AUC`.

### 5. Open the EDA notebook

```bash
uv run python -m ipykernel install --user --name disaster
uv run --with jupyter jupyter lab eda.ipynb
```

The second command temporarily provides Jupyter Lab via `uv --with`. You can also open the notebook in an IDE that can use the `.venv` kernel.

## Training Configuration

The default submission path uses the single TF-IDF logistic-regression model configured in `src/train.py`; the weighted ensemble is kept only as an experiment after its public score regressed. The optional transformer training parameters are defined in `src/config.py`:

| Parameter      | Value                     | Description                                  |
| -------------- | ------------------------- | -------------------------------------------- |
| `MODEL_NAME`   | `distilbert-base-uncased` | pretrained BERT-like model from Hugging Face |
| `MAX_LENGTH`   | `128`                     | maximum tokenized tweet length               |
| `BATCH_SIZE`   | `16`                      | training and validation batch size           |
| `LR`           | `3e-5`                    | AdamW learning rate                          |
| `NUM_EPOCHS`   | `3`                       | number of fine-tuning epochs                 |
| `SEED`         | `7`                       | random seed used for the split               |
| `VAL_STEP`     | `1`                       | validation evaluation frequency              |
| `WARMUP_RATIO` | `0.1`                     | share of steps used for LR warmup            |

To experiment with these values, edit `src/config.py` and rerun:

```bash
uv run python src/main.py
```

## Generated Artifacts

After `uv run python src/main.py` completes, the repository contains:

- `artifacts/best_model/training_metadata.json` with the validation F1, tuned threshold, seed, and default single-model name.
- `submission.csv` with Kaggle test-set `id,target` predictions generated by the calibrated single TF-IDF model.

## Future Improvements

- Add CLI arguments for data paths, seed, number of epochs, and batch size.
