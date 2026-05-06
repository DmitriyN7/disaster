# Disaster Tweets Classification

This project solves a binary tweet classification task: predict whether a tweet reports a real disaster (`target = 1`) or uses disaster-related language figuratively (`target = 0`). The repository includes exploratory data analysis, preprocessing, feature engineering, and a simple PyTorch baseline for the Kaggle **Natural Language Processing with Disaster Tweets** dataset.

## Key Results

**Data:** 7,613 labeled training tweets and 3,263 test tweets.
**Class balance:** 4,342 non-disaster tweets and 3,271 disaster tweets; the positive class share is approximately 43%.
**EDA findings:**

- missing values in `keyword` are rare: about 0.8% in both train and test;
- `location` is much noisier: about 33% missing values;
- train/test missing-value patterns are similar, which suggests the splits are comparable;
- engineered text meta-features include word count, unique word count, stop-word count, character count, punctuation count, hashtag count, and mention count.
  **Data cleaning:** missing `keyword`/`location` values are filled, unstable characters are removed from categorical fields, and several known contradictory duplicate labels are corrected manually.
  **Baseline model:** logistic regression with TF-IDF in the notebook achieved mean `F1 ≈ 0.522` on 5-fold cross-validation.
  **Main model:** a neural text classifier with `EmbeddingBag(mean) + Linear` is trained on tweet text and selects the best checkpoint by validation F1.
  **Notebook result:** after 5 epochs, the PyTorch model reached `F1 ≈ 0.622`, `ROC-AUC ≈ 0.762`, and accuracy of about `0.71` on the validation split.

## Repository Structure

```text
.
├── data/
│   ├── train.csv              # training set with target labels
│   ├── test.csv               # test set without target labels
│   └── sample_submission.csv  # example Kaggle submission
├── src/
│   ├── config.py              # training hyperparameters
│   ├── eda.py                 # data loading, cleaning, and feature engineering
│   ├── train.py               # tokenization, datasets, model, and training loop
│   └── main.py                # training entry point
|   └── experiments.ipynb      # experiments
├── pyproject.toml             # project dependencies
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

The script runs the full pipeline: it loads the data, applies preprocessing from `src/eda.py`, trains the model from `src/train.py`, selects the best checkpoint by validation F1, and prints `classification_report`, `F1`, and `ROC-AUC`.

### 5. Open the EDA notebook

```bash
uv run python -m ipykernel install --user --name disaster
uv run --with jupyter jupyter lab eda.ipynb
```

The second command temporarily provides Jupyter Lab via `uv --with`. You can also open the notebook in an IDE that can use the `.venv` kernel.

## Training Configuration

The main training parameters are defined in `src/config.py`:

| Parameter    |  Value | Description                     |
| ------------ | -----: | ------------------------------- |
| `LR`         | `3e-4` | Adam learning rate              |
| `NUM_EPOCHS` |   `20` | number of training epochs       |
| `SEED`       |    `7` | random seed used for the split  |
| `VAL_STEP`   |    `5` | validation evaluation frequency |

To experiment with these values, edit `src/config.py` and rerun:

```bash
uv run python src/main.py
```

## Future Improvements

- Generate a `submission.csv` file for the Kaggle test set.
- Save the best model checkpoint to disk.
- Try stronger text features: TF-IDF n-grams, pretrained embeddings, or a transformer model.
- Move preprocessing into pure functions and cover them with tests.
- Add CLI arguments for data paths, seed, number of epochs, and batch size.
