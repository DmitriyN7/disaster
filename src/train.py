from collections.abc import Iterable
from pathlib import Path
from typing import Any, Protocol
import json
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers.optimization import get_linear_schedule_with_warmup
import pandas as pd
import torch


from src.config import (
    BATCH_SIZE,
    CHECKPOINT_DIR,
    LR,
    MAX_LENGTH,
    MODEL_NAME,
    NUM_EPOCHS,
    SEED,
    SUBMISSION_PATH,
    VAL_STEP,
    WARMUP_RATIO,
)
from src.eda import df_test, df_train


class SavePretrained(Protocol):
    """Protocol for Hugging Face objects that can be saved with save_pretrained."""

    def save_pretrained(self, save_directory: str | Path, *args: Any, **kwargs: Any) -> Any:
        """Save model or tokenizer files to a directory."""
        ...


train_texts, val_texts, train_labels, val_labels = train_test_split(
    df_train["text"],
    df_train["target"],
    test_size=0.2,
    random_state=SEED,
    stratify=df_train["target"],
)


def get_device() -> torch.device:
    """Return the best available PyTorch device for transformer fine-tuning."""
    if torch.cuda.is_available():
        return torch.device("cuda")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def build_collate_fn(tokenizer: Any):
    """Create a collator that tokenizes tweets for a BERT-like model."""

    def collate_fn(batch: list[tuple[str, int]] | list[str]) -> dict[str, torch.Tensor]:
        has_labels = isinstance(batch[0], tuple)
        if has_labels:
            texts, labels = zip(*batch, strict=True)
        else:
            texts = batch
            labels = None
        encoded = tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        if labels is not None:
            encoded["labels"] = torch.tensor(labels, dtype=torch.long)
        return encoded

    return collate_fn


def move_batch_to_device(
    batch: dict[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    """Move all tensors in a tokenized batch to the selected device."""
    return {key: value.to(device) for key, value in batch.items()}


def predict_probabilities(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> tuple[list[float], list[int]]:
    """Return positive-class probabilities and gold labels for a labeled loader."""
    model.eval()
    probabilities = []
    targets = []

    with torch.no_grad():
        for batch in data_loader:
            labels = batch["labels"]
            batch = move_batch_to_device(batch, device)
            outputs = model(**batch)
            probs = torch.softmax(outputs.logits, dim=1)[:, 1]

            probabilities.extend(probs.detach().cpu().tolist())
            targets.extend(labels.tolist())

    return probabilities, targets


def predict_unlabeled_probabilities(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> list[float]:
    """Return positive-class probabilities for an unlabeled data loader."""
    model.eval()
    probabilities = []

    with torch.no_grad():
        for batch in data_loader:
            batch = move_batch_to_device(batch, device)
            outputs = model(**batch)
            probs = torch.softmax(outputs.logits, dim=1)[:, 1]
            probabilities.extend(probs.detach().cpu().tolist())

    return probabilities


def classification_metrics(
    probabilities: Iterable[float], targets: Iterable[int]
) -> tuple[list[int], float]:
    """Convert probabilities to labels and compute F1."""
    preds = [int(prob >= 0.5) for prob in probabilities]
    return preds, float(f1_score(list(targets), preds))


def save_best_checkpoint(
    model: SavePretrained,
    tokenizer: SavePretrained,
    val_f1: float,
    checkpoint_dir: Path = CHECKPOINT_DIR,
) -> None:
    """Persist the best fine-tuned transformer checkpoint and metadata."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)
    metadata = {"model_name": MODEL_NAME, "validation_f1": val_f1, "seed": SEED}
    (checkpoint_dir / "training_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


def create_submission(
    test_ids: pd.Series,
    probabilities: Iterable[float],
    submission_path=SUBMISSION_PATH,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Create and save a Kaggle submission dataframe from test probabilities."""
    submission = pd.DataFrame(
        {
            "id": test_ids.astype(int),
            "target": [int(prob >= threshold) for prob in probabilities],
        }
    )
    submission.to_csv(submission_path, index=False)
    return submission


def build_submission_text(df: pd.DataFrame) -> pd.Series:
    """Combine cleaned metadata and tweet text into one legacy linear-model input."""
    return (
        df["keyword"].astype(str) + " " + df["location"].astype(str) + " " + df["text"].astype(str)
    )


def build_model_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Build the dataframe consumed by the ensemble submission model."""
    model_frame = df[["keyword", "location", "text"]].copy()
    model_frame["combined_text"] = build_submission_text(df)
    return model_frame


def build_text_keyword_logistic_pipeline(
    *, C: float = 1.0, class_weight: str | None = "balanced"
) -> Pipeline:
    """Build a TF-IDF + keyword one-hot logistic model without noisy locations."""
    features = ColumnTransformer(
        [
            (
                "text_word_tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.95,
                    strip_accents="unicode",
                    sublinear_tf=True,
                ),
                "text",
            ),
            (
                "text_char_tfidf",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=3,
                    sublinear_tf=True,
                ),
                "text",
            ),
            ("keyword_one_hot", OneHotEncoder(handle_unknown="ignore"), ["keyword"]),
        ]
    )
    classifier = LogisticRegression(
        C=C,
        class_weight=class_weight,
        max_iter=2000,
        random_state=SEED,
        solver="liblinear",
    )
    return Pipeline([("features", features), ("classifier", classifier)])


def build_combined_text_logistic_pipeline(
    *, C: float = 1.0, class_weight: str | None = "balanced"
) -> Pipeline:
    """Build the previous combined-field TF-IDF logistic model for ensembling."""
    features = FeatureUnion(
        [
            (
                "word_tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.95,
                    strip_accents="unicode",
                    sublinear_tf=True,
                ),
            ),
            (
                "char_tfidf",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=3,
                    sublinear_tf=True,
                ),
            ),
        ]
    )
    classifier = LogisticRegression(
        C=C,
        class_weight=class_weight,
        max_iter=2000,
        random_state=SEED,
        solver="liblinear",
    )
    return Pipeline([("features", features), ("classifier", classifier)])


def build_complement_nb_pipeline() -> Pipeline:
    """Build a high-recall word n-gram ComplementNB model for probability averaging."""
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.95,
                    strip_accents="unicode",
                    sublinear_tf=True,
                ),
            ),
            ("classifier", ComplementNB(alpha=0.1)),
        ]
    )


class AveragedProbabilityEnsemble:
    """Average probabilities from diverse lightweight text classifiers."""

    def __init__(self, models: list[tuple[str, Pipeline, str, float]]) -> None:
        self.models = models
        self.fitted_models: list[tuple[str, Pipeline, str, float]] = []

    def _select_features(self, frame: pd.DataFrame, feature_key: str) -> pd.DataFrame | pd.Series:
        if feature_key == "frame":
            return frame

        return frame[feature_key]

    def fit(self, frame: pd.DataFrame, targets: Iterable[int]) -> "AveragedProbabilityEnsemble":
        """Fit every base model on its configured feature view."""
        self.fitted_models = []
        for name, model, feature_key, weight in self.models:
            fitted_model = clone(model)
            fitted_model.fit(self._select_features(frame, feature_key), targets)
            self.fitted_models.append((name, fitted_model, feature_key, weight))
        return self

    def predict_proba(self, frame: pd.DataFrame) -> list[float]:
        """Return weighted average positive-class probabilities."""
        if not self.fitted_models:
            raise RuntimeError("Ensemble must be fitted before predict_proba().")

        weighted_probabilities = [0.0] * len(frame)
        total_weight = 0.0
        for _, model, feature_key, weight in self.fitted_models:
            probabilities = model.predict_proba(self._select_features(frame, feature_key))[:, 1]
            weighted_probabilities = [
                current + (weight * float(probability))
                for current, probability in zip(weighted_probabilities, probabilities, strict=True)
            ]
            total_weight += weight

        return [probability / total_weight for probability in weighted_probabilities]


def build_tfidf_ensemble_model() -> AveragedProbabilityEnsemble:
    """Build the calibrated ensemble used for the public-score submission."""
    return AveragedProbabilityEnsemble(
        [
            ("text_keyword_lr_c1", build_text_keyword_logistic_pipeline(C=1.0), "frame", 2.0),
            ("text_keyword_lr_c2", build_text_keyword_logistic_pipeline(C=2.0), "frame", 1.0),
            ("combined_lr_c1", build_combined_text_logistic_pipeline(C=1.0), "combined_text", 1.0),
            ("combined_complement_nb", build_complement_nb_pipeline(), "combined_text", 1.0),
        ]
    )


def build_tfidf_logistic_pipeline() -> Pipeline:
    """Build the original fast TF-IDF logistic-regression baseline."""
    return build_combined_text_logistic_pipeline(C=2.0, class_weight="balanced")


def find_best_threshold(
    probabilities: Iterable[float], targets: Iterable[int]
) -> tuple[float, float]:
    """Select the decision threshold with the highest validation F1."""
    prob_list = list(probabilities)
    target_list = list(targets)
    best_threshold = 0.5
    best_f1 = -1.0

    for threshold_step in range(200, 801, 5):
        threshold = threshold_step / 1000
        preds = [int(prob >= threshold) for prob in prob_list]
        score = float(f1_score(target_list, preds))
        if score > best_f1:
            best_f1 = score
            best_threshold = threshold

    return best_threshold, best_f1


def save_classical_metadata(val_f1: float, threshold: float) -> None:
    """Persist metadata for the fast classical model used to create submission.csv."""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    metadata = {
        "model_name": "tfidf_weighted_probability_ensemble",
        "validation_f1": val_f1,
        "threshold": threshold,
        "seed": SEED,
    }
    (CHECKPOINT_DIR / "training_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


def run() -> None:
    """Train a calibrated TF-IDF model and write a non-degenerate Kaggle submission."""
    print("Training TF-IDF weighted probability ensemble")

    train_frame = build_model_frame(df_train.loc[train_texts.index])
    val_frame = build_model_frame(df_train.loc[val_texts.index])

    model = build_tfidf_ensemble_model()
    model.fit(train_frame, train_labels)

    val_probs = model.predict_proba(val_frame)
    threshold, tuned_f1 = find_best_threshold(val_probs, val_labels)
    val_preds = [int(prob >= threshold) for prob in val_probs]

    print(f"Best validation threshold: {threshold:.2f}")
    print(classification_report(val_labels, val_preds))
    print("F1:", tuned_f1)
    print("ROC-AUC:", roc_auc_score(val_labels, val_probs))

    full_model = build_tfidf_ensemble_model()
    full_features = build_model_frame(df_train)
    test_features = build_model_frame(df_test)
    full_model.fit(full_features, df_train["target"])
    test_probs = full_model.predict_proba(test_features)
    submission = create_submission(df_test["id"], test_probs, threshold=threshold)

    class_counts = submission["target"].value_counts().to_dict()
    if submission["target"].nunique() < 2:
        raise RuntimeError(
            "Submission contains a single class; tune features or threshold before submitting."
        )

    save_classical_metadata(tuned_f1, threshold)
    print(f"Saved model metadata to {CHECKPOINT_DIR / 'training_metadata.json'}")
    print(f"Saved Kaggle submission to {SUBMISSION_PATH}; class_counts={class_counts}")


def run_transformer() -> None:
    """Fine-tune a BERT-like transformer and report validation metrics."""

    class TextDataset(Dataset):
        """Dataset wrapper for text samples, with optional binary labels."""

        def __init__(self, texts: Any, labels: Any | None = None) -> None:
            self.texts = texts.reset_index(drop=True).astype(str)
            self.labels = None if labels is None else labels.reset_index(drop=True).astype(int)

        def __len__(self) -> int:
            return len(self.texts)

        def __getitem__(self, idx: int) -> tuple[str, int] | str:
            if self.labels is None:
                return self.texts[idx]
            return self.texts[idx], int(self.labels[idx])

    torch.manual_seed(SEED)
    device = get_device()
    print(f"Using device: {device}")
    print(f"Fine-tuning transformer model: {MODEL_NAME}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    model.to(device)
    train_dataset = TextDataset(train_texts, train_labels)
    val_dataset = TextDataset(val_texts, val_labels)
    test_dataset = TextDataset(df_test["text"])
    collate_fn = build_collate_fn(tokenizer)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    total_training_steps = len(train_loader) * NUM_EPOCHS
    warmup_steps = int(total_training_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps,
    )
    best_state_dict = None
    best_val_f1 = -1.0

    for epoch in tqdm(range(NUM_EPOCHS)):
        model.train()
        total_loss = 0.0

        for batch in tqdm(train_loader):
            batch = move_batch_to_device(batch, device)
            optimizer.zero_grad()

            outputs = model(**batch)
            loss = outputs.loss

            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        if (epoch + 1) % VAL_STEP == 0 or epoch == NUM_EPOCHS - 1:
            epoch_probs, epoch_targets = predict_probabilities(model, val_loader, device)
            _, epoch_f1 = classification_metrics(epoch_probs, epoch_targets)

            if epoch_f1 > best_val_f1:
                best_val_f1 = epoch_f1
                best_state_dict = {
                    key: value.detach().cpu().clone() for key, value in model.state_dict().items()
                }

            print(f"Epoch {epoch + 1}, loss={total_loss:.4f}, val_f1={epoch_f1:.4f}")

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        print(f"Loaded best model with val_f1={best_val_f1:.4f}")

    save_best_checkpoint(model, tokenizer, best_val_f1)
    print(f"Saved best model checkpoint to {CHECKPOINT_DIR}")

    all_probs, all_targets = predict_probabilities(model, val_loader, device)
    all_preds, final_f1 = classification_metrics(all_probs, all_targets)

    print(classification_report(all_targets, all_preds))
    print("F1:", final_f1)
    print("ROC-AUC:", roc_auc_score(all_targets, all_probs))

    test_probs = predict_unlabeled_probabilities(model, test_loader, device)
    create_submission(df_test["id"], test_probs)
    print(f"Saved Kaggle submission to {SUBMISSION_PATH}")
