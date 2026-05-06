"""Classical TF-IDF training pipeline and model builders."""

from collections.abc import Iterable
import json

import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, roc_auc_score
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.config import CHECKPOINT_DIR, SEED, SUBMISSION_PATH
from src.eda import df_test, df_train
from src.training.common import create_submission
from src.training.data import train_labels, train_texts, val_labels, val_texts


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
