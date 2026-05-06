"""Compatibility facade for training entry points and model helpers."""

from src.training.classical import (
    AveragedProbabilityEnsemble,
    build_combined_text_logistic_pipeline,
    build_complement_nb_pipeline,
    build_model_frame,
    build_submission_text,
    build_text_keyword_logistic_pipeline,
    build_tfidf_ensemble_model,
    build_tfidf_logistic_pipeline,
    find_best_threshold,
    run,
    save_classical_metadata,
)
from src.training.common import (
    SavePretrained,
    build_collate_fn,
    classification_metrics,
    create_submission,
    get_device,
    move_batch_to_device,
    predict_probabilities,
    predict_unlabeled_probabilities,
    save_best_checkpoint,
)
from src.training.data import train_labels, train_texts, val_labels, val_texts
from src.training.transformer import TextDataset, run_transformer

__all__ = [
    "AveragedProbabilityEnsemble",
    "SavePretrained",
    "TextDataset",
    "build_collate_fn",
    "build_combined_text_logistic_pipeline",
    "build_complement_nb_pipeline",
    "build_model_frame",
    "build_submission_text",
    "build_text_keyword_logistic_pipeline",
    "build_tfidf_ensemble_model",
    "build_tfidf_logistic_pipeline",
    "classification_metrics",
    "create_submission",
    "find_best_threshold",
    "get_device",
    "move_batch_to_device",
    "predict_probabilities",
    "predict_unlabeled_probabilities",
    "run",
    "run_transformer",
    "save_best_checkpoint",
    "save_classical_metadata",
    "train_labels",
    "train_texts",
    "val_labels",
    "val_texts",
]
