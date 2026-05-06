"""Common training helpers shared by classical and transformer pipelines."""

from collections.abc import Iterable
from pathlib import Path
from typing import Any, Protocol
import json

import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from src.config import CHECKPOINT_DIR, MAX_LENGTH, MODEL_NAME, SEED, SUBMISSION_PATH


class SavePretrained(Protocol):
    """Protocol for Hugging Face objects that can be saved with save_pretrained."""

    def save_pretrained(self, save_directory: str | Path, *args: Any, **kwargs: Any) -> Any:
        """Save model or tokenizer files to a directory."""
        ...


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
