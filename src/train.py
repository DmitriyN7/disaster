from collections.abc import Iterable
from pathlib import Path
from typing import Any, Protocol
import json

from sklearn.metrics import classification_report, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
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
) -> pd.DataFrame:
    """Create and save a Kaggle submission dataframe from test probabilities."""
    submission = pd.DataFrame(
        {
            "id": test_ids.astype(int),
            "target": [int(prob >= 0.5) for prob in probabilities],
        }
    )
    submission.to_csv(submission_path, index=False)
    return submission


def run() -> None:
    """Fine-tune a BERT-like transformer and report validation metrics."""
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
