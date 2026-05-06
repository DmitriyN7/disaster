"""Transformer fine-tuning pipeline."""

from typing import Any

import torch
from sklearn.metrics import classification_report, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers.optimization import get_linear_schedule_with_warmup

from src.config import (
    BATCH_SIZE,
    CHECKPOINT_DIR,
    LR,
    MODEL_NAME,
    NUM_EPOCHS,
    SEED,
    SUBMISSION_PATH,
    VAL_STEP,
    WARMUP_RATIO,
)
from src.eda import df_test
from src.training.common import (
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


def run_transformer() -> None:
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
