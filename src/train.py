from collections import Counter
import re

from sklearn.metrics import classification_report, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch

from src.config import LR, NUM_EPOCHS, SEED, VAL_STEP
from src.eda import df_train


def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())


train_texts, val_texts, train_labels, val_labels = train_test_split(
    df_train["text"],
    df_train["target"],
    test_size=0.2,
    random_state=SEED,
    stratify=df_train["target"],
)
counter = Counter()

for text in train_texts:
    counter.update(tokenize(text))
min_count = 2

vocab = {"<unk>": 0}

for word, count in counter.items():
    if count >= min_count:
        vocab[word] = len(vocab)


class Model(nn.Module):
    def __init__(self, vocab_size, embed_dim=100):
        super().__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, mode="mean")
        self.fc = nn.Linear(embed_dim, 1)

    def forward(self, x, offsets):
        x = self.embedding(x, offsets)
        return self.fc(x)


class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts.reset_index(drop=True)
        self.labels = labels.reset_index(drop=True)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]


def collate_fn(batch):
    text_list = []
    offsets = [0]
    labels = []

    for text, label in batch:
        encoded = encode(text)

        if len(encoded) == 0:
            encoded = [0]

        text_list.extend(encoded)
        offsets.append(len(text_list))
        labels.append(label)

    offsets = offsets[:-1]

    return (
        torch.tensor(text_list, dtype=torch.long),
        torch.tensor(offsets, dtype=torch.long),
        torch.tensor(labels, dtype=torch.float32),
    )


def encode(text):
    return [vocab.get(token, 0) for token in tokenize(text)]


def run():
    train_dataset = TextDataset(train_texts, train_labels)
    val_dataset = TextDataset(val_texts, val_labels)

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=collate_fn,
    )

    model = Model(vocab_size=len(vocab), embed_dim=100)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()
    best_state_dict = None
    best_val_f1 = 0.0

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0

        for x, offsets, y in train_loader:
            optimizer.zero_grad()

            logits = model(x, offsets).squeeze(1)
            loss = criterion(logits, y)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if epoch % VAL_STEP == 0:
            model.eval()
            epoch_probs = []
            epoch_targets = []

            with torch.no_grad():
                for x, offsets, y in val_loader:
                    logits = model(x, offsets).squeeze(1)
                    probs = torch.sigmoid(logits)
                    epoch_probs.extend(probs.numpy())
                    epoch_targets.extend(y.numpy())

            epoch_preds = [int(prob >= 0.5) for prob in epoch_probs]
            epoch_f1 = f1_score(epoch_targets, epoch_preds)

            if epoch_f1 > best_val_f1:
                best_val_f1 = epoch_f1
                best_state_dict = {
                    key: value.detach().clone() for key, value in model.state_dict().items()
                }

            print(f"Epoch {epoch + 1}, loss={total_loss:.4f}, val_f1={epoch_f1:.4f}")

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        print(f"Loaded best model with val_f1={best_val_f1:.4f}")

    model.eval()

    all_probs = []
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x, offsets, y in val_loader:
            logits = model(x, offsets).squeeze(1)
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).long()

            all_probs.extend(probs.numpy())
            all_preds.extend(preds.numpy())
            all_targets.extend(y.numpy())

    print(classification_report(all_targets, all_preds))
    print("F1:", f1_score(all_targets, all_preds))
    print("ROC-AUC:", roc_auc_score(all_targets, all_probs))
