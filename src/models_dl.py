"""Deep Learning 1D-CNN architecture, token vocabulary, and PyTorch dataset pipeline."""

import time
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from src.data import load_config
from src.eval.metrics import (
    compute_classification_metrics,
    log_result,
)


class TextVocabulary:
    """Deterministic token vocabulary mapping words to indices with special pad and unk tokens."""

    PAD_TOKEN = "<pad>"
    UNK_TOKEN = "<unk>"
    PAD_IDX = 0
    UNK_IDX = 1

    def __init__(self) -> None:
        self.token_to_idx: dict[str, int] = {
            self.PAD_TOKEN: self.PAD_IDX,
            self.UNK_TOKEN: self.UNK_IDX,
        }
        self.idx_to_token: dict[int, str] = {
            self.PAD_IDX: self.PAD_TOKEN,
            self.UNK_IDX: self.UNK_TOKEN,
        }

    def fit(
        self,
        texts: Sequence[str],
        max_vocab_size: int = 50000,
        min_freq: int = 2,
    ) -> "TextVocabulary":
        """Build vocabulary from training texts strictly without peeking at validation splits."""
        counter: Counter[str] = Counter()
        for text in texts:
            counter.update(text.split())

        for token, count in counter.most_common():
            if len(self.token_to_idx) >= max_vocab_size:
                break
            if count >= min_freq and token not in self.token_to_idx:
                idx = len(self.token_to_idx)
                self.token_to_idx[token] = idx
                self.idx_to_token[idx] = token

        return self

    def encode(self, text: str, max_len: int = 512) -> list[int]:
        """Convert whitespace-tokenized text into fixed-length integer sequence."""
        tokens = text.split()[:max_len]
        indices = [self.token_to_idx.get(t, self.UNK_IDX) for t in tokens]
        if len(indices) < max_len:
            indices.extend([self.PAD_IDX] * (max_len - len(indices)))
        return indices

    def __len__(self) -> int:
        return len(self.token_to_idx)


class ArabicTextDataset(Dataset):
    """PyTorch Dataset returning padded token tensor sequences and integer class labels."""

    def __init__(
        self,
        texts: Sequence[str],
        labels: Sequence[int],
        vocab: TextVocabulary,
        max_len: int = 512,
    ) -> None:
        self.labels = list(labels)
        self.vocab = vocab
        self.max_len = max_len
        self.encoded_texts = [self.vocab.encode(t, max_len=self.max_len) for t in texts]

    def __len__(self) -> int:
        return len(self.encoded_texts)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.tensor(self.encoded_texts[idx], dtype=torch.long)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y


class ArabicCNN_1D(nn.Module):
    """Multi-filter 1D-CNN capturing multi-scale n-gram features for Arabic text classification.

    Applies parallel convolutions with varying kernel sizes (e.g., 3, 5, 7) corresponding to
    sub-word morphemes, tri-consonantal roots, and phrasal contexts followed by global pooling.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        num_filters: int = 128,
        filter_sizes: Sequence[int] = (3, 5, 7),
        num_classes: int = 7,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=TextVocabulary.PAD_IDX)
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=embedding_dim,
                out_channels=num_filters,
                kernel_size=k,
                padding=k // 2,
            )
            for k in filter_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(filter_sizes), num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, seq_len)
        emb = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        emb = emb.permute(0, 2, 1)  # (batch_size, embedding_dim, seq_len)

        pooled = []
        for conv in self.convs:
            c = F.relu(conv(emb))  # (batch_size, num_filters, seq_len)
            p = F.adaptive_max_pool1d(c, output_size=1).squeeze(2)  # (batch_size, num_filters)
            pooled.append(p)

        h = torch.cat(pooled, dim=1)  # (batch_size, num_filters * len(filter_sizes))
        h = self.dropout(h)
        logits = self.fc(h)  # (batch_size, num_classes)
        return logits


class EarlyStopping:
    """Early stopping handler to terminate training when validation metric fails to improve."""

    def __init__(self, patience: int = 3, delta: float = 1e-4) -> None:
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score: float | None = None
        self.early_stop = False
        self.best_state_dict: dict[str, Any] | None = None

    def __call__(self, val_score: float, model: nn.Module) -> None:
        if self.best_score is None:
            self.best_score = val_score
            self.best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        elif val_score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_score
            self.best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            self.counter = 0


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Run one training epoch and return average cross-entropy loss."""
    model.train()
    total_loss = 0.0

    for x_batch, y_batch in dataloader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        logits = model(x_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(y_batch)

    return total_loss / len(dataloader.dataset)


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, dict[str, float], float]:
    """Evaluate model on partition and return loss, classification metrics, and latency ms/doc."""
    model.eval()
    total_loss = 0.0
    all_preds: list[int] = []
    all_targets: list[int] = []

    start_infer = time.perf_counter()
    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            total_loss += loss.item() * len(y_batch)

            preds = torch.argmax(logits, dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_targets.extend(y_batch.cpu().tolist())

    inference_duration_sec = time.perf_counter() - start_infer
    n_docs = len(dataloader.dataset)
    latency_ms = (inference_duration_sec / n_docs) * 1000.0 if n_docs > 0 else 0.0

    metrics = compute_classification_metrics(all_targets, all_preds)
    avg_loss = total_loss / n_docs if n_docs > 0 else 0.0
    return avg_loss, metrics, latency_ms


def run_dl_training(
    train_texts: Sequence[str],
    train_labels: Sequence[int],
    val_texts: Sequence[str],
    val_labels: Sequence[int],
    config: dict[str, Any] | Path | str | None = None,
    device: torch.device | None = None,
    split_name: str = "dev_val_split",
    project_root: Path | None = None,
) -> dict[str, Any]:
    """Train ArabicCNN_1D with EarlyStopping, log to results_dl.csv, and save checkpoint."""
    cfg = load_config(config) if isinstance(config, (str, Path)) or config is None else config
    root = Path.cwd() if project_root is None else Path(project_root)

    dl_cfg: dict[str, Any] = cfg.get("deep_learning", {})
    vocab_size = int(dl_cfg.get("vocab_size", 50000))
    embedding_dim = int(dl_cfg.get("embedding_dim", 128))
    num_filters = int(dl_cfg.get("num_filters", 128))
    filter_sizes = tuple(dl_cfg.get("filter_sizes", [3, 5, 7]))
    dropout = float(dl_cfg.get("dropout", 0.5))
    max_len = int(dl_cfg.get("max_len", 512))
    batch_size = int(dl_cfg.get("batch_size", 64))
    lr = float(dl_cfg.get("learning_rate", 0.001))
    epochs = int(dl_cfg.get("epochs", 15))
    patience = int(dl_cfg.get("early_stopping_patience", 3))

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Fit vocabulary strictly on training data
    vocab = TextVocabulary().fit(train_texts, max_vocab_size=vocab_size)

    train_ds = ArabicTextDataset(train_texts, train_labels, vocab, max_len=max_len)
    val_ds = ArabicTextDataset(val_texts, val_labels, vocab, max_len=max_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = ArabicCNN_1D(
        vocab_size=len(vocab),
        embedding_dim=embedding_dim,
        num_filters=num_filters,
        filter_sizes=filter_sizes,
        num_classes=len(cfg["dataset"]["categories"]),
        dropout=dropout,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    early_stopping = EarlyStopping(patience=patience)

    start_train = time.perf_counter()
    for _ in range(epochs):
        _ = train_epoch(model, train_loader, criterion, optimizer, device)
        _, val_metrics, _ = evaluate(model, val_loader, criterion, device)
        early_stopping(val_metrics["macro_f1"], model)
        if early_stopping.early_stop:
            break

    total_train_time = time.perf_counter() - start_train

    # Restore best checkpoint
    if early_stopping.best_state_dict is not None:
        model.load_state_dict(early_stopping.best_state_dict)

    _, final_metrics, infer_ms = evaluate(model, val_loader, criterion, device)

    run_id = f"ArabicCNN_1D_word_embeddings_none_light_normalized_{split_name}"
    results_csv_rel = dl_cfg.get("results_csv", "results/results_dl.csv")
    results_csv = root / results_csv_rel

    row: dict[str, Any] = {
        "run_id": run_id,
        "model": "ArabicCNN_1D",
        "representation": "word_embeddings",
        "dim_reduction": "none",
        "preprocessing": "light_normalized",
        "noise_level": 0.0,
        "seed": int(cfg["dataset"]["seed"]),
        "split": split_name,
        "accuracy": round(final_metrics["accuracy"], 6),
        "macro_precision": round(final_metrics["macro_precision"], 6),
        "macro_recall": round(final_metrics["macro_recall"], 6),
        "macro_f1": round(final_metrics["macro_f1"], 6),
        "train_time_sec": round(total_train_time, 4),
        "inference_ms_per_doc": round(infer_ms, 4),
    }

    log_result(results_csv, row)

    models_dir = root / dl_cfg.get("models_dir", "models_saved")
    models_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = models_dir / f"ArabicCNN_1D_light_normalized_{split_name}.joblib"
    joblib.dump(
        {
            "state_dict": early_stopping.best_state_dict,
            "vocab": vocab,
            "config": cfg,
            "metrics": final_metrics,
        },
        checkpoint_path,
    )

    return row
