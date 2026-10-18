"""Unit tests for PyTorch Arabic 1D-CNN, vocabulary, dataset, and training mechanics."""

from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.eval.metrics import RESULTS_COLUMNS, log_result
from src.models_dl import (
    ArabicCNN_1D,
    ArabicTextDataset,
    EarlyStopping,
    TextVocabulary,
    evaluate,
    train_epoch,
)


def test_text_vocabulary() -> None:
    """Test token vocabulary indexing, pad/unk handling, and padding encoding."""
    vocab = TextVocabulary()
    assert vocab.PAD_IDX == 0
    assert vocab.UNK_IDX == 1

    sample_texts = [
        "الرياضة في العالم العربي تحظى باهتمام كبير",
        "الاقتصاد والبورصة وأسعار الأسهم",
    ]
    vocab.fit(sample_texts, max_vocab_size=20, min_freq=1)
    assert len(vocab) > 2
    assert "الرياضة" in vocab.token_to_idx

    encoded = vocab.encode("الرياضة في الفضاء الخارجي", max_len=6)
    assert len(encoded) == 6
    assert encoded[0] == vocab.token_to_idx["الرياضة"]
    # Unknown tokens should map to UNK_IDX
    assert vocab.UNK_IDX in encoded


def test_arabic_text_dataset() -> None:
    """Test PyTorch dataset produces correct tensor types and shapes."""
    vocab = TextVocabulary().fit(["نص تجريبي أول", "نص تجريبي ثان"])
    dataset = ArabicTextDataset(
        texts=["نص تجريبي أول", "نص تجريبي ثان"],
        labels=[0, 1],
        vocab=vocab,
        max_len=10,
    )
    assert len(dataset) == 2

    x, y = dataset[0]
    assert isinstance(x, torch.Tensor)
    assert isinstance(y, torch.Tensor)
    assert x.shape == (10,)
    assert x.dtype == torch.long
    assert y.item() == 0


def test_arabic_cnn_forward_pass() -> None:
    """Test multi-scale 1D-CNN forward pass produces (batch_size, num_classes) logits."""
    vocab_size = 50
    model = ArabicCNN_1D(
        vocab_size=vocab_size,
        embedding_dim=16,
        num_filters=8,
        filter_sizes=(2, 3),
        num_classes=7,
        dropout=0.1,
    )

    batch_x = torch.randint(0, vocab_size, (4, 12), dtype=torch.long)
    logits = model(batch_x)
    assert logits.shape == (4, 7)
    assert not torch.isnan(logits).any()


def test_early_stopping() -> None:
    """Test early stopping triggers exactly when patience runs out without improvement."""
    es = EarlyStopping(patience=2, delta=0.01)
    dummy_model = nn.Linear(4, 2)

    es(0.90, dummy_model)
    assert es.early_stop is False
    assert es.counter == 0

    es(0.89, dummy_model)  # No improvement
    assert es.early_stop is False
    assert es.counter == 1

    es(0.89, dummy_model)  # No improvement, counter reaches 2
    assert es.early_stop is True


def test_train_and_evaluate_cycle() -> None:
    """Test one complete train epoch and evaluation pass on synthetic data."""
    vocab = TextVocabulary().fit(["كرة القدم رياضة جميلة", "البورصة والأسهم والمال"])
    dataset = ArabicTextDataset(
        texts=["كرة القدم رياضة جميلة", "البورصة والأسهم والمال"],
        labels=[0, 1],
        vocab=vocab,
        max_len=8,
    )
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

    model = ArabicCNN_1D(
        vocab_size=len(vocab),
        embedding_dim=16,
        num_filters=8,
        filter_sizes=(2, 3),
        num_classes=2,
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    device = torch.device("cpu")

    loss = train_epoch(model, dataloader, criterion, optimizer, device)
    assert isinstance(loss, float)
    assert loss > 0.0

    eval_loss, metrics, latency = evaluate(model, dataloader, criterion, device)
    assert "accuracy" in metrics
    assert "macro_f1" in metrics
    assert isinstance(latency, float)
    assert latency >= 0.0


def test_dl_results_csv_schema(tmp_path: Path) -> None:
    """Test results_dl.csv adheres strictly to the 14-column schema."""
    csv_file = tmp_path / "results_dl.csv"

    valid_record: dict[str, Any] = {
        "run_id": "ArabicCNN_1D_word_embeddings_none_light_normalized_fold0",
        "model": "ArabicCNN_1D",
        "representation": "word_embeddings",
        "dim_reduction": "none",
        "preprocessing": "light_normalized",
        "noise_level": 0.0,
        "seed": 42,
        "split": "cv_fold_0",
        "accuracy": 0.985,
        "macro_precision": 0.985,
        "macro_recall": 0.985,
        "macro_f1": 0.985,
        "train_time_sec": 45.2,
        "inference_ms_per_doc": 0.42,
    }

    log_result(csv_file, valid_record)
    assert csv_file.is_file()

    df = pd.read_csv(csv_file)
    assert list(df.columns) == RESULTS_COLUMNS
    assert len(df) == 1
    assert df.iloc[0]["model"] == "ArabicCNN_1D"
