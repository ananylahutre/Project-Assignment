"""
Comparative Study: Traditional ML vs DistilBERT for Spam Classification
Dataset required by assignment:
https://www.kaggle.com/datasets/ashfakyeafi/spam-email-classification

Run:
  pip install -r requirements.txt
  python main.py --data path/to/dataset.csv

The script detects likely text/label columns, creates stratified train/validation/test
splits, trains TF-IDF + Logistic Regression and Multinomial Naive Bayes, then fine-tunes
DistilBERT with a custom PyTorch loop (no Hugging Face Trainer API).
Outputs are written to ./outputs.
"""
import argparse
import os
import re
import time
import json
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.preprocessing import LabelEncoder
import joblib

warnings.filterwarnings("ignore", category=UserWarning)
SEED = 42
LABEL_NAMES = ["Ham", "Spam"]

def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

def normalize_text(value):
    """Conservative normalization: preserve words, numbers, and punctuation."""
    if pd.isna(value):
        return ""
    text = str(value)
    text = text.replace("\x00", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text

def find_columns(df):
    cols = list(df.columns)
    normalized = {c: re.sub(r"[^a-z0-9]", "", str(c).lower()) for c in cols}
    label_candidates = ["label", "category", "class", "target", "spam", "type", "v1"]
    text_candidates = ["text", "email", "message", "body", "content", "mail", "v2", "subject"]
    label_col = next((c for key in label_candidates for c in cols if normalized[c] == key), None)
    text_col = next((c for key in text_candidates for c in cols if normalized[c] == key and c != label_col), None)

    # Infer label column from low cardinality and text column from longest strings.
    if label_col is None:
        candidates = []
        for c in cols:
            nunique = df[c].nunique(dropna=True)
            if 2 <= nunique <= min(20, max(2, int(len(df) * 0.05))):
                candidates.append(c)
        if candidates:
            label_col = min(candidates, key=lambda c: df[c].nunique(dropna=True))
    if text_col is None:
        candidates = [c for c in cols if c != label_col and (pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_string_dtype(df[c]))]
        if candidates:
            text_col = max(candidates, key=lambda c: df[c].fillna("").astype(str).str.len().mean())
    if label_col is None or text_col is None:
        raise ValueError(f"Could not infer text and label columns from {cols}. Edit find_columns() for this CSV.")
    return text_col, label_col

def normalize_labels(series):
    raw = series.astype(str).str.strip().str.lower()
    mapping = {
        "ham": 0, "not spam": 0, "not_spam": 0, "legitimate": 0, "legit": 0,
        "0": 0, "false": 0, "spam": 1, "1": 1, "true": 1, "junk": 1
    }
    mapped = raw.map(mapping)
    if mapped.notna().all() and mapped.nunique() == 2:
        return mapped.astype(int), {0: "Ham", 1: "Spam"}
    encoder = LabelEncoder()
    encoded = encoder.fit_transform(raw)
    if len(encoder.classes_) != 2:
        raise ValueError(f"Expected exactly 2 classes (spam/ham), found {list(encoder.classes_)}")
    # If labels are numeric or named in unusual ways, report mapping explicitly.
    class_map = {int(i): str(name) for i, name in enumerate(encoder.classes_)}
    print("Label encoding:", class_map)
    return pd.Series(encoded, index=series.index), class_map

def metrics_for(y_true, y_pred):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "Recall": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "F1-score": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
    }

def save_confusion(y_true, y_pred, model_name, out_dir):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    plt.figure(figsize=(5.2, 4.2))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES)
    plt.title(f"Confusion Matrix — {model_name}")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.tight_layout()
    plt.savefig(out_dir / f"confusion_{model_name.lower().replace(' ', '_')}.png", dpi=180)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to the downloaded Kaggle CSV")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--sample-limit", type=int, default=0, help="Optional quick test limit; 0 uses all rows")
    args = parser.parse_args()

    seed_everything()
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    df = pd.read_csv(args.data, encoding_errors="replace")
    print("\nDataset preview:\n", df.head())
    print("\nShape:", df.shape)
    print("\nMissing values:\n", df.isna().sum())
    print("\nDuplicate rows:", int(df.duplicated().sum()))

    text_col, label_col = find_columns(df)
    print(f"\nSelected text column: {text_col!r}; label column: {label_col!r}")
    df = df[[text_col, label_col]].copy()
    df[text_col] = df[text_col].map(normalize_text)
    df = df.dropna(subset=[label_col])
    df = df[df[text_col].str.len() > 0]
    df = df.drop_duplicates(subset=[text_col, label_col]).reset_index(drop=True)
    y, label_map = normalize_labels(df[label_col])
    X = df[text_col].astype(str).reset_index(drop=True)
    y = y.reset_index(drop=True)
    df = pd.DataFrame({"text": X, "label": y})
    print("\nCleaned shape:", df.shape)
    print("Class counts (0/ham, 1/spam when standard labels):\n", df["label"].value_counts().sort_index())
    print("Label mapping:", label_map)
    df.to_csv(out_dir / "cleaned_dataset.csv", index=False)
    df["label"].value_counts().sort_index().rename(index={0:"Ham / class 0",1:"Spam / class 1"}).plot(kind="bar")
    plt.title("Class Distribution")
    plt.xlabel("Class")
    plt.ylabel("Number of emails")
    plt.tight_layout()
    plt.savefig(out_dir / "class_distribution.png", dpi=180)
    plt.close()

    if args.sample_limit and len(df) > args.sample_limit:
        # Stratified sampling preserves class proportions for quick debugging only.
        df, _ = train_test_split(df, train_size=args.sample_limit, random_state=SEED, stratify=df["label"])
        df = df.reset_index(drop=True)
        X, y = df["text"], df["label"]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=SEED, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=SEED, stratify=y_temp
    )
    print(f"\nSplit sizes — train: {len(X_train)}, validation: {len(X_val)}, test: {len(X_test)}")

    # Traditional pipeline: vectorizer is fitted only on training data to prevent leakage.
    models = {
        "Logistic Regression": Pipeline([
            ("tfidf", TfidfVectorizer(lowercase=True, strip_accents="unicode",
                                      ngram_range=(1, 2), min_df=2, max_features=100000,
                                      sublinear_tf=True)),
            ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED))
        ]),
        "Naive Bayes": Pipeline([
            ("tfidf", TfidfVectorizer(lowercase=True, strip_accents="unicode",
                                      ngram_range=(1, 2), min_df=2, max_features=100000,
                                      sublinear_tf=True)),
            ("classifier", MultinomialNB(alpha=0.5))
        ])
    }
    results, predictions, timings = [], {}, {}
    for name, model in models.items():
        start = time.perf_counter()
        model.fit(X_train, y_train)
        train_seconds = time.perf_counter() - start
        start = time.perf_counter()
        pred = model.predict(X_test)
        inference_seconds = time.perf_counter() - start
        timings[name] = {"training_seconds": train_seconds, "inference_seconds": inference_seconds}
        predictions[name] = pred
        row = {"Model": name, **metrics_for(y_test, pred),
               "Training time (seconds)": train_seconds,
               "Inference time (seconds)": inference_seconds}
        results.append(row)
        save_confusion(y_test, pred, name, out_dir)
        joblib.dump(model, out_dir / (name.lower().replace(" ", "_") + ".joblib"))
        print(f"\n{name}\n", classification_report(y_test, pred, target_names=LABEL_NAMES, zero_division=0))

    # DistilBERT fine-tuning using a custom PyTorch loop.
    try:
        import torch
        from torch.utils.data import Dataset, DataLoader
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
    except ImportError as e:
        raise SystemExit("Install requirements.txt before running DistilBERT. Missing dependency: " + str(e))

    model_id = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    class EmailDataset(Dataset):
        def __init__(self, texts, labels):
            self.texts = list(texts)
            self.labels = list(map(int, labels))
        def __len__(self):
            return len(self.texts)
        def __getitem__(self, idx):
            encoded = tokenizer(
                str(self.texts[idx]), truncation=True, padding="max_length",
                max_length=args.max_length, return_tensors="pt"
            )
            item = {k: v.squeeze(0) for k, v in encoded.items()}
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
            return item

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\nDistilBERT device:", device)
    train_loader = DataLoader(EmailDataset(X_train, y_train), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(EmailDataset(X_val, y_val), batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(EmailDataset(X_test, y_test), batch_size=args.batch_size, shuffle=False)
    bert = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=2)
    # Fine-tune all layers; low learning rate is a common stable starting point.
    bert.to(device)
    optimizer = torch.optim.AdamW(bert.parameters(), lr=args.learning_rate, weight_decay=0.01)
    history = {"train_loss": [], "val_loss": [], "val_f1": []}
    best_f1, best_state = -1.0, None
    bert_train_start = time.perf_counter()
    for epoch in range(args.epochs):
        bert.train()
        total_loss, batches = 0.0, 0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            outputs = bert(**batch)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(bert.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += float(loss.item())
            batches += 1
        train_loss = total_loss / max(batches, 1)

        bert.eval()
        val_losses, val_true, val_pred = [], [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = bert(**batch)
                val_losses.append(float(outputs.loss.item()))
                val_pred.extend(outputs.logits.argmax(dim=1).cpu().numpy().tolist())
                val_true.extend(batch["labels"].cpu().numpy().tolist())
        val_loss = float(np.mean(val_losses)) if val_losses else 0.0
        val_f1 = f1_score(val_true, val_pred, pos_label=1, zero_division=0)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_f1"].append(val_f1)
        print(f"Epoch {epoch+1}/{args.epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_spam_F1={val_f1:.4f}")
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.detach().cpu().clone() for k, v in bert.state_dict().items()}

    bert_training_seconds = time.perf_counter() - bert_train_start
    if best_state is not None:
        bert.load_state_dict(best_state)
    bert.eval()
    test_true, test_pred = [], []
    infer_start = time.perf_counter()
    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = bert(**batch)
            test_pred.extend(outputs.logits.argmax(dim=1).cpu().numpy().tolist())
            test_true.extend(batch["labels"].cpu().numpy().tolist())
    bert_inference_seconds = time.perf_counter() - infer_start
    predictions["DistilBERT"] = np.array(test_pred)
    results.append({"Model": "DistilBERT", **metrics_for(test_true, test_pred),
                    "Training time (seconds)": bert_training_seconds,
                    "Inference time (seconds)": bert_inference_seconds})
    save_confusion(test_true, test_pred, "DistilBERT", out_dir)
    os.makedirs(out_dir / "distilbert_model", exist_ok=True)
    bert.save_pretrained(out_dir / "distilbert_model")
    tokenizer.save_pretrained(out_dir / "distilbert_model")

    pd.DataFrame(history).to_csv(out_dir / "distilbert_training_history.csv", index=False)
    plt.figure(figsize=(7, 4.5))
    epochs = range(1, len(history["train_loss"]) + 1)
    plt.plot(epochs, history["train_loss"], marker="o", label="Training loss")
    plt.plot(epochs, history["val_loss"], marker="o", label="Validation loss")
    plt.title("DistilBERT Training Behaviour")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "distilbert_training_curves.png", dpi=180)
    plt.close()

    results_df = pd.DataFrame(results)
    results_df.to_csv(out_dir / "model_comparison.csv", index=False)
    metrics_long = results_df.melt(id_vars="Model", value_vars=["Accuracy", "Precision", "Recall", "F1-score"],
                                   var_name="Metric", value_name="Score")
    plt.figure(figsize=(9, 5))
    sns.barplot(data=metrics_long, x="Metric", y="Score", hue="Model")
    plt.ylim(0, 1.05)
    plt.title("Spam Classification: Model Metric Comparison")
    plt.ylabel("Score")
    plt.tight_layout()
    plt.savefig(out_dir / "metric_comparison.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 4.5))
    sns.barplot(data=results_df, x="Model", y="Training time (seconds)")
    plt.title("Computational Cost: Training Time")
    plt.ylabel("Seconds")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(out_dir / "training_time_comparison.png", dpi=180)
    plt.close()

    summary = {
        "dataset_rows_after_cleaning": int(len(df)),
        "text_column": str(text_col), "label_column": str(label_col),
        "label_mapping": label_map,
        "split_sizes": {"train": int(len(X_train)), "validation": int(len(X_val)), "test": int(len(X_test))},
        "device": str(device), "epochs": int(args.epochs),
        "batch_size": int(args.batch_size), "max_length": int(args.max_length),
        "learning_rate": float(args.learning_rate),
        "models": results
    }
    with open(out_dir / "experiment_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("\nFINAL TEST METRICS\n", results_df.to_string(index=False))
    print(f"\nAll outputs saved to: {out_dir.resolve()}")

if __name__ == "__main__":
    main()
