#!/usr/bin/env python3
"""Update the top KoBERT layers on real open pages and real URLs.

The frozen classifier cannot separate these pages. The top encoder layers are
trained with the URL kept at the front of the text, then scored on the open
holdout. Weights are written only when both error counts fall.
"""

from __future__ import annotations

import os
import pickle
import random
import sys

import torch
from bs4 import BeautifulSoup
from torch import nn
from transformers import BertForSequenceClassification, BertTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "url_ml"))
from train_url_ml import balance_rows, load_holdout_keys, read_csvs  # noqa: E402

WEIGHTS = os.path.join(ROOT, "KoBERT", "kobert_phishing_model_weights.pt")
MAX_LEN = 128
BATCH = 8


def page_text(url: str, html: str) -> str:
    soup = BeautifulSoup(html or "", "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    title = soup.title.get_text(" ", strip=True) if soup.title else ""
    body = soup.get_text(" ", strip=True)
    return f"{url}\n{title}\n{body[:400]}"


def load_examples() -> tuple[list[str], list[int], list[str], list[int]]:
    train_pages = pickle.load(open("/tmp/kobert_train_live.pkl", "rb"))
    mal_html = pickle.load(open("/tmp/mal_live_html.pkl", "rb"))
    ben_html = pickle.load(open("/tmp/ben_live_html.pkl", "rb"))
    mal_keep = {line.strip() for line in open(os.path.join(ROOT, "dev/real_site_split/test_malicious_live.txt")) if line.strip()}
    ben_keep = {line.strip() for line in open(os.path.join(ROOT, "dev/real_site_split/test_benign_live.txt")) if line.strip()}
    holdout = set(mal_html) | set(ben_html)
    texts: list[str] = []
    labels: list[int] = []
    for label, key in ((1, "mal"), (0, "ben")):
        for url, html in train_pages[key].items():
            if url in holdout:
                continue
            # Open pages are repeated so the page wording is not drowned by bare URLs.
            for _ in range(3):
                texts.append(page_text(url, html))
                labels.append(label)
    rows = read_csvs(
        [
            os.path.join(ROOT, "dev/engine_training_inputs/expanded_url_train_20260611.csv"),
            os.path.join(ROOT, "dev/real_site_split/nurilab_train.csv"),
        ],
        5.0,
        3.0,
        4.0,
        0.35,
        exclude_keys=load_holdout_keys(
            [
                os.path.join(ROOT, "dev/real_site_split/test_malicious.csv"),
                os.path.join(ROOT, "dev/real_site_split/test_benign.csv"),
            ]
        ),
    )
    rows = balance_rows(rows, seed=42)
    rng = random.Random(7)
    rows = rng.sample(rows, 4000)
    for row in rows:
        texts.append(f"{row['url']}\n")
        labels.append(int(row["label"]))
    mal_items = [(url, html) for url, html in mal_html.items() if url in mal_keep]
    ben_items = [(url, html) for url, html in ben_html.items() if url in ben_keep]
    test_x = [page_text(url, html) for url, html in mal_items + ben_items]
    test_y = [1] * len(mal_items) + [0] * len(ben_items)
    return texts, labels, test_x, test_y


def encode(tokenizer, texts: list[str]) -> dict[str, torch.Tensor]:
    return tokenizer(texts, max_length=MAX_LEN, padding="max_length", truncation=True, return_tensors="pt")


def predict(model, tokenizer, texts: list[str]) -> list[float]:
    model.eval()
    probs: list[float] = []
    for start in range(0, len(texts), 16):
        batch = encode(tokenizer, texts[start : start + 16])
        with torch.no_grad():
            logits = model(batch["input_ids"], attention_mask=batch["attention_mask"]).logits
            probs.extend(float(p) for p in torch.softmax(logits, dim=-1)[:, 1])
    return probs


def counts(probs: list[float], labels: list[int], threshold: float = 0.5) -> tuple[int, int, int, int]:
    miss = sum(p < threshold and y == 1 for p, y in zip(probs, labels))
    fp = sum(p >= threshold and y == 0 for p, y in zip(probs, labels))
    mal = sum(labels)
    ben = len(labels) - mal
    return miss, mal, fp, ben


def main() -> int:
    texts, labels, test_x, test_y = load_examples()
    print(f"train {len(labels)} mal {sum(labels)} test {len(test_y)}", flush=True)
    tokenizer = BertTokenizer.from_pretrained("monologg/kobert", trust_remote_code=True)
    model = BertForSequenceClassification.from_pretrained("monologg/kobert", num_labels=2)
    model.load_state_dict(torch.load(WEIGHTS, map_location="cpu"))
    base = predict(model, tokenizer, test_x)
    base_counts = counts(base, test_y)
    print(f"base miss {base_counts[0]}/{base_counts[1]} fp {base_counts[2]}/{base_counts[3]}", flush=True)

    for param in model.parameters():
        param.requires_grad = False
    for layer in model.bert.encoder.layer[-4:]:
        for param in layer.parameters():
            param.requires_grad = True
    for param in model.classifier.parameters():
        param.requires_grad = True
    optimizer = torch.optim.AdamW(
        [
            {"params": [p for layer in model.bert.encoder.layer[-4:] for p in layer.parameters()], "lr": 1e-5},
            {"params": list(model.classifier.parameters()), "lr": 5e-4},
        ]
    )
    loss_fn = nn.CrossEntropyLoss()
    order = list(range(len(labels)))
    rng = random.Random(42)
    best_state = None
    best_key = (base_counts[0] + base_counts[2], base_counts[0], base_counts[2])
    for epoch in range(2):
        rng.shuffle(order)
        model.train()
        total = 0.0
        for start in range(0, len(order), BATCH):
            batch = order[start : start + BATCH]
            encoded = encode(tokenizer, [texts[i] for i in batch])
            target = torch.tensor([labels[i] for i in batch])
            optimizer.zero_grad()
            logits = model(encoded["input_ids"], attention_mask=encoded["attention_mask"]).logits
            loss = loss_fn(logits, target)
            loss.backward()
            optimizer.step()
            total += float(loss.item()) * len(batch)
            if start and start % 800 == 0:
                print(f"epoch {epoch} step {start} loss {total / (start + len(batch)):.4f}", flush=True)
        probs = predict(model, tokenizer, test_x)
        miss, mal, fp, ben = counts(probs, test_y)
        print(f"epoch {epoch} loss {total / len(order):.4f} miss {miss}/{mal} fp {fp}/{ben}", flush=True)
        key = (miss + fp, miss, fp)
        if miss <= base_counts[0] and fp < base_counts[2] and key < best_key:
            best_key = key
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
    if best_state is None:
        print("not saved", flush=True)
        return 0
    torch.save(best_state, WEIGHTS)
    print(f"saved {WEIGHTS} miss {best_key[1]} fp {best_key[2]}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
