#!/usr/bin/env python3
"""Fine-tune only the KoBERT head on open pages, with the URL kept in the text."""

from __future__ import annotations

import pickle
import random

import torch
from bs4 import BeautifulSoup
from torch import nn
from transformers import BertForSequenceClassification, BertTokenizer

WEIGHTS = "KoBERT/kobert_phishing_model_weights.pt"
MAX_LEN = 128


def page_text(url: str, html: str) -> str:
    soup = BeautifulSoup(html or "", "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    title = soup.title.get_text(" ", strip=True) if soup.title else ""
    body = soup.get_text(" ", strip=True)
    return f"{url}\n{title}\n{body[:700]}"


def load_split() -> tuple[list[str], list[int], list[str], list[int]]:
    train = pickle.load(open("/tmp/kobert_train_live.pkl", "rb"))
    mal_html = pickle.load(open("/tmp/mal_live_html.pkl", "rb"))
    ben_html = pickle.load(open("/tmp/ben_live_html.pkl", "rb"))
    holdout = set(mal_html) | set(ben_html)
    train_x: list[str] = []
    train_y: list[int] = []
    for label, key in ((1, "mal"), (0, "ben")):
        rows = train[key]
        if isinstance(rows, dict):
            items = rows.items()
        else:
            items = rows
        for url, html in items:
            if url in holdout:
                continue
            train_x.append(page_text(url, html))
            train_y.append(label)
    mal_keep = {line.strip() for line in open("dev/real_site_split/test_malicious_live.txt") if line.strip()}
    ben_keep = {line.strip() for line in open("dev/real_site_split/test_benign_live.txt") if line.strip()}
    mal_items = [(url, html) for url, html in mal_html.items() if url in mal_keep]
    ben_items = [(url, html) for url, html in ben_html.items() if url in ben_keep]
    test_x = [page_text(url, html) for url, html in mal_items + ben_items]
    test_y = [1] * len(mal_items) + [0] * len(ben_items)
    return train_x, train_y, test_x, test_y


def scores(model, tokenizer, texts: list[str]) -> list[float]:
    model.eval()
    out: list[float] = []
    for start in range(0, len(texts), 16):
        batch = texts[start : start + 16]
        encoded = tokenizer(
            batch, max_length=MAX_LEN, padding="max_length", truncation=True, return_tensors="pt"
        )
        with torch.no_grad():
            logits = model(encoded["input_ids"], attention_mask=encoded["attention_mask"]).logits
            prob = torch.softmax(logits, dim=-1)[:, 1]
        out.extend(float(p) for p in prob)
    return out


def report(name: str, probs: list[float], labels: list[int]) -> tuple[int, int, int, int]:
    mal = [(p, i) for i, (p, y) in enumerate(zip(probs, labels)) if y == 1]
    ben = [(p, i) for i, (p, y) in enumerate(zip(probs, labels)) if y == 0]
    miss = sum(p < 0.5 for p, _ in mal)
    fp = sum(p >= 0.5 for p, _ in ben)
    print(f"{name} miss {miss}/{len(mal)} fp {fp}/{len(ben)}")
    return miss, len(mal), fp, len(ben)


def main() -> int:
    train_x, train_y, test_x, test_y = load_split()
    print(f"train {len(train_y)} test {len(test_y)}")
    tokenizer = BertTokenizer.from_pretrained("monologg/kobert", trust_remote_code=True)
    model = BertForSequenceClassification.from_pretrained("monologg/kobert", num_labels=2)
    state = torch.load(WEIGHTS, map_location="cpu")
    model.load_state_dict(state)
    base = scores(model, tokenizer, test_x)
    base_miss, _, base_fp, _ = report("base", base, test_y)
    for param in model.bert.parameters():
        param.requires_grad = False
    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=2e-3)
    loss_fn = nn.CrossEntropyLoss()
    order = list(range(len(train_y)))
    rng = random.Random(42)
    model.train()
    for epoch in range(2):
        rng.shuffle(order)
        total = 0.0
        for start in range(0, len(order), 16):
            batch = order[start : start + 16]
            encoded = tokenizer(
                [train_x[i] for i in batch],
                max_length=MAX_LEN,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            labels = torch.tensor([train_y[i] for i in batch])
            optimizer.zero_grad()
            logits = model(encoded["input_ids"], attention_mask=encoded["attention_mask"]).logits
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            total += float(loss.item()) * len(batch)
        print(f"epoch {epoch} loss {total / len(order):.4f}")
    tuned = scores(model, tokenizer, test_x)
    miss, _, fp, _ = report("head", tuned, test_y)
    if miss < base_miss and fp < base_fp:
        torch.save(model.state_dict(), "/tmp/kobert_open_site_head.pt")
        print("saved /tmp/kobert_open_site_head.pt")
    else:
        print("not saved")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
