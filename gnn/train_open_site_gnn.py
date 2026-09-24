#!/usr/bin/env python3
"""Retrain GraphSAGE on real URLs with the URL-node character sketch. No page fetch."""

from __future__ import annotations

import os
import random
import sys

import torch
from torch import nn

GNN_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(GNN_DIR)
URL_ML_DIR = os.path.join(ROOT, "url_ml")
for path in (GNN_DIR, URL_ML_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

import joblib
from sklearn.feature_extraction.text import HashingVectorizer

from gnn_engine import (  # noqa: E402
    GraphSAGEPhishingNet,
    feature_map_for_url,
    graph_sample_from_feature_map,
    save_gnn_artifact,
)
import gnn_engine  # noqa: E402
from train_url_ml import balance_rows, load_holdout_keys, read_csvs  # noqa: E402


def main() -> int:
    inputs = [
        os.path.join(ROOT, "dev/engine_training_inputs/expanded_url_train_20260611.csv"),
        os.path.join(ROOT, "dev/real_site_split/nurilab_train.csv"),
    ]
    holdouts = [
        os.path.join(ROOT, "dev/real_site_split/test_malicious.csv"),
        os.path.join(ROOT, "dev/real_site_split/test_benign.csv"),
    ]
    rows = read_csvs(inputs, 5.0, 3.0, 4.0, 0.35, exclude_keys=load_holdout_keys(holdouts))
    rows = balance_rows(rows, seed=42)
    rng = random.Random(42)
    if len(rows) > 48000:
        rows = rng.sample(rows, 48000)
    vectorizer = HashingVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        n_features=509,
        alternate_sign=False,
        norm="l2",
    )
    vec_path = os.path.join(GNN_DIR, "url_node_tfidf.joblib")
    joblib.dump(vectorizer, vec_path)
    gnn_engine._URL_VECTORIZER = vectorizer
    samples = []
    labels = []
    for index, row in enumerate(rows):
        fmap = feature_map_for_url(str(row["url"]), fetch=False)
        samples.append(graph_sample_from_feature_map(fmap))
        labels.append(int(row["label"]))
        if index and index % 5000 == 0:
            print(f"built {index}", flush=True)
    xs = torch.tensor([sample.x for sample in samples], dtype=torch.float32)
    edge_index = torch.tensor(samples[0].edges, dtype=torch.long).t().contiguous()
    ys = torch.tensor(labels, dtype=torch.float32)
    net = GraphSAGEPhishingNet(xs.shape[-1])
    optimizer = torch.optim.AdamW(net.parameters(), lr=0.004, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    order = list(range(len(labels)))
    for epoch in range(8):
        rng.shuffle(order)
        net.train()
        total = 0.0
        for start in range(0, len(order), 256):
            batch = order[start : start + 256]
            optimizer.zero_grad()
            logits = []
            for idx in batch:
                logits.append(net(xs[idx], edge_index))
            loss = loss_fn(torch.stack(logits), ys[batch])
            loss.backward()
            optimizer.step()
            total += float(loss.item()) * len(batch)
        print(f"epoch {epoch} loss {total / len(order):.4f}", flush=True)
    from gnn_engine import WebStructureGNNModel

    model = WebStructureGNNModel(
        {key: value.detach().cpu() for key, value in net.state_dict().items()},
        0.5,
        {
            "kind": "web_structure_graphsage",
            "node_feature_dim": int(xs.shape[-1]),
            "hidden_dim": 48,
            "dropout": 0.12,
            "training_rows": len(labels),
            "trained_on": "real_url_no_fetch",
            "open_site_only_eval": True,
        },
    )
    out_model = os.path.join(GNN_DIR, "gnn_model_open_site.pkl")
    out_features = os.path.join(GNN_DIR, "gnn_model_open_site_features.pkl")
    save_gnn_artifact(model, out_model, out_features)
    print(f"saved {out_model} rows={len(labels)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
