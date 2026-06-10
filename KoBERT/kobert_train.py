#!/usr/bin/env python3
"""
KoBERT 피싱 탐지 모델 Fine-tuning 스크립트
한국어 사이트 대상 (영어 전용 사이트 제외)

사용법:
  python KoBERT/kobert_train.py --input train_urls_korean.csv

입력 CSV: url, label (0=정상, 1=악성)
출력: kobert_phishing_model_weights.pt (기존은 .bak 백업)

요구사항: torch, transformers, sentencepiece, beautifulsoup4, curl_cffi
GPU 있으면 자동 사용, 없으면 CPU (느리지만 가능)
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import ssl
import sys
import time
import urllib.request
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Tuple

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_FILE = os.path.join(BASE_DIR, "kobert_phishing_model_weights.pt")
WEIGHTS_META_FILE = os.path.join(BASE_DIR, "kobert_phishing_model_weights.meta.json")

# ──────────────────────────────────────────────
# HTML → 텍스트 추출 (기존 koBERT.py 함수 재사용)
# ──────────────────────────────────────────────

def _extract_text_from_html(html: str) -> str:
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return html[:300]

    soup = BeautifulSoup(html, "html.parser")
    for noise in soup(["script", "style", "noscript", "iframe", "header", "footer", "nav"]):
        noise.decompose()

    vital_kws = ["이름", "성함", "연락처", "전화", "핸드폰", "내용", "주소", "나이", "계좌",
                 "비밀번호", "신청", "결제", "본인확인", "계정", "비정상", "차단", "안전한 사용"]
    blacklist = ["바로가기", "레이어", "새창", "건너뛰기", "닫기", "펼치기"]
    num_pat = re.compile(r"^[\d,\.%+\-\s]+$")

    texts = []
    short_count = 0
    for text in soup.stripped_strings:
        if any(b in text for b in blacklist):
            continue
        if num_pat.match(text):
            continue
        if any(kw in text for kw in vital_kws):
            texts.append(text)
        elif 1 < len(text) <= 25 and short_count < 30:
            texts.append(text)
            short_count += 1
        elif 25 < len(text) <= 500:
            texts.append(text)

    title = soup.title.string.strip() if soup.title and soup.title.string else ""
    combined = (title + " " + " ".join(texts)).strip()
    return combined[:512] if combined else ""


UA = "Mozilla/5.0 (iPhone; CPU iPhone OS 16_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.6 Mobile/15E148 Safari/604.1"

HANGUL = re.compile(r'[가-힣ᄀ-ᇿ]')


def _fetch_html(url: str, timeout: int = 10) -> str:
    # curl_cffi 우선 시도 (User-Agent 회피 가능)
    try:
        from curl_cffi import requests as curl_req
        r = curl_req.get(url, timeout=timeout, headers={"User-Agent": UA,
                                                         "Accept-Language": "ko-KR,ko;q=0.9"},
                         impersonate="chrome110", verify=False)
        return r.text
    except Exception:
        pass

    # fallback: urllib
    try:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        req = urllib.request.Request(url, headers={"User-Agent": UA,
                                                    "Accept-Language": "ko-KR,ko;q=0.9"})
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
            raw = resp.read(65536)
            return raw.decode("utf-8", errors="replace")
    except Exception:
        return ""


def fetch_text(url: str) -> Tuple[str, bool]:
    """
    Returns (text, is_korean).
    is_korean: True if Hangul detected.
    """
    html = _fetch_html(url)
    if not html:
        return "", False
    text = _extract_text_from_html(html)
    is_korean = bool(HANGUL.search(text))
    return text, is_korean


# ──────────────────────────────────────────────
# 학습 데이터 로드
# ──────────────────────────────────────────────

def load_csv(path: str) -> Tuple[List[str], List[int], List[str]]:
    urls, labels, texts = [], [], []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            u = (row.get("url") or "").strip()
            lb = str(row.get("label", "")).strip()
            if not u or lb not in ("0", "1"):
                continue
            urls.append(u)
            labels.append(int(lb))
            texts.append((row.get("text") or "").strip())
    return urls, labels, texts


def use_cached_texts(texts: List[str], labels: List[int]) -> Tuple[List[str], List[int]]:
    texts_out, labels_out = [], []
    skipped_empty = 0
    skipped_non_kr = 0
    used_fallback = 0
    for text, label in zip(texts, labels):
        if not text.strip():
            skipped_empty += 1
            continue
        if not HANGUL.search(text):
            skipped_non_kr += 1
            continue
        if "웹사이트 주소 분석 텍스트" in text:
            used_fallback += 1
        texts_out.append(text[:512])
        labels_out.append(label)
    print(
        f"  캐시 텍스트 사용: {len(texts_out)}개 "
        f"(비어있음 {skipped_empty}개 제외, 비한국어 {skipped_non_kr}개 제외, "
        f"URL fallback {used_fallback}개)"
    )
    return texts_out, labels_out


# ──────────────────────────────────────────────
# 텍스트 병렬 수집
# ──────────────────────────────────────────────

def collect_texts(urls: List[str], labels: List[int],
                  workers: int = 8) -> Tuple[List[str], List[int]]:
    print(f"  텍스트 수집: {len(urls)}개 URL, workers={workers}")
    results = [None] * len(urls)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        future_map = {pool.submit(fetch_text, url): i for i, url in enumerate(urls)}
        done = 0
        for future in as_completed(future_map):
            idx = future_map[future]
            done += 1
            try:
                text, is_korean = future.result()
                results[idx] = (text, labels[idx], is_korean)
            except Exception:
                results[idx] = ("", labels[idx], False)
            if done % 20 == 0 or done == len(urls):
                print(f"    [{done}/{len(urls)}]", flush=True)

    # 한국어 텍스트만 유지 (비어있는 것 제외)
    texts_out, labels_out = [], []
    skipped_non_kr = 0
    skipped_empty = 0
    for item in results:
        if item is None:
            continue
        text, label, is_korean = item
        if not text.strip():
            skipped_empty += 1
            continue
        if not is_korean:
            skipped_non_kr += 1
            continue
        texts_out.append(text)
        labels_out.append(label)

    print(f"  수집 완료: {len(texts_out)}개 (비어있음 {skipped_empty}개 제외, 비한국어 {skipped_non_kr}개 제외)")
    return texts_out, labels_out


# ──────────────────────────────────────────────
# 데이터셋 클래스
# ──────────────────────────────────────────────

def make_dataset(texts: List[str], labels: List[int], tokenizer, max_len: int):
    import torch
    from torch.utils.data import Dataset

    class PhishingDataset(Dataset):
        def __init__(self):
            self.encodings = tokenizer(
                texts,
                max_length=max_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            self.labels = torch.tensor(labels, dtype=torch.long)

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            return {
                "input_ids": self.encodings["input_ids"][idx],
                "attention_mask": self.encodings["attention_mask"][idx],
                "labels": self.labels[idx],
            }

    return PhishingDataset()


# ──────────────────────────────────────────────
# Fine-tuning
# ──────────────────────────────────────────────

def finetune(
    texts: List[str],
    labels: List[int],
    epochs: int = 3,
    batch_size: int = 8,
    lr: float = 2e-5,
    max_len: int = 128,
    val_ratio: float = 0.15,
    seed: int = 42,
    freeze_encoder: bool = False,
    preserve_if_worse: bool = True,
) -> None:
    import torch
    import torch.nn.functional as F
    from torch.optim import AdamW
    from torch.utils.data import DataLoader, Subset
    from transformers import BertForSequenceClassification, BertTokenizer
    from sklearn.model_selection import train_test_split

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  디바이스: {device}")
    if device.type == "cpu":
        print("  ⚠️  GPU 없음 — CPU로 학습 (느림). 에포크 수를 줄이면 시간 단축 가능.")

    print("  KoBERT 모델/토크나이저 로딩...")
    tokenizer = BertTokenizer.from_pretrained("monologg/kobert")
    model = BertForSequenceClassification.from_pretrained(
        "monologg/kobert", num_labels=2, attn_implementation="eager"
    )

    # 기존 가중치 로드 (이어서 학습)
    if os.path.isfile(WEIGHTS_FILE):
        print(f"  기존 가중치 로드: {WEIGHTS_FILE}")
        model.load_state_dict(
            torch.load(WEIGHTS_FILE, map_location=device, weights_only=False)
        )
    else:
        print("  기존 가중치 없음 — 처음부터 fine-tuning")

    if freeze_encoder:
        frozen = 0
        for name, param in model.named_parameters():
            if name.startswith("bert."):
                param.requires_grad = False
                frozen += param.numel()
        print(f"  Encoder freeze: ON (frozen_params={frozen})")

    model.to(device)

    # 데이터셋 분할
    torch.manual_seed(seed)
    dataset = make_dataset(texts, labels, tokenizer, max_len)
    n_val = max(1, int(len(dataset) * val_ratio))
    n_train = len(dataset) - n_val
    indices = list(range(len(dataset)))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=n_val,
        random_state=seed,
        stratify=labels,
    )
    train_ds = Subset(dataset, train_idx)
    val_ds = Subset(dataset, val_idx)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    print(f"  Train: {n_train}개  Val: {n_val}개  Epochs: {epochs}  LR: {lr}")

    optimizer = AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=lr,
        weight_decay=0.01,
    )

    # 클래스 불균형 보정
    import numpy as np
    label_arr = np.array(labels)
    pos = label_arr.sum()
    neg = len(label_arr) - pos
    class_weights = torch.tensor(
        [
            len(label_arr) / max(2 * neg, 1),
            len(label_arr) / max(2 * pos, 1),
        ],
        dtype=torch.float32,
    ).to(device)

    best_val_acc = 0.0
    best_weights = None
    previous_best_val_acc: float | None = None
    if preserve_if_worse and os.path.isfile(WEIGHTS_META_FILE):
        try:
            with open(WEIGHTS_META_FILE, "r", encoding="utf-8") as f:
                previous_meta = json.load(f)
            previous_best_val_acc = float(previous_meta.get("best_val_accuracy"))
        except Exception:
            previous_best_val_acc = None

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        t0 = time.perf_counter()
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            batch_labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = F.cross_entropy(outputs.logits, batch_labels, weight=class_weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            preds = outputs.logits.argmax(dim=-1)
            correct += (preds == batch_labels).sum().item()
            total += len(batch_labels)

        train_acc = correct / total
        train_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        val_tp = val_fp = val_fn = val_tn = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                batch_labels = batch["labels"].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                preds = outputs.logits.argmax(dim=-1)
                val_correct += (preds == batch_labels).sum().item()
                val_total += len(batch_labels)
                for p, t in zip(preds.cpu().numpy(), batch_labels.cpu().numpy()):
                    if p == 1 and t == 1: val_tp += 1
                    elif p == 1 and t == 0: val_fp += 1
                    elif p == 0 and t == 1: val_fn += 1
                    else: val_tn += 1

        val_acc = val_correct / val_total
        precision = val_tp / max(val_tp + val_fp, 1)
        recall = val_tp / max(val_tp + val_fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)
        elapsed = time.perf_counter() - t0

        print(f"  Epoch {epoch}/{epochs} — "
              f"Loss: {train_loss:.4f}  TrainAcc: {train_acc:.4f}  "
              f"ValAcc: {val_acc:.4f}  F1: {f1:.4f}  "
              f"P: {precision:.3f}  R: {recall:.3f}  ({elapsed:.1f}s)")
        print(f"    TP={val_tp} FP={val_fp} FN={val_fn} TN={val_tn}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_weights = {k: v.clone() for k, v in model.state_dict().items()}

    # 최고 성능 가중치 저장
    if best_weights is not None:
        model.load_state_dict(best_weights)

    print(f"\n  최고 검증 정확도: {best_val_acc:.4f}")
    if previous_best_val_acc is not None and best_val_acc < previous_best_val_acc:
        print(
            "  기존 최고 검증 정확도보다 낮아 저장 생략: "
            f"previous={previous_best_val_acc:.4f} new={best_val_acc:.4f}"
        )
        return

    # 백업 후 저장
    if os.path.isfile(WEIGHTS_FILE):
        bak = WEIGHTS_FILE + ".bak"
        shutil.copy2(WEIGHTS_FILE, bak)
        print(f"  백업: {bak}")

    torch.save(model.state_dict(), WEIGHTS_FILE)
    print(f"  저장: {WEIGHTS_FILE}")
    meta = {
        "training_rows": len(texts),
        "label_counts": {"0": int(neg), "1": int(pos)},
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "learning_rate": float(lr),
        "max_len": int(max_len),
        "val_ratio": float(val_ratio),
        "seed": int(seed),
        "freeze_encoder": bool(freeze_encoder),
        "best_val_accuracy": float(best_val_acc),
        "weights_file": os.path.basename(WEIGHTS_FILE),
    }
    with open(WEIGHTS_META_FILE, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"  메타 저장: {WEIGHTS_META_FILE}")


# ──────────────────────────────────────────────
# main
# ──────────────────────────────────────────────

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="KoBERT 피싱 탐지 fine-tuning (한국어 전용)")
    parser.add_argument("--input", required=True, help="학습 CSV (url, label 컬럼)")
    parser.add_argument("--epochs", type=int, default=3, help="에포크 수 (기본 3)")
    parser.add_argument("--batch-size", type=int, default=8, help="배치 크기 (GPU 메모리에 따라 조정)")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--max-len", type=int, default=128, help="토큰 최대 길이")
    parser.add_argument("--workers", type=int, default=8, help="텍스트 수집 병렬 수")
    parser.add_argument("--skip-fetch", action="store_true",
                        help="CSV에 text 컬럼이 있으면 직접 사용 (크롤링 건너뜀)")
    parser.add_argument("--min-training-rows", type=int, default=0,
                        help="한국어 텍스트 필터 후 학습 row가 이 값보다 작으면 실패")
    parser.add_argument("--prepare-only", action="store_true",
                        help="텍스트 로딩/필터/최소 row 검증만 수행하고 fine-tuning은 건너뜀")
    parser.add_argument("--freeze-encoder", action="store_true",
                        help="KoBERT encoder를 고정하고 classifier head만 학습")
    parser.add_argument(
        "--allow-lower-val-save",
        action="store_true",
        help="기존 메타의 best_val_accuracy보다 낮아도 새 가중치를 저장합니다.",
    )
    args = parser.parse_args(argv)

    print("=== KoBERT Fine-tuning 시작 ===")
    print(f"입력: {args.input}")

    urls, labels, cached_texts = load_csv(args.input)
    if not urls:
        print("오류: CSV에서 URL을 읽을 수 없습니다.")
        return 1
    print(f"로드: {len(urls)}개 (악성 {sum(labels)} / 정상 {labels.count(0)})")

    if args.skip_fetch:
        texts, filtered_labels = use_cached_texts(cached_texts, labels)
    else:
        texts, filtered_labels = collect_texts(urls, labels, workers=args.workers)

    if len(texts) < 10:
        print(f"오류: 한국어 텍스트가 너무 적음 ({len(texts)}개). 한국어 사이트를 더 추가하세요.")
        return 1

    mal_count = sum(filtered_labels)
    ben_count = len(filtered_labels) - mal_count
    print(f"\n학습 데이터: {len(texts)}개 (악성 {mal_count} / 정상 {ben_count})")
    if len(texts) < args.min_training_rows:
        print(f"오류: 학습 데이터 {len(texts)}개 < 최소 {args.min_training_rows}개")
        return 1

    if mal_count == 0 or ben_count == 0:
        print("오류: 악성/정상 샘플이 모두 있어야 합니다.")
        return 1

    if args.prepare_only:
        print("prepare_only=PASS")
        return 0

    finetune(
        texts, filtered_labels,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_len=args.max_len,
        freeze_encoder=args.freeze_encoder,
        preserve_if_worse=not args.allow_lower_val_save,
    )

    print("\n=== KoBERT Fine-tuning 완료 ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
