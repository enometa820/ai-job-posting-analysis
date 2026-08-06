"""verdict 3분류 — 세 모델을 같은 분할 위에서 비교한다.

    A. 제목 TF-IDF        "제목만 보고도 되는가"의 상한
    B. 본문 TF-IDF        베이스라인
    C. KLUE-RoBERTa       본문 파인튜닝

A 와 B·C 의 격차가 곧 "본문을 봐야 한다"의 실증이다.
라벨은 사람 정답이 아니라 LLM 판정이므로, 지표는 정확도가 아니라
**LLM 판정과의 일치율**로 읽어야 한다.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

SEED = 42
LABELS = ["부적합", "검토", "지원권장"]
FIG = Path("reports/figures")
PALETTE = ["#3a5340", "#c47651", "#8a8071", "#2f7d5a"]


def setup_style() -> None:
    for candidate in ("Malgun Gothic", "NanumGothic", "AppleGothic"):
        try:
            plt.rcParams["font.family"] = candidate
            break
        except Exception:  # noqa: BLE001
            continue
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 130
    plt.rcParams["savefig.bbox"] = "tight"
    sns.set_theme(style="whitegrid", font=plt.rcParams["font.family"][0])
    sns.set_context("notebook", font_scale=0.85)


def load_split(path: str = "data/dataset.parquet"):
    """공고 단위 stratified 분할. 같은 공고가 양쪽에 걸치지 않게 id 기준 유일성을 보장한다."""
    df = pd.read_parquet(path).drop_duplicates(subset="id")
    df = df[df["verdict"].isin(LABELS)].reset_index(drop=True)
    train, test = train_test_split(
        df, test_size=0.2, random_state=SEED, stratify=df["verdict"]
    )
    return train.reset_index(drop=True), test.reset_index(drop=True)


def evaluate(name: str, y_true, y_pred, elapsed: float) -> dict:
    macro = f1_score(y_true, y_pred, average="macro", labels=LABELS, zero_division=0)
    weighted = f1_score(y_true, y_pred, average="weighted", labels=LABELS, zero_division=0)
    acc = float((np.asarray(y_true) == np.asarray(y_pred)).mean())
    print(f"\n── {name}")
    print(f"   일치율 {acc * 100:.1f}% · macro-F1 {macro:.3f} · weighted-F1 {weighted:.3f} · {elapsed:.1f}s")
    print(classification_report(y_true, y_pred, labels=LABELS, zero_division=0, digits=3))
    return {"model": name, "accuracy": acc, "macro_f1": macro,
            "weighted_f1": weighted, "seconds": round(elapsed, 1)}


def run_tfidf(train, test, field: str, name: str) -> tuple[dict, np.ndarray]:
    t0 = time.time()
    pipe = make_pipeline(
        TfidfVectorizer(
            max_features=50_000, ngram_range=(1, 2), min_df=2, sublinear_tf=True,
            token_pattern=r"(?u)\b[가-힣A-Za-z][가-힣A-Za-z0-9]+\b",
        ),
        LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED),
    )
    pipe.fit(train[field], train["verdict"])
    pred = pipe.predict(test[field])
    return evaluate(name, test["verdict"], pred, time.time() - t0), pred


def run_transformer(train, test, model_name: str, epochs: int, max_len: int,
                    batch: int) -> tuple[dict, np.ndarray]:
    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n   장치 {device}" + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))
    t0 = time.time()

    tok = AutoTokenizer.from_pretrained(model_name)
    label2id = {label: i for i, label in enumerate(LABELS)}

    class Posts(Dataset):
        def __init__(self, frame: pd.DataFrame):
            self.texts = frame["body"].tolist()
            self.labels = [label2id[v] for v in frame["verdict"]]

        def __len__(self) -> int:
            return len(self.labels)

        def __getitem__(self, i: int):
            enc = tok(self.texts[i], truncation=True, max_length=max_len,
                      padding="max_length", return_tensors="pt")
            item = {k: v.squeeze(0) for k, v in enc.items()}
            item["labels"] = torch.tensor(self.labels[i])
            return item

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(LABELS)
    ).to(device)

    # 클래스 불균형(63:29:8)이 커서 가중치를 안 주면 다수 클래스로 쏠린다
    counts = train["verdict"].value_counts().reindex(LABELS).to_numpy()
    weights = torch.tensor((counts.sum() / (len(LABELS) * counts)), dtype=torch.float, device=device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)

    loader = DataLoader(Posts(train), batch_size=batch, shuffle=True, drop_last=False)
    optim = torch.optim.AdamW(model.parameters(), lr=2e-5)
    scaler = torch.amp.GradScaler(device, enabled=(device == "cuda"))
    total_steps = len(loader) * epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(optim, max_lr=2e-5, total_steps=total_steps,
                                                pct_start=0.1)

    model.train()
    step = 0
    for epoch in range(epochs):
        running = 0.0
        for bat in loader:
            bat = {k: v.to(device) for k, v in bat.items()}
            labels = bat.pop("labels")
            optim.zero_grad(set_to_none=True)
            with torch.amp.autocast(device, dtype=torch.float16, enabled=(device == "cuda")):
                out = model(**bat)
                loss = loss_fn(out.logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            sched.step()
            running += loss.item()
            step += 1
            if step % 20 == 0:
                print(f"   epoch {epoch + 1}/{epochs} step {step}/{total_steps} loss {running / 20:.4f}")
                running = 0.0

    model.eval()
    preds: list[int] = []
    with torch.no_grad():
        for bat in DataLoader(Posts(test), batch_size=batch):
            bat = {k: v.to(device) for k, v in bat.items()}
            bat.pop("labels")
            with torch.amp.autocast(device, dtype=torch.float16, enabled=(device == "cuda")):
                logits = model(**bat).logits
            preds.extend(logits.argmax(-1).cpu().tolist())

    pred = np.array([LABELS[i] for i in preds])
    return evaluate(f"C. 본문 {model_name.split('/')[-1]}", test["verdict"], pred,
                    time.time() - t0), pred


def fig_compare(results: list[dict]) -> None:
    frame = pd.DataFrame(results)
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    y = np.arange(len(frame))
    ax.barh(y - 0.2, frame["macro_f1"], height=0.38, label="macro-F1", color=PALETTE[0])
    ax.barh(y + 0.2, frame["accuracy"], height=0.38, label="일치율", color=PALETTE[1])
    ax.set_yticks(y, frame["model"])
    ax.set_xlim(0, 1)
    ax.set_xlabel("점수")
    ax.set_title("같은 분할 위에서 비교 — 제목만 보면 어디까지인가")
    ax.legend(fontsize=8, loc="lower right")
    for i, row in frame.iterrows():
        ax.text(row["macro_f1"] + 0.01, i - 0.2, f"{row['macro_f1']:.3f}", va="center", fontsize=8)
        ax.text(row["accuracy"] + 0.01, i + 0.2, f"{row['accuracy'] * 100:.1f}%", va="center", fontsize=8)
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "07-model-compare.png", facecolor="white")
    plt.close(fig)
    print("\n  저장 07-model-compare.png")


def fig_confusion(test, preds: dict[str, np.ndarray]) -> None:
    fig, axes = plt.subplots(1, len(preds), figsize=(4.2 * len(preds), 3.6))
    axes = np.atleast_1d(axes)
    for ax, (name, pred) in zip(axes, preds.items()):
        cm = confusion_matrix(test["verdict"], pred, labels=LABELS)
        sns.heatmap(cm, annot=True, fmt=",d", cmap="YlOrBr", cbar=False, ax=ax,
                    xticklabels=LABELS, yticklabels=LABELS, linewidths=1, linecolor="white")
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("예측")
        ax.set_ylabel("LLM 판정")
    fig.suptitle("혼동행렬 — 어디서 틀리는가", y=1.03, fontsize=12)
    fig.savefig(FIG / "08-confusion.png", facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("  저장 08-confusion.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-transformer", action="store_true")
    parser.add_argument("--model", default="klue/roberta-base")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--batch", type=int, default=8)
    args = parser.parse_args()

    setup_style()
    train, test = load_split()
    print(f"학습 {len(train):,} · 평가 {len(test):,}")
    print(f"학습 라벨 {train['verdict'].value_counts().to_dict()}")

    results, preds = [], {}
    res, pred = run_tfidf(train, test, "title", "A. 제목 TF-IDF")
    results.append(res); preds["A. 제목"] = pred
    res, pred = run_tfidf(train, test, "body", "B. 본문 TF-IDF")
    results.append(res); preds["B. 본문 TF-IDF"] = pred

    if not args.skip_transformer:
        res, pred = run_transformer(train, test, args.model, args.epochs,
                                    args.max_len, args.batch)
        results.append(res); preds["C. KLUE-RoBERTa"] = pred

    fig_compare(results)
    fig_confusion(test, preds)

    Path("reports").mkdir(exist_ok=True)
    Path("reports/model_results.json").write_text(
        json.dumps({"seed": SEED, "n_train": len(train), "n_test": len(test),
                    "results": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    print("  저장 reports/model_results.json")


if __name__ == "__main__":
    main()
