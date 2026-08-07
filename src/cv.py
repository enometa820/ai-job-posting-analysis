"""교차검증 — 점수 하나를 구간으로 바꾼다.

단일 분할의 macro-F1 0.554 는 그 분할에서만 참이다.
5-fold × 시드 3개로 15번 재서 평균과 표준편차를 낸다.
딥러닝은 시간이 커서 시드 3개 반복만 한다(fold 는 나누지 않는다).
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline

LABELS = ["부적합", "검토", "지원권장"]
SEEDS = [42, 7, 2026]
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


def make_pipe(seed: int):
    return make_pipeline(
        TfidfVectorizer(
            max_features=50_000, ngram_range=(1, 2), min_df=2, sublinear_tf=True,
            token_pattern=r"(?u)\b[가-힣A-Za-z][가-힣A-Za-z0-9]+\b",
        ),
        LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed),
    )


def cross_validate(df: pd.DataFrame, field: str, folds: int = 5) -> list[float]:
    scores: list[float] = []
    for seed in SEEDS:
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        for train_idx, test_idx in skf.split(df, df["verdict"]):
            train, test = df.iloc[train_idx], df.iloc[test_idx]
            pipe = make_pipe(seed).fit(train[field], train["verdict"])
            pred = pipe.predict(test[field])
            scores.append(f1_score(test["verdict"], pred, average="macro",
                                   labels=LABELS, zero_division=0))
    return scores


def summarize(name: str, scores: list[float]) -> dict:
    mean = statistics.mean(scores)
    sd = statistics.stdev(scores) if len(scores) > 1 else 0.0
    print(f"  {name:22} macro-F1 {mean:.3f} ± {sd:.3f}  "
          f"(n={len(scores)} · 최소 {min(scores):.3f} · 최대 {max(scores):.3f})")
    return {"model": name, "mean": round(mean, 4), "std": round(sd, 4),
            "n": len(scores), "min": round(min(scores), 4), "max": round(max(scores), 4),
            "scores": [round(s, 4) for s in scores]}


def fig_cv(results: list[dict]) -> None:
    frame = pd.DataFrame([{"model": r["model"], "score": s}
                          for r in results for s in r["scores"]])
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    sns.boxplot(data=frame, y="model", x="score", hue="model", legend=False, ax=ax,
                palette=PALETTE[:len(results)], width=0.5, fliersize=3)
    sns.stripplot(data=frame, y="model", x="score", ax=ax, color="#333", size=3, alpha=0.55)
    for i, r in enumerate(results):
        ax.text(r["mean"], i - 0.32, f"{r['mean']:.3f} ± {r['std']:.3f}",
                ha="center", fontsize=9, weight="bold")
    ax.set_xlabel("macro-F1")
    ax.set_ylabel("")
    ax.set_title("5-fold × 시드 3개 — 점수 하나가 아니라 구간으로")
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "09-cv.png", facecolor="white")
    plt.close(fig)
    print("\n  저장 09-cv.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/dataset.parquet")
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    setup_style()
    df = pd.read_parquet(args.data).drop_duplicates(subset="id" if "id" in
                                                    pd.read_parquet(args.data).columns
                                                    else "posting_id")
    df = df[df["verdict"].isin(LABELS)].reset_index(drop=True)
    print(f"교차검증 {len(df):,}건 · {args.folds}-fold × 시드 {len(SEEDS)}개 = "
          f"{args.folds * len(SEEDS)}회\n")

    results = []
    for field, name in (("title", "A. 제목 TF-IDF"), ("body", "B. 본문 TF-IDF")):
        t0 = time.time()
        scores = cross_validate(df, field, args.folds)
        results.append(summarize(name, scores))
        print(f"     ({time.time() - t0:.1f}s)")

    gap = results[1]["mean"] - results[0]["mean"]
    pooled = (results[0]["std"] + results[1]["std"]) / 2
    print(f"\n  본문 − 제목 = +{gap:.3f} (평균 표준편차 {pooled:.3f}의 "
          f"{gap / max(pooled, 1e-9):.1f}배)")

    fig_cv(results)
    Path("reports").mkdir(exist_ok=True)
    Path("reports/cv_results.json").write_text(
        json.dumps({"seeds": SEEDS, "folds": args.folds, "n": len(df),
                    "results": results, "gap_body_minus_title": round(gap, 4)},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print("  저장 reports/cv_results.json")


if __name__ == "__main__":
    main()
