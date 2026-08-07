"""1차 필터를 job-radar(Node)에 넣을 수 있는 형태로 내보낸다.

job-radar 는 "외부 의존성 0 · node: 내장만"이 원칙이라 파이썬 런타임을 붙일 수 없다.
TF-IDF + 로지스틱 회귀는 선형이라 **어휘·idf·계수만 있으면 JS 로 추론이 된다.**

경량화가 필요하다 — 5만 피처를 JSON 으로 내보내면 파일이 수십 MB 가 된다.
피처를 줄이면서 성능이 얼마나 떨어지는지 재고, 그 값을 함께 기록한다.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

LABELS = ["부적합", "검토", "지원권장"]
SEED = 42
TOKEN_RE = r"(?u)\b[가-힣A-Za-z][가-힣A-Za-z0-9]+\b"


def train(df: pd.DataFrame, max_features: int, min_df: int):
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=SEED,
                                         stratify=df["verdict"])
    vec = TfidfVectorizer(max_features=max_features, ngram_range=(1, 1), min_df=min_df,
                          sublinear_tf=True, token_pattern=TOKEN_RE)
    X = vec.fit_transform(train_df["body"])
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED)
    clf.fit(X, train_df["verdict"])
    pred = clf.predict(vec.transform(test_df["body"]))
    macro = f1_score(test_df["verdict"], pred, average="macro", labels=LABELS, zero_division=0)
    return vec, clf, macro, test_df


def main() -> None:
    df = pd.read_parquet("data/dataset.parquet").drop_duplicates(subset="id")
    df = df[df["verdict"].isin(LABELS)].reset_index(drop=True)

    print("피처 수에 따른 성능 — 어디까지 줄여도 되나")
    trials = []
    for max_features in (50_000, 20_000, 8_000, 4_000, 2_000):
        _, _, macro, _ = train(df, max_features, 2)
        trials.append({"max_features": max_features, "macro_f1": round(macro, 4)})
        print(f"  {max_features:>6,} 피처 → macro-F1 {macro:.3f}")

    CHOSEN = 8_000
    vec, clf, macro, test_df = train(df, CHOSEN, 2)
    print(f"\n채택 {CHOSEN:,} 피처 · macro-F1 {macro:.3f}")

    # 클래스 순서를 LABELS 순서로 맞춰 내보낸다 — JS 쪽에서 인덱스로 읽는다
    order = [list(clf.classes_).index(label) for label in LABELS]
    model = {
        "format": "tfidf-logreg-v1",
        "note": "sublinear_tf=True · l2 normalize · smooth_idf=True. JS 추론은 이 순서를 그대로 따라야 한다.",
        "token_pattern": "[가-힣A-Za-z][가-힣A-Za-z0-9]+",
        "classes": LABELS,
        "vocabulary": {term: int(i) for term, i in vec.vocabulary_.items()},
        "idf": [round(float(v), 6) for v in vec.idf_],
        "coef": [[round(float(v), 6) for v in clf.coef_[i]] for i in order],
        "intercept": [round(float(clf.intercept_[i]), 6) for i in order],
        "macro_f1": round(float(macro), 4),
        "n_features": len(vec.vocabulary_),
        "trained_on": len(df),
    }

    out = Path("artifacts/prefilter-model.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(model, ensure_ascii=False), encoding="utf-8")
    size_mb = out.stat().st_size / 1024 / 1024
    print(f"  저장 {out} ({size_mb:.1f}MB · 피처 {model['n_features']:,})")

    # Node 이식이 맞는지 대조할 기준 — 같은 입력에 같은 예측이 나와야 한다
    probe = test_df.sample(min(60, len(test_df)), random_state=SEED)
    proba = clf.predict_proba(vec.transform(probe["body"]))
    fixture = [
        {
            "id": row["id"],
            "body": row["body"],
            "expected": clf.classes_[int(np.argmax(p))],
            "proba": {label: round(float(p[list(clf.classes_).index(label)]), 6)
                      for label in LABELS},
        }
        for (_, row), p in zip(probe.iterrows(), proba)
    ]
    Path("artifacts/parity-fixture.json").write_text(
        json.dumps(fixture, ensure_ascii=False), encoding="utf-8")
    print(f"  저장 artifacts/parity-fixture.json ({len(fixture)}건 — Node 이식 검증용)")

    Path("reports").mkdir(exist_ok=True)
    Path("reports/export_model.json").write_text(json.dumps({
        "trials": trials, "chosen": CHOSEN, "macro_f1": round(float(macro), 4),
        "size_mb": round(size_mb, 2),
    }, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
