"""EDA — 'AI 채용공고'라고 불리는 것들의 실체.

핵심 질문 하나로 전체를 꿴다.
    제목에 AI 가 붙은 공고는 실제로 AI 직무인가?

차트는 reports/figures/, 수치 요약은 reports/eda_summary.md 로 나간다.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer

FIG = Path("reports/figures")
PALETTE = ["#3a5340", "#c47651", "#8a8071", "#2f7d5a", "#b9b0a0", "#5f5a52"]


def setup_style() -> None:
    """한글이 깨지면 차트가 통째로 무의미해진다. 폰트를 먼저 고정한다."""
    for candidate in ("Malgun Gothic", "NanumGothic", "AppleGothic"):
        try:
            plt.rcParams["font.family"] = candidate
            break
        except Exception:  # noqa: BLE001
            continue
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 130
    plt.rcParams["savefig.bbox"] = "tight"
    plt.rcParams["axes.titlesize"] = 13
    plt.rcParams["axes.titleweight"] = "bold"
    sns.set_theme(style="whitegrid", palette=PALETTE, font=plt.rcParams["font.family"][0])
    sns.set_context("notebook", font_scale=0.85)


def save(fig: plt.Figure, name: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.png", facecolor="white")
    plt.close(fig)
    print(f"  저장 {name}.png")


# ────────────────────────────────────────────────────────────
# 1. 표본이 어떻게 만들어졌는가 — 이걸 먼저 밝히지 않으면 뒤가 전부 거짓이 된다
# ────────────────────────────────────────────────────────────
def fig_sample(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [1.6, 1]})

    top = df["keyword"].value_counts().head(15).sort_values()
    axes[0].barh(top.index, top.values, color=PALETTE[0])
    axes[0].set_title("무슨 검색어로 모았나 (상위 15 / 전체 53종)")
    axes[0].set_xlabel("공고 수")

    src = df.groupby(["source", "verdict"]).size().unstack(fill_value=0)
    src = src[["부적합", "검토", "지원권장"]]
    src.plot(kind="barh", stacked=True, ax=axes[1], color=[PALETTE[2], PALETTE[1], PALETTE[3]])
    axes[1].set_title("출처별 판정 분포")
    axes[1].set_xlabel("공고 수")
    axes[1].set_ylabel("")
    axes[1].legend(title="", fontsize=8)

    fig.suptitle("표본 — 일반 채용시장이 아니라 특정 검색어로 모은 공고다", y=1.02, fontsize=12)
    save(fig, "01-sample")


# ────────────────────────────────────────────────────────────
# 2. ★ 제목의 AI 와 본문의 AI — 이 분석의 결론
# ────────────────────────────────────────────────────────────
def title_filter_scores(df: pd.DataFrame) -> dict[str, float]:
    """'제목에 AI가 있으면 타깃 직무'라는 규칙을 분류기로 보고 채점한다."""
    pred = df["title_has_ai"]
    truth = ~df["is_offtrack"]
    tp = int((pred & truth).sum())
    fp = int((pred & ~truth).sum())
    fn = int((~pred & truth).sum())
    tn = int((~pred & ~truth).sum())
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": tp / max(tp + fp, 1) * 100,
        "recall": tp / max(tp + fn, 1) * 100,
    }


def fig_title_vs_reality(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1, 1.15]})
    s = title_filter_scores(df)

    ct = pd.crosstab(
        df["title_has_ai"].map({True: "제목에 AI·자동화 있음", False: "제목에 없음"}),
        df["is_offtrack"].map({True: "본문 판정: 트랙밖", False: "본문 판정: 타깃 직무"}),
    )
    sns.heatmap(ct, annot=True, fmt=",d", cmap="YlOrBr", cbar=False, ax=axes[0],
                linewidths=1.5, linecolor="white", annot_kws={"size": 13, "weight": "bold"})
    axes[0].set_title(f"제목으로 거르면 정밀도 {s['precision']:.0f}% · 재현율 {s['recall']:.0f}%")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("")
    axes[0].text(0.5, -0.22,
                 f"헛짚음 {s['fp']}건  ·  놓침 {s['fn']}건",
                 transform=axes[0].transAxes, ha="center", fontsize=10,
                 color=PALETTE[1], weight="bold")

    ai = df[df["title_has_ai"]]
    order = ai["track"].value_counts().head(6).index
    sns.countplot(data=ai[ai["track"].isin(order)], y="track", order=order,
                  color=PALETTE[0], ax=axes[1])
    axes[1].set_title(f"제목에 AI가 붙은 {len(ai)}건의 실제 직무\n(트랙밖 {ai['is_offtrack'].mean() * 100:.0f}%)")
    axes[1].set_xlabel("공고 수")
    axes[1].set_ylabel("")
    axes[1].margins(x=0.12)

    fig.subplots_adjust(wspace=0.55, bottom=0.2)
    save(fig, "02-title-vs-reality")


# ────────────────────────────────────────────────────────────
# 3. 판정이 트랙별로 어떻게 갈리는가
# ────────────────────────────────────────────────────────────
def fig_verdict(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), gridspec_kw={"width_ratios": [1, 1.3]})

    order = ["부적합", "검토", "지원권장"]
    counts = df["verdict"].value_counts().reindex(order)
    axes[0].bar(counts.index, counts.values, color=[PALETTE[2], PALETTE[1], PALETTE[3]])
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 20, f"{v:,}\n({v / len(df) * 100:.0f}%)", ha="center", fontsize=9)
    axes[0].set_title(f"판정 분포 (n={len(df):,})")
    axes[0].set_ylabel("공고 수")
    axes[0].set_ylim(0, counts.max() * 1.2)

    tracks = df["track"].value_counts().head(7).index
    ct = pd.crosstab(df[df["track"].isin(tracks)]["track"], df["verdict"])
    ct = ct.reindex(columns=order, fill_value=0)
    ct = ct.div(ct.sum(axis=1), axis=0) * 100
    ct = ct.loc[ct["지원권장"].sort_values(ascending=False).index]
    ct.plot(kind="barh", stacked=True, ax=axes[1], color=[PALETTE[2], PALETTE[1], PALETTE[3]])
    axes[1].set_title("트랙별 판정 비율 (%)")
    axes[1].set_xlabel("비율 (%)")
    axes[1].set_ylabel("")
    axes[1].legend(title="", fontsize=8, loc="lower right")

    save(fig, "03-verdict")


# ────────────────────────────────────────────────────────────
# 4. 진입 장벽 — 학력·경력·고용형태
# ────────────────────────────────────────────────────────────
def fig_barriers(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))

    edu_order = ["학력무관", "고졸", "초대졸", "대졸", "미표기"]
    ct = pd.crosstab(df["education"], df["verdict"]).reindex(edu_order).fillna(0)
    ct = ct.reindex(columns=["부적합", "검토", "지원권장"], fill_value=0)
    ct.plot(kind="bar", stacked=True, ax=axes[0], color=[PALETTE[2], PALETTE[1], PALETTE[3]])
    axes[0].set_title("학력 요건")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("공고 수")
    axes[0].tick_params(axis="x", rotation=30)
    axes[0].legend(title="", fontsize=7)

    exp = df["annual_from"].map({0: "0년(신입 가능)", 1: "1년+", 2: "2년+"}).fillna("미표기")
    sns.countplot(x=exp, order=["0년(신입 가능)", "1년+", "2년+"], color=PALETTE[0], ax=axes[1])
    axes[1].set_title("최소 경력 요건")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("")
    axes[1].tick_params(axis="x", rotation=20)

    emp = df["employment"].value_counts().head(5).sort_values()
    axes[2].barh(emp.index, emp.values, color=PALETTE[0])
    axes[2].set_title("고용 형태")
    axes[2].set_xlabel("공고 수")

    fig.suptitle("진입 장벽 — 표기된 요건 기준", y=1.02, fontsize=12)
    save(fig, "04-barriers")


# ────────────────────────────────────────────────────────────
# 5. 어디에 있는가 — 정제 전후를 함께 보여준다
# ────────────────────────────────────────────────────────────
def fig_location(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))

    top = df["location"].value_counts().head(15).sort_values()
    axes[0].barh(top.index, top.values, color=PALETTE[0])
    axes[0].set_title("근무지 상위 15 (정제 후)")
    axes[0].set_xlabel("공고 수")

    region = df["region"].value_counts().head(8).sort_values()
    axes[1].barh(region.index, region.values, color=PALETTE[3])
    axes[1].set_title("시·도 단위")
    axes[1].set_xlabel("공고 수")
    total = len(df)
    seoul = (df["region"] == "서울").sum()
    axes[1].text(0.98, 0.05, f"서울 {seoul / total * 100:.0f}%", transform=axes[1].transAxes,
                 ha="right", fontsize=11, weight="bold", color=PALETTE[1])

    save(fig, "05-location")


# ────────────────────────────────────────────────────────────
# 6. 트랙을 가르는 말 — 본문에서만 나오는 신호
# ────────────────────────────────────────────────────────────
def fig_track_terms(df: pd.DataFrame) -> None:
    tracks = [t for t in df["track"].value_counts().head(6).index if t != "트랙밖"][:4]
    tracks = ["트랙밖"] + tracks

    vec = TfidfVectorizer(max_features=4000, ngram_range=(1, 2), min_df=5,
                          token_pattern=r"(?u)\b[가-힣A-Za-z][가-힣A-Za-z0-9]+\b")
    X = vec.fit_transform(df["body"])
    terms = np.array(vec.get_feature_names_out())

    fig, axes = plt.subplots(1, len(tracks), figsize=(3.1 * len(tracks), 4.4))
    for ax, track in zip(axes, tracks):
        mask = (df["track"] == track).to_numpy()
        inside = np.asarray(X[mask].mean(axis=0)).ravel()
        outside = np.asarray(X[~mask].mean(axis=0)).ravel()
        score = inside - outside
        idx = np.argsort(score)[-10:]
        ax.barh(terms[idx], score[idx], color=PALETTE[0] if track != "트랙밖" else PALETTE[2])
        ax.set_title(track, fontsize=10)
        ax.set_xlabel("TF-IDF 차이")
        ax.tick_params(axis="y", labelsize=8)

    fig.suptitle("트랙마다 본문에서 튀는 말 — 제목에는 안 나오는 신호다", y=1.02, fontsize=12)
    save(fig, "06-track-terms")


def write_summary(df: pd.DataFrame) -> None:
    ai = df[df["title_has_ai"]]
    off = ai["is_offtrack"].sum()
    s = title_filter_scores(df)
    lines = [
        "# EDA 요약 — 자동 생성",
        "",
        f"- 표본 {len(df):,}건 · 수집 {df['collected_at'].min()} ~ {df['collected_at'].max()}",
        f"- 출처 {df['source'].value_counts().to_dict()}",
        f"- 검색어 {df['keyword'].nunique()}종 · 상위 3 = "
        + ", ".join(f"{k}({v})" for k, v in df["keyword"].value_counts().head(3).items()),
        "",
        "## 핵심 — 제목으로 거르면 안 되는 이유",
        f"- 제목에 AI·자동화가 있는 공고 **{len(ai)}건** 중 본문 판정이 트랙밖인 것 "
        f"**{off}건({off / len(ai) * 100:.0f}%)**",
        f"- 같은 규칙을 분류기로 채점하면 **정밀도 {s['precision']:.0f}% · 재현율 {s['recall']:.0f}%**",
        f"  - 헛짚음(FP) {s['fp']}건 — 제목엔 AI인데 실제로는 트랙밖",
        f"  - **놓침(FN) {s['fn']}건 — 제목엔 없는데 실제로는 타깃 직무.** 제목 필터의 진짜 손실은 이쪽이다",
        f"- 전체 트랙밖 비율 {df['is_offtrack'].mean() * 100:.0f}%",
        f"- 판정 분포 {df['verdict'].value_counts().to_dict()}",
        "",
        "## 진입 장벽",
        f"- 학력 {df['education'].value_counts().to_dict()}",
        f"- 최소 경력 0년 {(df['annual_from'] == 0).mean() * 100:.0f}%",
        f"- 서울 소재 {(df['region'] == '서울').mean() * 100:.0f}%",
        "",
        "## 데이터 품질",
        f"- 근무지 원본 {df['location_raw'].nunique()}종 → 정제 후 {df['location'].nunique()}종",
        f"- 본문 평균 {df['body_len'].mean():.0f}자 · 최소 {df['body_len'].min()} · 빈 본문 "
        f"{(df['body_len'] < 50).sum()}건",
        f"- `annual_to` 이상값(100년) {(df['annual_to'] == 100).sum()}건 — 상한 미표기의 센티널로 보인다",
    ]
    Path("reports").mkdir(exist_ok=True)
    Path("reports/eda_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("  저장 reports/eda_summary.md")


def main() -> None:
    setup_style()
    df = pd.read_parquet("data/dataset.parquet")
    print(f"EDA 시작 — {len(df):,}건")
    fig_sample(df)
    fig_title_vs_reality(df)
    fig_verdict(df)
    fig_barriers(df)
    fig_location(df)
    fig_track_terms(df)
    write_summary(df)
    print("완료")


if __name__ == "__main__":
    main()
