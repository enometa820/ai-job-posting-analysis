"""사람 라벨과 LLM 판정을 대조한다.

이 분석 전체가 LLM 판정을 정답처럼 쓰고 있어서, **그 판정이 얼마나 맞는지**를
재지 않으면 모든 지표가 공중에 뜬다. 여기가 그 바닥을 짚는 자리다.

  labels/human-labels.json   ← 본인이 시트에서 내려받은 파일
  labels/llm-labels.raw.json ← 같은 100건에 대한 LLM 판정
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import cohen_kappa_score, confusion_matrix

LABELS = ["부적합", "검토", "지원권장"]
FIG = Path("reports/figures")


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


def main() -> None:
    human_path = Path("labels/human-labels.json")
    if not human_path.exists():
        raise SystemExit(
            "labels/human-labels.json 이 없다.\n"
            "  1) labels/label-sheet.html 을 브라우저로 연다\n"
            "  2) 100건을 직접 판정한다\n"
            "  3) 'JSON 내려받기' 를 눌러 labels/human-labels.json 으로 저장한다"
        )

    human = json.loads(human_path.read_text(encoding="utf-8"))
    llm = json.loads(Path("labels/llm-labels.raw.json").read_text(encoding="utf-8"))

    ids = [k for k in human if k in llm]
    h = [human[i] for i in ids]
    m = [llm[i] for i in ids]
    agree = sum(a == b for a, b in zip(h, m))

    kappa = cohen_kappa_score(h, m, labels=LABELS)
    print(f"대조 {len(ids)}건")
    print(f"  단순 일치율   {agree / len(ids) * 100:.1f}%  ({agree}/{len(ids)})")
    print(f"  Cohen's kappa {kappa:.3f}  (우연 일치를 제거한 값)")

    # 어느 방향으로 어긋나는지 — 이게 진짜 정보다
    cm = confusion_matrix(h, m, labels=LABELS)
    setup_style()
    fig, ax = plt.subplots(figsize=(4.6, 4))
    sns.heatmap(cm, annot=True, fmt=",d", cmap="YlOrBr", cbar=False, ax=ax,
                xticklabels=LABELS, yticklabels=LABELS, linewidths=1, linecolor="white")
    ax.set_xlabel("LLM 판정")
    ax.set_ylabel("본인 판정")
    ax.set_title(f"라벨 신뢰도 — 일치율 {agree / len(ids) * 100:.0f}% · κ={kappa:.2f}")
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "10-label-agreement.png", facecolor="white")
    plt.close(fig)
    print("  저장 10-label-agreement.png")

    frame = pd.DataFrame({"human": h, "llm": m})
    by_class = (frame.groupby("human")
                .apply(lambda g: (g["human"] == g["llm"]).mean(), include_groups=False)
                .round(3).to_dict())
    print(f"  본인 판정 기준 클래스별 일치율 {by_class}")

    Path("reports").mkdir(exist_ok=True)
    Path("reports/label_agreement.json").write_text(json.dumps({
        "n": len(ids), "agreement": round(agree / len(ids), 4),
        "cohen_kappa": round(float(kappa), 4),
        "per_class_agreement": by_class,
        "confusion_human_rows_llm_cols": cm.tolist(), "labels": LABELS,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print("  저장 reports/label_agreement.json")


if __name__ == "__main__":
    main()
