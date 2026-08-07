"""라벨 신뢰도 — 같은 기준으로 다시 판정하면 얼마나 같은 답이 나오는가.

이 분석 전체가 LLM 판정을 정답처럼 쓰고 있어서, 그 판정이 얼마나 흔들리는지를
재지 않으면 모든 지표가 공중에 뜬다. 여기가 그 바닥을 짚는 자리다.

두 종류의 대조를 지원한다.

  labels/human-labels.json    사람이 직접 판정한 것 → **정확도**를 말할 수 있다
  labels/agent-labels.json    같은 기준으로 다시 돌린 것 → **재현성**만 말할 수 있다

★ 둘을 같은 이름으로 부르지 않는다. 재현성이 높다고 라벨이 맞는 것은 아니다 —
  같은 편향을 두 번 반복해도 일치율은 100%가 나온다.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
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


def merge_agent_batches() -> dict:
    """서브에이전트가 배치별로 남긴 판정을 하나로 모은다."""
    merged: dict[str, str] = {}
    for path in sorted(Path("labels/agent-raw").glob("batch-*.json")):
        merged.update(json.loads(path.read_text(encoding="utf-8")))
    if merged:
        Path("labels/agent-labels.json").write_text(
            json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    return merged


def compare(second: dict, source: str) -> dict:
    baseline = json.loads(Path("labels/llm-labels.raw.json").read_text(encoding="utf-8"))
    ids = [k for k in second if k in baseline]
    if not ids:
        raise SystemExit("대조할 id 가 없다")

    a = [baseline[i] for i in ids]   # 원래 판정 (배치 40건)
    b = [second[i] for i in ids]     # 다시 판정 (배치 25건 또는 사람)
    agree = sum(x == y for x, y in zip(a, b))
    kappa = float(cohen_kappa_score(a, b, labels=LABELS))

    kind = "정확도(사람 대비)" if source == "human" else "재현성(재판정 대비)"
    print(f"\n[{kind}] 대조 {len(ids)}건")
    print(f"  일치율        {agree / len(ids) * 100:.1f}%  ({agree}/{len(ids)})")
    print(f"  Cohen's kappa {kappa:.3f}  (우연 일치 제거)")

    frame = pd.DataFrame({"base": a, "second": b})
    per_class = {
        label: round((frame[frame.base == label].second == label).mean(), 3)
        for label in LABELS if (frame.base == label).any()
    }
    print(f"  원래 라벨 기준 클래스별 일치율 {per_class}")
    print(f"  분포  원래 {pd.Series(a).value_counts().to_dict()}")
    print(f"        재판정 {pd.Series(b).value_counts().to_dict()}")

    setup_style()
    cm = confusion_matrix(a, b, labels=LABELS)
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    im = ax.imshow(cm, cmap="YlOrBr")
    ax.set_xticks(range(3), LABELS)
    ax.set_yticks(range(3), LABELS)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() * 0.6 else "#333",
                    fontsize=13, fontweight="bold")
    ax.set_xlabel("다시 판정" if source != "human" else "사람 판정")
    ax.set_ylabel("원래 LLM 판정")
    ax.set_title(f"라벨 {kind.split('(')[0]} — 일치율 {agree / len(ids) * 100:.0f}% · κ={kappa:.2f}")
    fig.colorbar(im, ax=ax, fraction=0.046)
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "10-label-agreement.png", facecolor="white")
    plt.close(fig)
    print("  저장 10-label-agreement.png")

    out = {
        "source": source, "measure": kind, "n": len(ids),
        "agreement": round(agree / len(ids), 4), "cohen_kappa": round(kappa, 4),
        "per_class_agreement": per_class,
        "dist_baseline": pd.Series(a).value_counts().to_dict(),
        "dist_second": pd.Series(b).value_counts().to_dict(),
        "confusion_rows_baseline_cols_second": cm.tolist(), "labels": LABELS,
        "note": ("사람 정답 대비 정확도" if source == "human" else
                 "같은 기준으로 다시 판정했을 때의 재현성. 정확도가 아니다 — "
                 "같은 편향을 반복해도 일치율은 높게 나온다."),
    }
    Path("reports").mkdir(exist_ok=True)
    Path("reports/label_agreement.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("  저장 reports/label_agreement.json")
    return out


def main() -> None:
    human = Path("labels/human-labels.json")
    if human.exists():
        compare(json.loads(human.read_text(encoding="utf-8")), "human")
        return
    agent = merge_agent_batches()
    if agent:
        print(f"서브에이전트 재판정 {len(agent)}건 병합")
        compare(agent, "agent")
        return
    raise SystemExit(
        "대조할 판정이 없다.\n"
        "  사람 검증  labels/label-sheet.html 을 채워 labels/human-labels.json 으로 저장\n"
        "  재현성    labels/agent-raw/batch-*.json 이 있어야 한다"
    )


if __name__ == "__main__":
    main()
