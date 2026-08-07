"""재현성 검사용 배치 생성.

사람이 100건을 직접 라벨링하지 못하는 상황에서, 대신 **같은 기준으로 다시 판정**해
원래 라벨과 얼마나 일치하는지를 잰다.

  이것은 "정확도"가 아니다. 사람 정답이 없으므로 재는 것은 **판정 재현성**이다.
  일치율이 낮으면 라벨 자체가 노이즈이고, 그게 모델 성능의 상한을 설명한다.

원래 판정은 배치 40건이었다. 여기서는 25건으로 줄여 돌린다 — job-radar 가
"한 세션이 수백 건을 읽으면 뒤로 갈수록 판단이 흐려진다"고 배치를 쪼갠 판단이
실제로 근거가 있는지도 같이 보려는 것이다.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--out", default="labels/agent-batches")
    args = parser.parse_args()
    BATCH_SIZE, OUT = args.batch_size, Path(args.out)

    truth = json.loads(Path("labels/llm-labels.raw.json").read_text(encoding="utf-8"))
    df = pd.read_parquet("data/dataset.parquet").drop_duplicates(subset="id")
    sample = df[df["id"].isin(truth)].reset_index(drop=True)

    OUT.mkdir(parents=True, exist_ok=True)
    for f in OUT.glob("*.md"):
        f.unlink()

    batches = [sample.iloc[i:i + BATCH_SIZE] for i in range(0, len(sample), BATCH_SIZE)]
    for bi, batch in enumerate(batches, 1):
        lines = [
            f"# 재판정 배치 {bi}/{len(batches)} — {len(batch)}건",
            "",
            "각 공고를 **본문으로** 읽고 verdict 를 하나 고른다: `지원권장` · `검토` · `부적합`.",
            "판정 기준은 `C:\\Projects\\job-radar\\persona-consultant.md` 를 따른다.",
            "",
            "---",
            "",
        ]
        for _, row in batch.iterrows():
            lines += [
                f"## {row['id']}",
                f"- 제목: {row['title']}",
                f"- 회사: {row['company']} · 업종: {row['sector']}",
                f"- 근무지: {row['location']} · 학력: {row['education']} · 최소경력: {row['annual_from']}년 · 고용형태: {row['employment']}",
                "",
                "```",
                (row["body"] or "")[:3500],
                "```",
                "",
            ]
        (OUT / f"batch-{bi:02d}.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"배치 {len(batches)}개 생성 (건당 {BATCH_SIZE}건) → {OUT}")
    print(f"  총 {len(sample)}건 · 원래 라벨 분포 "
          f"{pd.Series([truth[i] for i in sample['id']]).value_counts().to_dict()}")


if __name__ == "__main__":
    main()
