"""job-radar 산출물에서 분석·학습용 데이터셋을 만든다.

두 소스를 id로 조인한다.
  data/jobs-*.json          수집 원본 (공개 공고에서 긁은 필드 + 본문)
  out/consult-result-*.json LLM 판정 (verdict·트랙·적합도)

같은 공고가 여러 날 수집되므로 **가장 최근 판정만** 남긴다.
원문 본문은 재배포하지 않는다 — parquet 은 로컬에만 두고 공개하지 않는다.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from pathlib import Path

import pandas as pd

# 판정 JSON 의 한글 키 → 영문 컬럼. 분석 코드에서 한글 키를 계속 쓰면 깨지기 쉽다.
VERDICT_FIELDS = {
    "verdict": "verdict",
    "실제직무": "actual_role",
    "적합도": "fit_score",
    "근거": "reason",
    "트랙": "track",
    "승산": "odds",
    "어필포인트": "appeal",
    "주의": "caution",
}


def load_jobs(radar: Path) -> dict[str, dict]:
    """수집 원본. 같은 id 가 여러 날 나오면 나중 파일이 이긴다."""
    jobs: dict[str, dict] = {}
    for path in sorted(glob.glob(str(radar / "data" / "jobs-*.json"))):
        snapshot = json.load(open(path, encoding="utf-8"))
        date = snapshot.get("date")
        for row in snapshot.get("rows", []):
            key = f"{row['source']}:{row['id']}"
            row["collected_at"] = date
            jobs[key] = row
    return jobs


def load_verdicts(radar: Path) -> dict[str, dict]:
    """LLM 판정. 재평가된 공고는 마지막 판정을 쓴다."""
    verdicts: dict[str, dict] = {}
    for path in sorted(glob.glob(str(radar / "out" / "consult-result-*.json"))):
        payload = json.load(open(path, encoding="utf-8"))
        date = payload.get("date")
        for rec in payload.get("results", []):
            rec["judged_at"] = date
            verdicts[rec["id"]] = rec
    return verdicts


def normalize_location(raw: str | None) -> str:
    """지역 파싱 오염을 걷어낸다.

    사람인 목록 HTML 을 문자열로 자르다 보니 '서울 강남구 신입·' 처럼
    뒤 칸이 섞여 들어온 값이 실측 71건 있었다. 시/도 + 시군구까지만 남긴다.
    """
    if not raw:
        return "미표기"
    text = re.sub(r"\s+", " ", str(raw)).strip()
    # 뒤에 붙은 경력·고용형태 조각 제거
    text = re.split(r"\s*(?:신입|경력|경력무관|정규직|계약직|인턴|아르바이트)", text)[0]
    text = text.replace("·", "").strip()
    if not text:
        return "미표기"
    parts = text.split()
    return " ".join(parts[:2]) if len(parts) > 1 else parts[0]


def title_mentions_ai(title: str) -> bool:
    return bool(re.search(r"AI|A\.I|인공지능|자동화|automation|LLM|GPT", title or "", re.I))


def build(radar: Path, out_dir: Path) -> pd.DataFrame:
    jobs, verdicts = load_jobs(radar), load_verdicts(radar)
    keys = sorted(set(jobs) & set(verdicts))

    rows = []
    for key in keys:
        job, verdict = jobs[key], verdicts[key]
        body = job.get("body") or ""
        record = {
            "id": key,
            "source": job.get("source"),
            "title": job.get("title") or "",
            "company": job.get("company") or "",
            "sector": job.get("sector") or "미표기",
            "body": body,
            "body_len": len(body),
            "skills": job.get("skills") or "",
            "annual_from": job.get("annualFrom"),
            "annual_to": job.get("annualTo"),
            "employment": job.get("employment") or "미표기",
            "education": job.get("education") or "미표기",
            "location_raw": job.get("location") or "",
            "location": normalize_location(job.get("location")),
            "keyword": job.get("keyword") or "미표기",
            "flags": "|".join(job.get("flags") or []),
            "collected_at": job.get("collected_at"),
            "judged_at": verdict.get("judged_at"),
            "title_has_ai": title_mentions_ai(job.get("title") or ""),
        }
        for ko, en in VERDICT_FIELDS.items():
            record[en] = verdict.get(ko)
        rows.append(record)

    df = pd.DataFrame(rows)
    df["region"] = df["location"].str.split().str[0]
    df["is_offtrack"] = df["track"].eq("트랙밖")

    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_dir / "dataset.parquet", index=False)

    # 본문을 뺀 공개 가능 버전 — 집계·검증용
    df.drop(columns=["body", "reason", "appeal", "caution"]).to_csv(
        out_dir / "dataset_nobody.csv", index=False, encoding="utf-8-sig"
    )
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--radar", default=r"C:\Projects\job-radar", help="job-radar 저장소 경로")
    parser.add_argument("--out", default="data", help="출력 디렉터리")
    args = parser.parse_args()

    df = build(Path(args.radar), Path(args.out))

    print(f"조인 완료 {len(df):,}건")
    print(f"  기간      {df['collected_at'].min()} ~ {df['collected_at'].max()}")
    print(f"  출처      {df['source'].value_counts().to_dict()}")
    print(f"  verdict   {df['verdict'].value_counts().to_dict()}")
    print(f"  본문 길이  평균 {df['body_len'].mean():.0f}자 · 빈 본문 {(df['body_len'] < 50).sum()}건")
    ai = df["title_has_ai"]
    print(f"  제목 AI   {ai.sum()}건 중 트랙밖 {(ai & df['is_offtrack']).sum()}건 "
          f"({(ai & df['is_offtrack']).sum() / max(ai.sum(), 1) * 100:.0f}%)")
    print(f"  지역 정제  '{df['location_raw'].nunique()}종' → '{df['location'].nunique()}종'")


if __name__ == "__main__":
    main()
