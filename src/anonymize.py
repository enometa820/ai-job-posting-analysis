"""공개용 데이터셋 — 개인정보를 지우고 회사를 익명화한다.

그대로 올릴 수 없는 이유가 실측으로 확인됐다.
    본문에 전화번호 486건(22.4%) · 이메일 17건
    '부적합' 라벨이 붙는 실명 회사 1,086곳

본문 텍스트 자체는 남긴다 — 없애면 EDA 도 모델도 재현이 안 된다.
지우는 것은 **사람에게 닿는 값**과 **회사 식별자**뿐이다.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pandas as pd

# 마스킹 규칙. 순서가 중요하다 — 이메일을 먼저 지워야 전화 패턴이 이메일 조각을 먹지 않는다.
SIDO = ("서울|부산|대구|인천|광주|대전|울산|세종|경기|강원|충북|충남|전북|전남|경북|경남|제주")

PATTERNS: list[tuple[str, str]] = [
    (r"[\w.\-]+@[\w\-]+\.[a-zA-Z]{2,}", "<EMAIL>"),
    (r"https?://\S*", "<URL>"),  # 사람인 기업정보의 빈 'http://' 까지 잡으려면 \S* 여야 한다
    (r"(?<!\d)01[016-9][-. ]?\d{3,4}[-. ]?\d{4}(?!\d)", "<PHONE>"),
    (r"(?<!\d)0\d{1,2}[-. ]?\d{3,4}[-. ]?\d{4}(?!\d)", "<PHONE>"),
    (r"(?:오픈)?카카오톡?\s*(?:아이디|ID|채팅)?\s*[:：]?\s*[\w.\-]{3,}", "<KAKAO>"),
    (r"(?:사업자등록번호|사업자번호)\s*[:：]?\s*\d{3}-?\d{2}-?\d{5}", "<BIZNO>"),
    # 회사명을 해시해도 주소가 남으면 역추적된다. 사람인 기업정보 블록은 형태가 일정해 블록째 지운다.
    (r"기업주소\s*[^\n]{0,90}?(?=\s*(?:채용정보|기업리뷰|홈페이지|기업형태|사원수|$))",
     "기업주소 <ADDRESS>"),
    (rf"(?:{SIDO})\s*[가-힣]{{1,6}}(?:시|군|구)\s*[가-힣0-9]{{1,12}}(?:로|길|동|읍|면)\s*[\d\-,]+[^\n,]{{0,25}}",
     "<ADDRESS>"),
]

LEAK_CHECKS: list[tuple[str, str]] = [
    ("이메일", r"[\w.\-]+@[\w\-]+\.[a-zA-Z]{2,}"),
    ("전화번호", r"(?<!\d)0\d{1,2}[-. ]?\d{3,4}[-. ]?\d{4}(?!\d)"),
    ("URL", r"https?://"),
    ("도로명주소", rf"(?:{SIDO})\s*[가-힣]{{1,6}}(?:시|군|구)\s*[가-힣0-9]{{1,12}}(?:로|길)\s*\d"),
]


def mask(text: str) -> str:
    if not isinstance(text, str):
        return ""
    for pattern, token in PATTERNS:
        text = re.sub(pattern, token, text)
    return text


def hash_company(name: str, salt: str = "ai-job-posting-analysis") -> str:
    """같은 회사는 같은 id 로 묶이되 원래 이름은 복원되지 않게."""
    if not name:
        return "company_unknown"
    digest = hashlib.sha256(f"{salt}:{name}".encode()).hexdigest()[:8]
    return f"company_{digest}"


def build_public(src: str = "data/dataset.parquet",
                 out: str = "data_public/dataset_public.parquet") -> pd.DataFrame:
    df = pd.read_parquet(src)
    pub = df.copy()

    pub["body"] = pub["body"].map(mask)
    pub["title"] = pub["title"].map(mask)
    for col in ("reason", "appeal", "caution", "odds", "actual_role"):
        if col in pub.columns:
            pub[col] = pub[col].map(mask)

    pub["company_id"] = pub["company"].map(hash_company)
    pub["posting_id"] = pub["id"].map(lambda x: hashlib.sha256(x.encode()).hexdigest()[:10])

    # 원본 식별자는 통째로 뺀다 — 해시가 있으니 그룹 분석은 그대로 된다
    pub = pub.drop(columns=[c for c in ("company", "id", "location_raw") if c in pub.columns])
    pub = pub[["posting_id", "company_id"] + [c for c in pub.columns
                                              if c not in ("posting_id", "company_id")]]

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pub.to_parquet(out_path, index=False)
    pub.drop(columns=["body"]).to_csv(
        out_path.with_name("dataset_public_meta.csv"), index=False, encoding="utf-8-sig"
    )
    return pub


def verify(pub: pd.DataFrame) -> bool:
    """마스킹이 실제로 됐는지 다시 센다. 통과 못 하면 공개하지 않는다."""
    text = pd.concat([pub["body"], pub["title"]]).astype(str)
    clean = True
    print("\n[유출 검사]")
    for name, pattern in LEAK_CHECKS:
        hits = text.str.contains(pattern, regex=True, na=False).sum()
        mark = "OK" if hits == 0 else "실패"
        print(f"  {name:8} 잔여 {hits:>4}건  {mark}")
        clean &= hits == 0
    return clean


def main() -> None:
    before = pd.read_parquet("data/dataset.parquet")
    b_phone = before["body"].str.contains(r"(?<!\d)0\d{1,2}[-. ]?\d{3,4}[-. ]?\d{4}(?!\d)",
                                          regex=True, na=False).sum()
    b_mail = before["body"].str.contains(r"[\w.\-]+@[\w\-]+\.[a-zA-Z]{2,}",
                                         regex=True, na=False).sum()
    print(f"익명화 전 — 전화 {b_phone}건 · 이메일 {b_mail}건 · 실명 회사 {before['company'].nunique():,}개")

    pub = build_public()
    print(f"익명화 후 — {len(pub):,}행 · 회사 {pub['company_id'].nunique():,}개(해시)")
    print(f"  컬럼 {len(pub.columns)}개 · 본문 평균 {pub['body'].str.len().mean():.0f}자")

    if verify(pub):
        print("\n공개 가능 — data_public/ 에 저장했다")
    else:
        raise SystemExit("마스킹 실패 — 공개하지 않는다")


if __name__ == "__main__":
    main()
