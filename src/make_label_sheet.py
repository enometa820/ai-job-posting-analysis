"""사람이 직접 채울 라벨링 시트를 만든다.

**이 100건은 반드시 본인이 판정해야 한다.**
내가(또는 다른 모델이) 채우면 그건 사람 검증이 아니라 또 하나의 AI 판정이고,
"LLM 라벨이 얼마나 맞는가"라는 질문에 아무 답도 되지 않는다.

시트는 판정을 브라우저 localStorage 에 담고, 끝나면 JSON 을 내려준다.
그 JSON 을 labels/ 에 두면 src/label_agreement.py 가 일치율을 낸다.
"""

from __future__ import annotations

import html
import json
from pathlib import Path

import pandas as pd

LABELS = ["부적합", "검토", "지원권장"]
N_PER_CLASS = {"부적합": 40, "검토": 40, "지원권장": 20}
SEED = 42


def build(src: str = "data/dataset.parquet", out: str = "labels/label-sheet.html") -> None:
    df = pd.read_parquet(src).drop_duplicates(subset="id")
    df = df[df["verdict"].isin(LABELS)]

    picks = [df[df["verdict"] == label].sample(n, random_state=SEED)
             for label, n in N_PER_CLASS.items()]
    sample = pd.concat(picks).sample(frac=1, random_state=SEED).reset_index(drop=True)

    cards = []
    for i, row in sample.iterrows():
        body = html.escape(row["body"][:2600])
        cards.append(f"""
    <article class="card" data-id="{html.escape(row['id'])}" data-idx="{i}">
      <header>
        <span class="num">{i + 1} / {len(sample)}</span>
        <h2>{html.escape(row['title'])}</h2>
        <p class="meta">{html.escape(row['company'])} · {html.escape(str(row['sector']))}
           · {html.escape(str(row['location']))} · 학력 {html.escape(str(row['education']))}</p>
      </header>
      <div class="body">{body}</div>
      <div class="pick">
        <button data-v="부적합">부적합</button>
        <button data-v="검토">검토</button>
        <button data-v="지원권장">지원권장</button>
        <span class="done"></span>
      </div>
    </article>""")

    truth = {row["id"]: row["verdict"] for _, row in sample.iterrows()}

    page = f"""<!DOCTYPE html>
<html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>라벨링 시트 — {len(sample)}건</title>
<style>
 :root{{--line:#e6ddcf;--ink:#232323;--soft:#5f5a52;--sage:#3a5340;--peach:#c47651}}
 *{{box-sizing:border-box;margin:0;padding:0}}
 body{{font-family:'Pretendard',-apple-system,system-ui,sans-serif;background:#fdfcf9;
   color:var(--ink);line-height:1.65;padding:24px 16px 120px;max-width:860px;margin:0 auto}}
 h1{{font-size:22px;margin-bottom:6px}}
 .lead{{font-size:14px;color:var(--soft);margin-bottom:6px}}
 .warn{{font-size:13px;color:var(--peach);font-weight:700;margin-bottom:20px}}
 .card{{background:#fff;border:1px solid var(--line);border-radius:14px;padding:18px 20px;
   margin-bottom:18px}}
 .card.done{{opacity:.5}}
 .num{{font-size:11px;color:var(--soft);font-family:ui-monospace,monospace}}
 h2{{font-size:16px;margin:4px 0 4px}}
 .meta{{font-size:12px;color:var(--soft);margin-bottom:12px}}
 .body{{font-size:13px;color:#3d3a35;white-space:pre-wrap;max-height:230px;overflow:auto;
   background:#faf8f3;border:1px solid var(--line);border-radius:9px;padding:12px 14px}}
 .pick{{display:flex;gap:8px;align-items:center;margin-top:14px}}
 button{{font:inherit;font-size:13px;font-weight:700;padding:8px 16px;border-radius:9px;
   border:1px solid var(--line);background:#fff;cursor:pointer}}
 button:hover{{border-color:var(--sage)}}
 button.on{{background:var(--sage);color:#fff;border-color:var(--sage)}}
 .done{{font-size:12px;color:var(--sage);font-weight:700}}
 .bar{{position:fixed;left:0;right:0;bottom:0;background:#fff;border-top:1px solid var(--line);
   padding:12px 16px;display:flex;gap:14px;align-items:center;justify-content:center}}
 .bar b{{font-variant-numeric:tabular-nums}}
 .bar button{{background:var(--sage);color:#fff;border-color:var(--sage)}}
</style></head><body>
<h1>라벨링 시트 — 공고 {len(sample)}건</h1>
<p class="lead">본문만 읽고 <b>이 사람이 지원할 자리인가</b>를 판정합니다. LLM이 뭐라고 했는지는 보이지 않습니다.</p>
<p class="warn">※ 반드시 본인이 채워야 합니다. AI가 채우면 "LLM 라벨이 맞는가"라는 질문 자체가 무의미해집니다.</p>
{''.join(cards)}
<div class="bar">
  <span>판정 <b id="cnt">0</b> / {len(sample)}</span>
  <button id="save">JSON 내려받기</button>
  <button id="reset">초기화</button>
</div>
<script>
const KEY='label-sheet-v1';
const store=JSON.parse(localStorage.getItem(KEY)||'{{}}');
const cnt=document.getElementById('cnt');
function paint(){{
  document.querySelectorAll('.card').forEach(c=>{{
    const v=store[c.dataset.id];
    c.classList.toggle('done',!!v);
    c.querySelector('.done').textContent=v?('선택: '+v):'';
    c.querySelectorAll('button').forEach(b=>b.classList.toggle('on',b.dataset.v===v));
  }});
  cnt.textContent=Object.keys(store).length;
}}
document.querySelectorAll('.card button').forEach(b=>b.onclick=()=>{{
  const card=b.closest('.card');
  store[card.dataset.id]=b.dataset.v;
  localStorage.setItem(KEY,JSON.stringify(store));
  paint();
  const next=card.nextElementSibling;
  if(next&&next.classList.contains('card')) next.scrollIntoView({{behavior:'smooth',block:'start'}});
}});
document.getElementById('save').onclick=()=>{{
  const blob=new Blob([JSON.stringify(store,null,2)],{{type:'application/json'}});
  const a=document.createElement('a');
  a.href=URL.createObjectURL(blob); a.download='human-labels.json'; a.click();
}};
document.getElementById('reset').onclick=()=>{{
  if(confirm('판정을 전부 지웁니다.')){{ localStorage.removeItem(KEY); location.reload(); }}
}};
paint();
</script></body></html>"""

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(page, encoding="utf-8")

    # 정답(LLM 판정)은 시트에 넣지 않는다 — 보이면 사람이 끌려간다
    Path("labels/llm-labels.raw.json").write_text(
        json.dumps(truth, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"시트 생성 {out_path} — {len(sample)}건")
    print(f"  구성 {sample['verdict'].value_counts().to_dict()}")
    print(f"  LLM 판정은 labels/llm-labels.raw.json 에 따로 뒀다(시트에는 안 보인다)")


if __name__ == "__main__":
    build()
