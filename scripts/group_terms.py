"""第5回 宿題: term_index.json の用語から、同じ概念・ものを指す表記揺れを Gemini に探させる。

    uv run --env-file .env python -m scripts.group_terms [--out PATH] [--runs 3] [--wait 60]

入力: data/rag_data/term_index.json（scripts.build_term_index の出力）
出力: data/rag_data/term_groups.json
      {"model": ..., "runs": 実行回数, "groups": [{"canonical": 代表表記, "variants": [表記, ...], "kind": 種類, "reason": 理由}]}
normalize() で同じになる表記は、LLM に渡す前にプログラムで1つにまとめる。
LLM の結果は実行ごとにぶれるので、--runs 回実行し、過半数の実行で出た表記の組だけを残す。
表記が MAX_VARIANTS を超えるグループがあれば、保存したうえで一覧を出し、終了コード1で止まる。
"""

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Literal

from google import genai
from google.genai import types
from pydantic import BaseModel, Field

from app.config import MODEL, get_api_key
from app.term_index import TERM_GROUPS, TERM_INDEX, normalize
from scripts.build_term_index import generate

MAX_VARIANTS = 5  # これを超える表記揺れは、別の概念を混ぜている疑いが強い

PROMPT = """\
次は講義資料から抜き出した専門用語の一覧です。この中から、同じ概念・ものを指す別表記（英語と日本語、略称と正式名、書き方の違い）を探してください。

{terms}
"""


class Variation(BaseModel):
    canonical: str = Field(description="一覧の中で最も代表的な表記。一覧の表記のまま")
    variants: list[str] = Field(description="同じ概念・ものを指す表記すべて（canonical を含む）。一覧の表記のまま")
    kind: Literal["translation", "abbreviation", "spelling"] = Field(
        description="表記揺れの種類。translation=英語と日本語、abbreviation=略称と正式名、spelling=同じ言語の中の書き方の違い",
    )
    reason: str = Field(description="同じ概念・ものを指すと言える理由を1文で")


class Variations(BaseModel):
    variations: list[Variation]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=TERM_GROUPS)
    parser.add_argument("--runs", type=int, default=3, help="実行回数。過半数の実行で出た表記の組だけを残す")
    parser.add_argument("--wait", type=float, default=60.0, help="実行の間に待つ秒数")
    args = parser.parse_args()

    index = json.loads(TERM_INDEX.read_text(encoding="utf-8"))
    by_key = defaultdict(list)
    for terms in index["pages"].values():
        for term in terms:
            if term not in by_key[normalize(term)]:
                by_key[normalize(term)].append(term)
    representatives = sorted(forms[0] for forms in by_key.values())
    print(f"用語 {sum(map(len, by_key.values()))} 表記 → 正規化後 {len(representatives)} 語", flush=True)

    client = genai.Client(api_key=get_api_key())
    runs = []
    for run in range(args.runs):
        if run:
            time.sleep(args.wait)
        runs.append(find_groups(client, representatives))
        print(f"{run + 1} 回目: {len(runs[-1])} グループ", flush=True)
    groups = runs[0] if args.runs == 1 else majority(runs)
    args.out.write_text(
        json.dumps({"model": MODEL, "runs": args.runs, "groups": groups}, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    print(f"{len(groups)} グループを {args.out} に保存", flush=True)

    too_large = [g for g in groups if len(g["variants"]) > MAX_VARIANTS]
    if too_large:
        for g in too_large:
            print(f"表記が {len(g['variants'])} 個: {g['canonical']} | {' / '.join(g['variants'])}")
        sys.exit(1)


def majority(runs: list[list[dict]]) -> list[dict]:
    """過半数の実行で同じグループに入った表記の組だけを残し、つながった表記を1グループにする。"""
    votes = Counter()
    source = {}
    for groups in runs:
        pairs = set()
        for group in groups:
            for pair in combinations(sorted(set(group["variants"])), 2):
                pairs.add(pair)
                source.setdefault(pair, group)
        votes.update(pairs)
    kept = [pair for pair, count in votes.items() if count * 2 > len(runs)]

    parent = {}

    def root(term: str) -> str:
        while parent.setdefault(term, term) != term:
            term = parent[term]
        return term

    for a, b in kept:
        parent[root(a)] = root(b)
    members = defaultdict(list)
    for term in sorted(parent):
        members[root(term)].append(term)

    result = []
    for variants in members.values():
        first = next(source[p] for p in kept if p[0] in variants)
        canonical = first["canonical"] if first["canonical"] in variants else variants[0]
        result.append({"canonical": canonical, "variants": [canonical, *[v for v in variants if v != canonical]],
                       "kind": first["kind"], "reason": first["reason"]})
    return result


def find_groups(client: genai.Client, representatives: list[str]) -> list[dict]:
    """LLM に1回探させ、一覧にない表記を除いたグループを返す。"""
    response = generate(client,
        model=MODEL,
        contents=PROMPT.format(terms="\n".join(representatives)),
        config=types.GenerateContentConfig(
            temperature=0, response_mime_type="application/json", response_schema=Variations,
        ),
    )
    print(f"LLM の出力 {len(response.parsed.variations)} 件", flush=True)
    known = set(representatives)
    groups = []
    for item in response.parsed.variations:
        variants = [v for v in dict.fromkeys(item.variants) if v in known]
        if len(variants) < 2:
            print(f"一覧にない表記を除いて1語以下になったので捨てる: {item.variants}", flush=True)
            continue
        canonical = item.canonical if item.canonical in variants else variants[0]
        groups.append({"canonical": canonical, "variants": variants, "kind": item.kind, "reason": item.reason})
    return groups


if __name__ == "__main__":
    main()
