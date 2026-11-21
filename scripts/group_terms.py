"""第5回 宿題: term_index.json の用語から、同じ概念・ものを指す表記揺れを Gemini に探させる。

    uv run --env-file .env python -m scripts.group_terms [--out PATH]

入力: data/rag_data/term_index.json（scripts.build_term_index の出力）
出力: data/rag_data/term_groups.json
      {"model": ..., "groups": [{"canonical": 代表表記, "variants": [表記, ...], "kind": 種類, "reason": 理由}]}
normalize() で同じになる表記は、LLM に渡す前にプログラムで1つにまとめる。
表記が MAX_VARIANTS を超えるグループがあれば、保存したうえで一覧を出し、終了コード1で止まる。
"""

import argparse
import json
import sys
from collections import defaultdict
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
次は講義資料から抜き出した専門用語の一覧です。この中から、同じ概念・ものを指す表記揺れを探してください。

表記揺れではないもの（関係があっても、指している概念・ものが違う）：
- 上位と下位、全体と部分：GitHub と Git、評価 と 評価指標、Fine-Tuning と Full Fine-Tuning
- 会社と製品、製品どうし：Anthropic と Claude、Claude と Claude Code、Google と Gemma
- 手法とその特徴や部品：QLoRA と 4bit、LoRA と 低ランク、パラメータ と 重み
- 似た手法、同じ分野の別の手法：LoRA と QLoRA、プロンプトチューニング と プレフィックスチューニング
- 連想や因果でつながるだけ：ハルシネーション と 情報漏えい、ナレッジカットオフ と 古い情報、PyTorch と pytest

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
    args.out.write_text(
        json.dumps({"model": MODEL, "groups": groups}, ensure_ascii=False, indent=1), encoding="utf-8",
    )
    print(f"{len(groups)} グループを {args.out} に保存", flush=True)

    too_large = [g for g in groups if len(g["variants"]) > MAX_VARIANTS]
    if too_large:
        for g in too_large:
            print(f"表記が {len(g['variants'])} 個: {g['canonical']} | {' / '.join(g['variants'])}")
        sys.exit(1)


if __name__ == "__main__":
    main()
