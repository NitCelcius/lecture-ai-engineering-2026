"""第5回 宿題: 本講義の各ページから Gemini に専門用語を抜き出させ、用語索引を作る。

    uv run --env-file .env python -m scripts.build_term_index [--out PATH] [--batch 10] [--wait 20]

入力: data/rag_data/lectures.json（RAG_DOCUMENT_IDS の資料だけ）
出力: data/rag_data/term_index.json  {"model": ..., "pages": {page_id: [用語, ...]}}
途中で止まっても、出力済みのページは飛ばして続きから再開する。
"""

import argparse
import json
import time
from pathlib import Path

from google import genai
from google.genai import types
from google.genai.errors import ServerError
from pydantic import BaseModel

from app.config import MODEL, ROOT, get_api_key
from app.retrieval import load_pages

DEFAULT_OUT = ROOT / "data/rag_data/term_index.json"

PROMPT = """\
次の講義スライドの各ページから、そのページで説明・使用されている専門用語を抜き出してください。
- 1ページあたり5〜15語。技術用語・手法名・概念名・ツール名・固有の仕組みの名前を選ぶ。
- 「ページ」「場合」「方法」「今日」のような一般語、人名、数値だけの語は入れない。
- 本文に出てくる表記をそのまま使う。英語と日本語の両方が出ていれば両方入れてよい。
- page_id は入力のものをそのまま返す。

{pages}
"""


class PageTerms(BaseModel):
    page_id: str
    terms: list[str]


class BatchTerms(BaseModel):
    pages: list[PageTerms]


def format_pages(pages: list[dict]) -> str:
    return "\n\n".join(
        f"<page id=\"{p['page_id']}\">\nタイトル: {p['title']}\n{p['content']}\n</page>" for p in pages
    )


RETRY_WAITS = (30, 60, 120)  # 503（混雑）などサーバー側の一時的なエラーで待つ秒数


def generate(client: genai.Client, **kwargs):
    for wait in RETRY_WAITS:
        try:
            return client.models.generate_content(**kwargs)
        except ServerError as e:
            print(f"サーバーエラー {e.code}。{wait} 秒待って再試行", flush=True)
            time.sleep(wait)
    return client.models.generate_content(**kwargs)


def extract(client: genai.Client, pages: list[dict]) -> dict[str, list[str]]:
    response = generate(client,
        model=MODEL,
        contents=PROMPT.format(pages=format_pages(pages)),
        config=types.GenerateContentConfig(
            temperature=0,
            response_mime_type="application/json",
            response_schema=BatchTerms,
        ),
    )
    wanted = {p["page_id"] for p in pages}
    return {item.page_id: item.terms for item in response.parsed.pages if item.page_id in wanted}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--batch", type=int, default=10)
    parser.add_argument("--wait", type=float, default=20.0, help="呼び出しの間に待つ秒数")
    args = parser.parse_args()

    index = {"model": MODEL, "pages": {}}
    if args.out.exists():
        index = json.loads(args.out.read_text(encoding="utf-8"))
    todo = [p for p in load_pages() if p["page_id"] not in index["pages"]]
    client = genai.Client(api_key=get_api_key())

    for start in range(0, len(todo), args.batch):
        batch = todo[start:start + args.batch]
        index["pages"].update(extract(client, batch))
        args.out.write_text(json.dumps(index, ensure_ascii=False, indent=1), encoding="utf-8")
        missing = [p["page_id"] for p in batch if p["page_id"] not in index["pages"]]
        print(f"{start + len(batch)}/{len(todo)} 件処理" + (f"（未取得: {missing}）" if missing else ""), flush=True)
        if start + args.batch < len(todo):
            time.sleep(args.wait)


if __name__ == "__main__":
    main()
