"""第5回 宿題: Gemini で作った用語索引（term_index.json）と、表記をまとめた結果（term_groups.json）を引く。"""

import json
import re
import unicodedata
from collections import defaultdict
from functools import cache

from app.config import ROOT

TERM_INDEX = ROOT / "data/rag_data/term_index.json"
TERM_GROUPS = ROOT / "data/rag_data/term_groups.json"
LOOKUP_LIMIT = 5        # lookup_term が返す候補の数
RARE_TERM_MAX_PAGES = 4  # 検索結果に添える用語は、出てくるページがこれ以下のものだけ（多い語は手がかりにならない）


def normalize(term: str) -> str:
    """表記の機械的なゆれ（全角半角・大文字小文字・空白や記号・語末の長音）をそろえる。"""
    s = unicodedata.normalize("NFKC", term).casefold()
    s = re.sub(r"[\s・\-_/()（）「」『』]", "", s)
    return re.sub(r"(?<=[ァ-ヶ])ー$", "", s)


@cache
def load_entries() -> dict[str, dict]:
    """正規化した表記 → {"term": 代表表記, "variants": 表記, "pages": ページID}。別表記のグループは1項目にまとめる。"""
    index = json.loads(TERM_INDEX.read_text(encoding="utf-8"))
    groups = json.loads(TERM_GROUPS.read_text(encoding="utf-8"))["groups"] if TERM_GROUPS.exists() else []

    canonical_of = {}
    for group in groups:
        for variant in group["variants"]:
            canonical_of[normalize(variant)] = normalize(group["canonical"])

    entries = defaultdict(lambda: {"term": "", "variants": [], "pages": []})
    for page_id, terms in index["pages"].items():
        for term in terms:
            entry = entries[canonical_of.get(normalize(term), normalize(term))]
            entry["term"] = entry["term"] or term
            if term not in entry["variants"]:
                entry["variants"].append(term)
            if page_id not in entry["pages"]:
                entry["pages"].append(page_id)
    for group in groups:
        key = normalize(group["canonical"])
        if key in entries:
            entries[key]["term"] = group["canonical"]

    # 別表記からも同じ項目を引けるようにする
    by_key = dict(entries)
    for key, entry in entries.items():
        for variant in entry["variants"]:
            by_key.setdefault(normalize(variant), entry)
    return by_key


def _bigrams(s: str) -> set[str]:
    return {s[i:i + 2] for i in range(len(s) - 1)} or {s}


def _score(query: str, key: str) -> float:
    """完全一致 > 部分一致 > 2文字ずつの組の重なり（Jaccard係数）。"""
    if query == key:
        return 3.0
    if query in key or key in query:
        return 2.0 + min(len(query), len(key)) / max(len(query), len(key))
    a, b = _bigrams(query), _bigrams(key)
    return len(a & b) / len(a | b)


def lookup(query: str) -> list[dict]:
    """近い用語を上位 LOOKUP_LIMIT 件、その用語が出てくるページIDと一緒に返す。"""
    q = normalize(query)
    if not q:
        raise ValueError("用語が空です。")
    best = {}
    for key, entry in load_entries().items():
        score = _score(q, key)
        if score > best.get(id(entry), (0.0, None))[0]:
            best[id(entry)] = (score, entry)
    ranked = sorted(best.values(), key=lambda item: -item[0])[:LOOKUP_LIMIT]
    return [{"term": e["term"], "variants": e["variants"], "pages": e["pages"], "match": round(s, 2)}
            for s, e in ranked if s > 0]


@cache
def _terms_by_page() -> dict[str, list[dict]]:
    result = defaultdict(list)
    seen = set()
    for entry in load_entries().values():
        if id(entry) in seen:
            continue
        seen.add(id(entry))
        for page_id in entry["pages"]:
            result[page_id].append(entry)
    return result


def related_terms(page_id: str) -> list[dict]:
    """そのページの用語のうち、ほかの少数のページにも出てくるものと、そのページIDを返す。"""
    return [
        {"term": e["term"], "also_in": [p for p in e["pages"] if p != page_id]}
        for e in _terms_by_page().get(page_id, [])
        if 1 < len(e["pages"]) <= RARE_TERM_MAX_PAGES
    ]
