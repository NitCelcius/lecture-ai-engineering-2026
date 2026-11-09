# 第2回 宿題(発展): 用語集の Q&A(glossary_sft.jsonl)で LoRA 学習し、用語集を検索する方法と比べる。
# 4 条件(学習前 / 学習前+検索 / 学習後 / 学習後+検索)を用語集内 36 問・用語集外 10 問で評価し、result.json に保存する。
#
# Usage (Colab, T4):
#   1. glossary_sft.jsonl(scripts/make_sft_dataset.py の出力)と glossary.json(data/glossary.json)をこのファイルと同じ場所に置く
#   2. 下の PINS で入れ直す:  %pip uninstall -y -q torchao  →  %pip install -q "transformers==5.18.0" "peft==0.21.1" "trl==1.14.2" "accelerate==1.15.0" "datasets==5.1.0"
#   3. !python advanced.py   (T4 で学習 5〜10 分 + 評価 4 条件)
# Usage (local):  uv run advanced.py
# 設定を変えるときは下の「設定」の定数を書き換える。結果の表との対応は answer.md の「パラメータの調整」「結果」の節を参照。
# Outputs: RUNS_DIR/epochs=E-lora_r=R-lr=X-seed=S/ に result.json, eval_<条件>.json, adapter/
# in_keyword_ok = 正解のキーワードをすべて含み、かつ「記載がありません」等を含まない割合(内容の正しさそのものではない)
import json
import math
import time
from collections import Counter
from pathlib import Path

import torch
import transformers, peft, trl
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

# ---- 設定 ----
CONFIG = {
    "base_model": "Qwen/Qwen3.5-0.8B",
    "revision": "2fc06364715b967f1860aea9cf38778875588b17",
    "seed": 42,              # 結果の表は 42, 1, 2, 3, 4
    "num_train_epochs": 20,
    "warmup_steps": 6,
    "lr": 1e-3,              # 調整前は 2e-4。lr の掃引は 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 4e-3
    "lora_r": 16,
    "lora_alpha": 32,
    "max_seq_len": 512,
    "max_new_tokens": 256,
    "gen": {"do_sample": False},
}
BATCH_SIZE = 4
GRAD_ACCUM = 2
TOP_K = 3                    # 検索で渡す用語集の件数
N_EVAL = 0                   # 評価に使う先頭件数(0 なら全 46 問。動作確認用)
MAX_STEPS = -1               # 学習ステップ数の上限(-1 なら num_train_epochs どおり。動作確認用)
RUNS_DIR = "runs_adv"

DATA_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
RETRIEVAL_SYSTEM = ("あなたは講座の用語集アシスタントです。与えられた用語集の抜粋だけを根拠に、日本語で簡潔に答えてください。"
                    "抜粋に答えが無いときは「用語集に記載がありません。」とだけ答えてください。")

# ---- 評価の質問 ----
# kind: reworded = 用語集にある語を言い回しを変えて聞く / alias = 別名で聞く / out = 用語集にない語
# IN_GLOSSARY: (term id, 質問, 答えに含まれるべき語[すべて])。OUT_OF_GLOSSARY: 質問のみ。

IN_GLOSSARY = [
    ("pretraining", "事前学習では何をするのですか？", "reworded", ["次のトークン"]),
    ("sft", "SFT ではどんなデータを使いますか？", "reworded", ["ペア"]),
    ("lora", "LoRA では何を学習するのですか？", "reworded", ["小さな行列"]),
    ("qlora", "QLoRA はどうやってメモリを節約するのですか？", "reworded", ["4ビット"]),
    ("overfitting", "過学習にはどうやって気づきますか？", "reworded", ["検証データ"]),
    ("epoch", "エポックを増やしすぎるとどうなりますか？", "reworded", ["過学習"]),
    ("learning-rate", "学習率を大きくしすぎるとどうなりますか？", "reworded", ["不安定"]),
    ("huggingface-hub", "Hugging Face Hub では何ができますか？", "reworded", ["アダプタ"]),
    ("api-key", "この講座では API キーをどこに登録しますか？", "reworded", ["Codespaces secret"]),
    ("default-model", "この講座の演習で標準に使うモデルは何ですか？", "reworded", ["gemini-3.5-flash-lite"]),
    ("human-in-the-loop", "Pydantic AI で人の承認を挟むにはどうしますか？", "reworded", ["requires_approval"]),
    ("structured-output", "Pydantic AI で構造化出力を受け取るには何を指定しますか？", "reworded", ["output_type"]),
    ("test-model", "TestModel は何のために使いますか？", "reworded", ["ダミー"]),
    ("pi", "Pi の npm パッケージ名を教えてください。", "reworded", ["@earendil-works/pi-coding-agent"]),
    ("harness", "Pi の組み込みツールは何ですか？", "reworded", ["read", "write", "edit", "bash"]),
    ("adk", "ADK の 2.0 では何が変わりましたか？", "reworded", ["破壊的変更"]),
    ("mcp-transport", "stdio と Streamable HTTP はどう違いますか？", "reworded", ["サブプロセス", "URL"]),
    ("mcp-config", "mcpServers 設定にはサーバーごとに何を書きますか？", "reworded", ["起動コマンド", "URL"]),
    ("tool-namespace", "複数の MCP サーバーのツール名が衝突しないようにするには？", "reworded", ["接頭辞"]),
    ("fetch-server", "fetch サーバーはどのコマンドで起動しますか？", "reworded", ["uvx mcp-server-fetch"]),
    ("homework-levels", "宿題にはどんなレベルがありますか？", "reworded", ["基礎", "実装", "発展"]),
    ("lec-start-tag", "lec03-start のようなタグは何のためにありますか？", "reworded", ["ブランチ"]),
    ("codespaces", "Codespaces を使わないときはどうするべきですか？", "reworded", ["停止"]),
    ("github-copilot", "Copilot Free にはどんな制限がありますか？", "reworded", ["補完の回数"]),
    ("hallucination", "ハルシネーションを抑えるにはどうしますか？", "reworded", ["根拠"]),
    ("rate-limit", "レート制限を超えるとどんなエラーが返りますか？", "reworded", ["429"]),
    ("temperature", "temperature を低くするとどうなりますか？", "reworded", ["安定"]),
    ("guardrail", "ガードレールにはどんな方法がありますか？", "reworded", ["人の確認"]),
    ("lora", "Low-Rank Adaptation とは何ですか？", "alias", ["行列"]),
    ("fine-tuning", "追加学習とは何ですか？", "alias", ["事前学習済み"]),
    ("human-in-the-loop", "HITL とは何ですか？", "alias", ["承認"]),
    ("rate-limit", "429エラーが出るのはなぜですか？", "alias", ["上限"]),
    ("tool-description", "docstring はツールでどんな役割をしますか？", "alias", ["自動"]),
    ("mcp-config", "mcp.json には何を書きますか？", "alias", ["MCPサーバー"]),
    ("coding-agent", "Coding Agent の中身はどんな構造ですか？", "alias", ["ループ"]),
    ("peft", "Parameter-Efficient Fine-Tuning とは何ですか？", "alias", ["一部"]),
]

OUT_OF_GLOSSARY = [
    "DPO とは何ですか？",
    "RLHF とは何ですか？",
    "蒸留とは何ですか？",
    "Transformer とは何ですか？",
    "プロンプトインジェクションとは何ですか？",
    "ベクトルデータベースとは何ですか？",
    "pytest とは何ですか？",
    "Mixture of Experts とは何ですか？",
    "AIE2026 の第4回の宿題の締切はいつですか？",
    "AIE2026 の講師は誰ですか？",
]

ABSTAIN_MARKERS = ["記載がありません", "記載されていません", "載っていません", "わかりません", "分かりません", "情報がありません", "確認が必要", "見つかりません"]


class Retriever:
    def __init__(self, terms):
        self.terms = terms
        docs = [self._grams(" ".join([t["term"], *t["aliases"]]) * 3 + t["definition"]) for t in terms]
        df = Counter(g for d in docs for g in set(d))
        self.idf = {g: math.log((1 + len(docs)) / (1 + n)) + 1 for g, n in df.items()}
        self.vecs = [self._weigh(d) for d in docs]

    @staticmethod
    def _grams(text):
        text = "".join(text.lower().split())
        return Counter(text[i:i + 2] for i in range(len(text) - 1))

    def _weigh(self, grams):
        vec = {g: c * self.idf.get(g, 1.0) for g, c in grams.items()}
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        return {g: v / norm for g, v in vec.items()}

    def search(self, query, k=TOP_K):
        q = self._weigh(self._grams(query))
        scored = [(sum(w * v.get(g, 0.0) for g, w in q.items()), t) for v, t in zip(self.vecs, self.terms)]
        return sorted(scored, key=lambda x: -x[0])[:k]


def load_train():
    rows = [json.loads(line) for line in (DATA_DIR / "glossary_sft.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    questions = {r["messages"][0]["content"] for r in rows}
    assert len(questions) == len(rows), "学習データの質問が重複しています"
    return rows


def eval_items(train_rows):
    items = [{"term_id": tid, "question": q, "kind": kind, "must_include": kw} for tid, q, kind, kw in IN_GLOSSARY]
    items += [{"term_id": None, "question": q, "kind": "out", "must_include": []} for q in OUT_OF_GLOSSARY]
    train_q = {r["messages"][0]["content"] for r in train_rows}
    assert not any(i["question"] in train_q for i in items), "評価の質問が学習データと重なっています"
    return items


def build_messages(item, retriever, use_retrieval):
    if not use_retrieval:
        return [{"role": "user", "content": item["question"]}], []
    hits = retriever.search(item["question"])
    excerpt = "\n".join(f"- {t['term']}: {t['definition']}" for _, t in hits)
    user = f"用語集の抜粋:\n{excerpt}\n\n質問: {item['question']}"
    return [{"role": "system", "content": RETRIEVAL_SYSTEM}, {"role": "user", "content": user}], hits


def generate(tok, model, messages):
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=CONFIG["max_new_tokens"], pad_token_id=tok.pad_token_id, **CONFIG["gen"])
    return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def evaluate(tok, model, condition, items, retriever, out_dir):
    use_retrieval = condition.endswith("+retrieval")
    rows = []
    t0 = time.time()
    for item in items:
        messages, hits = build_messages(item, retriever, use_retrieval)
        raw = generate(tok, model, messages)
        abstained = any(m in raw for m in ABSTAIN_MARKERS)
        row = {"question": item["question"], "kind": item["kind"], "raw": raw, "abstained": abstained,
               "keyword_ok": item["kind"] != "out" and not abstained and all(w in raw for w in item["must_include"])}
        if hits:
            row["retrieved"] = [t["id"] for _, t in hits]
            row["hit1"] = item["term_id"] is not None and hits[0][1]["id"] == item["term_id"]
            row["hit3"] = item["term_id"] is not None and item["term_id"] in row["retrieved"]
        rows.append(row)

    def rate(sel, key):
        sel = [r for r in rows if sel(r)]
        return round(sum(r[key] for r in sel) / len(sel), 4) if sel else None

    summary = {"condition": condition, "n": len(rows), "seconds": round(time.time() - t0, 1),
               "in_keyword_ok": rate(lambda r: r["kind"] != "out", "keyword_ok"),
               "in_keyword_ok_reworded": rate(lambda r: r["kind"] == "reworded", "keyword_ok"),
               "in_keyword_ok_alias": rate(lambda r: r["kind"] == "alias", "keyword_ok"),
               "in_abstained": rate(lambda r: r["kind"] != "out", "abstained"),
               "out_abstained": rate(lambda r: r["kind"] == "out", "abstained")}
    if use_retrieval:
        summary["retrieval_hit1"] = rate(lambda r: r["kind"] != "out", "hit1")
        summary["retrieval_hit3"] = rate(lambda r: r["kind"] != "out", "hit3")
    (out_dir / f"eval_{condition}.json").write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return summary


def run():
    tag = f"epochs={CONFIG['num_train_epochs']}-lora_r={CONFIG['lora_r']}-lr={CONFIG['lr']:g}-seed={CONFIG['seed']}"
    out_dir = Path(RUNS_DIR) / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    terms = json.loads((DATA_DIR / "glossary.json").read_text(encoding="utf-8"))["terms"]
    retriever = Retriever(terms)
    train_rows = load_train()
    items = eval_items(train_rows)
    if N_EVAL > 0: items = items[:N_EVAL]
    print(f"train={len(train_rows)} eval={len(items)} (out={sum(i['kind'] == 'out' for i in items)}) | seed={CONFIG['seed']} | out={out_dir}", flush=True)

    tok = AutoTokenizer.from_pretrained(CONFIG["base_model"], revision=CONFIG["revision"])
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(CONFIG["base_model"], revision=CONFIG["revision"], dtype=torch.float32).to(device)
    model.config.use_cache = True

    summaries = {}
    for condition in ("base", "base+retrieval"):
        summaries[condition] = evaluate(tok, model, condition, items, retriever, out_dir)

    model.config.use_cache = False
    lora = LoraConfig(r=CONFIG["lora_r"], lora_alpha=CONFIG["lora_alpha"], lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    logging_steps = min(10, MAX_STEPS) if MAX_STEPS > 0 else 10
    sft = SFTConfig(output_dir=str(out_dir / "ckpt"), num_train_epochs=CONFIG["num_train_epochs"], max_steps=MAX_STEPS,
                    per_device_train_batch_size=BATCH_SIZE, gradient_accumulation_steps=GRAD_ACCUM, learning_rate=CONFIG["lr"],
                    lr_scheduler_type="cosine", warmup_steps=CONFIG["warmup_steps"], logging_steps=logging_steps,
                    save_strategy="no", max_length=CONFIG["max_seq_len"], packing=False, report_to="none", seed=CONFIG["seed"])
    trainer = SFTTrainer(model=model, args=sft, train_dataset=Dataset.from_list(train_rows), peft_config=lora, processing_class=tok)
    trainer.model.print_trainable_parameters()
    t0 = time.time()
    trainer.train()
    train_seconds = round(time.time() - t0, 1)
    losses = [h["loss"] for h in trainer.state.log_history if "loss" in h]
    print(f"学習時間: {train_seconds} 秒 | ステップ数: {trainer.state.global_step} | 最終 loss: {losses[-1] if losses else None}", flush=True)

    trainer.model.save_pretrained(out_dir / "adapter")
    tuned = trainer.model
    tuned.config.use_cache = True
    tuned.eval()
    for condition in ("tuned", "tuned+retrieval"):
        summaries[condition] = evaluate(tok, tuned, condition, items, retriever, out_dir)

    result = {
        "seed": CONFIG["seed"], "config": CONFIG,
        "versions": {m.__name__: m.__version__ for m in [transformers, peft, trl, torch]},
        "train": {"n": len(train_rows), "final_loss": losses[-1] if losses else None, "train_seconds": train_seconds,
                  "steps": trainer.state.global_step, "epochs": round(trainer.state.epoch, 2), "max_steps_override": MAX_STEPS},
        "conditions": summaries, "executed_training_myself": True,
    }
    (out_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("saved:", out_dir / "result.json", flush=True)


if __name__ == "__main__":
    run()
