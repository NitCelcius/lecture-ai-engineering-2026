# 第2回 宿題(発展): 用語集の Q&A(glossary_sft.jsonl)で LoRA 学習し、用語集の検索と比べる。
# Usage:
#   uv run advanced.py run [--seed N] [--epochs N] [--lora-r N] [--out DIR] [--n-eval N] [--max-steps N]
# Inputs (同じフォルダ): glossary_sft.jsonl(scripts/make_sft_dataset.py の出力)、glossary.json(data/glossary.json のコピー)
# Outputs (--out, default runs_adv/epochs=E-lora_r=R-seed=S): result.json, eval_<条件>.json, adapter/
# in_keyword_ok = 正解のキーワードをすべて含み、かつ「記載がありません」等を含まない割合(内容の正しさそのものではない)
import argparse
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

from basic import CONFIG
from data_glossary_eval import ABSTAIN_MARKERS, IN_GLOSSARY, OUT_OF_GLOSSARY

HERE = Path(__file__).resolve().parent
MAX_NEW_TOKENS = 256
TOP_K = 3
RETRIEVAL_SYSTEM = ("あなたは講座の用語集アシスタントです。与えられた用語集の抜粋だけを根拠に、日本語で簡潔に答えてください。"
                    "抜粋に答えが無いときは「用語集に記載がありません。」とだけ答えてください。")
CONDITIONS = ("base", "base+retrieval", "tuned", "tuned+retrieval")


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
    rows = [json.loads(line) for line in (HERE / "glossary_sft.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    questions = {r["messages"][0]["content"] for r in rows}
    assert len(questions) == len(rows), "学習データの質問が重複しています"
    return rows


def eval_items():
    items = [{"term_id": tid, "question": q, "kind": kind, "must_include": kw} for tid, q, kind, kw in IN_GLOSSARY]
    items += [{"term_id": None, "question": q, "kind": "out", "must_include": []} for q in OUT_OF_GLOSSARY]
    train_q = {r["messages"][0]["content"] for r in load_train()}
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
        out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, pad_token_id=tok.pad_token_id, **CONFIG["gen"])
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


def run(args):
    config = dict(CONFIG)
    config["max_new_tokens"] = MAX_NEW_TOKENS
    config["seed"] = args.seed
    if args.epochs is not None: config["num_train_epochs"] = args.epochs
    if args.lora_r is not None: config["lora_r"] = args.lora_r
    if args.lr is not None: config["lr"] = args.lr
    tag = f"epochs={config['num_train_epochs']}-lora_r={config['lora_r']}" + (f"-lr={args.lr:g}" if args.lr is not None else "") + f"-seed={args.seed}"
    out_dir = Path(args.out or f"runs_adv/{tag}")
    out_dir.mkdir(parents=True, exist_ok=True)

    terms = json.loads((HERE / "glossary.json").read_text(encoding="utf-8"))["terms"]
    retriever = Retriever(terms)
    train_rows = load_train()
    items = eval_items()
    if args.n_eval > 0: items = items[:args.n_eval]
    print(f"train={len(train_rows)} eval={len(items)} (out={sum(i['kind'] == 'out' for i in items)}) | seed={args.seed} | out={out_dir}", flush=True)

    tok = AutoTokenizer.from_pretrained(config["base_model"], revision=config["revision"])
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(config["base_model"], revision=config["revision"], dtype=torch.float32).to(device)
    model.config.use_cache = True

    summaries = {}
    for condition in ("base", "base+retrieval"):
        summaries[condition] = evaluate(tok, model, condition, items, retriever, out_dir)

    model.config.use_cache = False
    lora = LoraConfig(r=config["lora_r"], lora_alpha=config["lora_alpha"], lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    logging_steps = min(10, args.max_steps) if args.max_steps > 0 else 10
    sft = SFTConfig(output_dir=str(out_dir / "ckpt"), num_train_epochs=config["num_train_epochs"], max_steps=args.max_steps,
                    per_device_train_batch_size=4, gradient_accumulation_steps=2, learning_rate=config["lr"],
                    lr_scheduler_type="cosine", warmup_steps=config["warmup_steps"], logging_steps=logging_steps,
                    save_strategy="no", max_length=config["max_seq_len"], packing=False, report_to="none", seed=config["seed"])
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
        "seed": args.seed, "config": config,
        "versions": {m.__name__: m.__version__ for m in [transformers, peft, trl, torch]},
        "train": {"n": len(train_rows), "final_loss": losses[-1] if losses else None, "train_seconds": train_seconds,
                  "steps": trainer.state.global_step, "epochs": round(trainer.state.epoch, 2), "max_steps_override": args.max_steps},
        "conditions": summaries, "executed_training_myself": True,
    }
    (out_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("saved:", out_dir / "result.json", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p_run = sub.add_parser("run", help="4 条件(学習前/学習後 x 検索なし/あり)を評価する")
    p_run.add_argument("--seed", type=int, default=CONFIG["seed"], help="乱数 seed(既定 42)")
    p_run.add_argument("--epochs", type=int, help="num_train_epochs(既定 20)")
    p_run.add_argument("--lora-r", type=int, help="LoRA の次元(既定 16)")
    p_run.add_argument("--lr", type=float, help="学習率(既定 2e-4)")
    p_run.add_argument("--out", help="出力先(既定 runs_adv/epochs=E-lora_r=R-seed=S)")
    p_run.add_argument("--n-eval", type=int, default=0, help="評価に使う先頭件数(0 なら全問。smoke test 用)")
    p_run.add_argument("--max-steps", type=int, default=-1, help="学習ステップ数の上限(-1 なら epochs どおり。smoke test 用)")
    p_run.set_defaults(func=run)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
