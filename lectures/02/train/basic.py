# 第2回 宿題(基礎): lora_r / lr / num_train_epochs のどれか 1 つを変えて再学習し、元の設定と比べる。
# Usage:
#   uv run basic.py run [--lora-r N | --lr X | --epochs N] [--seed N] [--data course|hitorajio|hitorajio_more] [--out DIR] [--n-test N] [--max-steps N]
#   uv run basic.py compare runs/baseline/result.json runs/<variant>/result.json
# Outputs (--out, default runs/<tag>): result.json, eval_before.json, eval_after.json, adapter/, ckpt/
import argparse
import importlib
import json
import re
import time
from pathlib import Path

import torch
import transformers, peft, trl
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

CONFIG = {
    "base_model": "Qwen/Qwen3.5-0.8B",
    "revision": "2fc06364715b967f1860aea9cf38778875588b17",
    "seed": 42,
    "num_train_epochs": 20,
    "warmup_steps": 6,
    "lr": 2e-4,
    "lora_r": 16,
    "lora_alpha": 32,
    "max_seq_len": 512,
    "max_new_tokens": 96,
    "gen": {"do_sample": False},
}
TUNABLE = ("num_train_epochs", "lr", "lora_r")
DATA_MODULES = {"course": "data_course", "hitorajio": "data_hitorajio", "hitorajio_more": "data_hitorajio_more"}


def make_example(data, question: str, answer: str, needs_review: bool) -> dict:
    target = json.dumps({"answer": answer, "needs_review": needs_review}, ensure_ascii=False)
    return {"question": question, "faq": data.FAQ_TEXT, "system": data.SYSTEM, "target": target, "needs_review": needs_review}


def build_datasets(data):
    train, valid, test, gold = [], [], [], {}
    for key, (templates, must_include) in data.GROUNDED.items():
        n = data.GROUNDED_TEMPLATES_IN_TRAIN
        for q in templates[:n]: train.append(make_example(data, q, data.FAQ[key], False))
        for q in templates[n:n + 1]: valid.append(make_example(data, q, data.FAQ[key], False))
        for q in templates[n + 1:n + 2] + data.TEST_EXTRA_GROUNDED[key]:
            test.append(make_example(data, q, data.FAQ[key], False))
            gold[q] = {"needs_review": False, "must_include": must_include}
    for i, q in enumerate(data.UNGROUNDED):
        ex = make_example(data, q, data.UNGROUNDED_ANSWER, True)
        bucket = [train, train, valid, test][i % 4]
        bucket.append(ex)
        if bucket is test: gold[q] = {"needs_review": True, "must_include": ["確認"]}
    for q in data.TEST_EXTRA_UNGROUNDED:
        test.append(make_example(data, q, data.UNGROUNDED_ANSWER, True))
        gold[q] = {"needs_review": True, "must_include": ["確認"]}
    for q in data.TRAIN_EXTRA_UNGROUNDED:
        train.append(make_example(data, q, data.UNGROUNDED_ANSWER, True))

    all_questions = [r["question"] for r in train + valid + test]
    assert len(set(all_questions)) == len(all_questions), "質問が train / valid / test で重複しています"
    return train, valid, test, gold


def build_messages(example):
    user = f"FAQ:\n{example['faq']}\n\n質問: {example['question']}"
    return [{"role": "system", "content": example["system"]}, {"role": "user", "content": user}]


def generate(tok, model, example, config):
    prompt = tok.apply_chat_template(build_messages(example), tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=config["max_new_tokens"], pad_token_id=tok.pad_token_id, **config["gen"])
    return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def check_format(text: str):
    try:
        obj = json.loads(text)
    except Exception as e:
        return False, f"json_error: {type(e).__name__}", None
    if not isinstance(obj, dict): return False, "not_object", None
    if set(obj) - {"answer", "needs_review"}: return False, "extra_keys", obj
    if "answer" not in obj or "needs_review" not in obj: return False, "missing_key", obj
    if not isinstance(obj["answer"], str) or not isinstance(obj["needs_review"], bool): return False, "wrong_type", obj
    return True, "", obj


def check_content(obj, gold_entry):
    if obj is None: return False
    if obj.get("needs_review") != gold_entry["needs_review"]: return False
    return all(w in obj.get("answer", "") for w in gold_entry["must_include"])


def evaluate(tok, model, label, test, gold, config, out_dir):
    rows = []
    t0 = time.time()
    for ex in test:
        raw = generate(tok, model, ex, config)
        ok_fmt, reason, obj = check_format(raw)
        loose = obj
        if not ok_fmt:
            m = re.search(r"\{.*\}", raw, re.S)
            try: loose = json.loads(m.group(0)) if m else None
            except Exception: loose = None
        ok_content = check_content(loose if isinstance(loose, dict) else None, gold[ex["question"]])
        rows.append({"question": ex["question"], "raw": raw, "format_ok": ok_fmt, "format_reason": reason,
                     "content_ok": ok_content, "both_ok": ok_fmt and ok_content,
                     "gold_needs_review": gold[ex["question"]]["needs_review"]})
    n = len(rows)
    summary = {"label": label, "n": n,
               "format_rate": sum(r["format_ok"] for r in rows) / n,
               "content_rate": sum(r["content_ok"] for r in rows) / n,
               "both_rate": sum(r["both_ok"] for r in rows) / n,
               "seconds": round(time.time() - t0, 1)}
    (out_dir / f"eval_{label}.json").write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return summary, rows


def classify(raw):
    ok, _, obj = check_format(raw)
    if not ok or obj is None: return "format_error"
    return "needs_review" if obj["needs_review"] else "asserted"


def run(args):
    config = dict(CONFIG)
    changed = {"num_train_epochs": args.epochs, "lr": args.lr, "lora_r": args.lora_r}
    changed = {k: v for k, v in changed.items() if v is not None}
    assert len(changed) <= 1, f"基礎は {TUNABLE} のどれか 1 つだけ変える: {changed}"
    config.update(changed)
    tag = "baseline" if not changed else "-".join(f"{k}={v}" for k, v in changed.items())
    if args.seed is not None:
        config["seed"] = args.seed
        tag += f"-seed={args.seed}"
    if args.data != "course": tag = f"{args.data}-{tag}"
    out_dir = Path(args.out or f"runs/{tag}")
    out_dir.mkdir(parents=True, exist_ok=True)

    data = importlib.import_module(DATA_MODULES[args.data])
    train, valid, test, gold = build_datasets(data)
    print(f"train={len(train)} valid={len(valid)} test={len(test)} | tag={tag} | out={out_dir}", flush=True)
    if args.n_test > 0: test = test[:args.n_test]

    tok = AutoTokenizer.from_pretrained(config["base_model"], revision=config["revision"])
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(config["base_model"], revision=config["revision"], dtype=torch.float32).to(device)
    model.config.use_cache = True
    print("loaded:", config["base_model"], "| device:", device, "| 重みのメモリ(GB):", round(model.get_memory_footprint() / 1e9, 2), flush=True)

    before_summary, before_rows = evaluate(tok, model, "before", test, gold, config, out_dir)

    def to_chat(example):
        return {"messages": build_messages(example) + [{"role": "assistant", "content": example["target"]}]}

    train_ds = Dataset.from_list([to_chat(e) for e in train])
    valid_ds = Dataset.from_list([to_chat(e) for e in valid])
    model.config.use_cache = False
    lora = LoraConfig(r=config["lora_r"], lora_alpha=config["lora_alpha"], lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    logging_steps = min(10, args.max_steps) if args.max_steps > 0 else 10
    sft = SFTConfig(output_dir=str(out_dir / "ckpt"), num_train_epochs=config["num_train_epochs"], max_steps=args.max_steps,
                    per_device_train_batch_size=4, gradient_accumulation_steps=2, learning_rate=config["lr"],
                    lr_scheduler_type="cosine", warmup_steps=config["warmup_steps"], logging_steps=logging_steps,
                    save_strategy="no", max_length=config["max_seq_len"], packing=False, report_to="none", seed=config["seed"])
    trainer = SFTTrainer(model=model, args=sft, train_dataset=train_ds, eval_dataset=valid_ds, peft_config=lora, processing_class=tok)
    trainer.model.print_trainable_parameters()
    t0 = time.time()
    trainer.train()
    train_seconds = round(time.time() - t0, 1)
    losses = [h["loss"] for h in trainer.state.log_history if "loss" in h]
    final_loss = losses[-1] if losses else None
    print(f"学習時間: {train_seconds} 秒 | ステップ数: {trainer.state.global_step} | エポック: {trainer.state.epoch:.1f} | 最終 loss: {final_loss}", flush=True)

    trainer.model.save_pretrained(out_dir / "adapter")
    tuned = trainer.model
    tuned.config.use_cache = True
    tuned.eval()

    after_summary, after_rows = evaluate(tok, tuned, "after", test, gold, config, out_dir)
    regressed = [b["question"] for b, a in zip(before_rows, after_rows) if b["both_ok"] and not a["both_ok"]]
    print("退行:", regressed, flush=True)

    knowledge_rows = []
    for q in data.KNOWLEDGE_QUESTIONS:
        ex = make_example(data, q, "", True)
        with tuned.disable_adapter():
            before_raw = generate(tok, tuned, ex, config)
        after_raw = generate(tok, tuned, ex, config)
        knowledge_rows.append({"質問": q, "FAQ にある": q in gold, "学習前の分類": classify(before_raw), "学習前の出力": before_raw,
                               "学習後の分類": classify(after_raw), "学習後の出力": after_raw})

    result = {
        "tag": tag,
        "data": args.data,
        "config": config,
        "changed": changed,
        "versions": {m.__name__: m.__version__ for m in [transformers, peft, trl, torch]},
        "method": "LoRA(量子化なし)",
        "train": {"final_loss": final_loss, "train_seconds": train_seconds, "steps": trainer.state.global_step,
                  "epochs": round(trainer.state.epoch, 2), "max_steps_override": args.max_steps, "n_test": len(test)},
        "before": before_summary, "after": after_summary,
        "regressed": regressed,
        "knowledge": knowledge_rows,
        "executed_training_myself": True,
    }
    (out_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("saved:", out_dir / "result.json", flush=True)


def compare(args):
    results = [json.loads(Path(p).read_text(encoding="utf-8")) for p in args.results]
    cols = [("tag", lambda r: r["tag"]),
            ("lora_r", lambda r: r["config"]["lora_r"]),
            ("seed", lambda r: r["config"]["seed"]),
            ("lr", lambda r: r["config"]["lr"]),
            ("epochs", lambda r: r["config"]["num_train_epochs"]),
            ("steps", lambda r: r["train"]["steps"]),
            ("format_rate", lambda r: f"{r['before']['format_rate']:.2f} -> {r['after']['format_rate']:.2f}"),
            ("content_rate", lambda r: f"{r['before']['content_rate']:.2f} -> {r['after']['content_rate']:.2f}"),
            ("final_loss", lambda r: "n/a" if r["train"]["final_loss"] is None else f"{r['train']['final_loss']:.4f}"),
            ("train_seconds", lambda r: r["train"]["train_seconds"]),
            ("regressed", lambda r: len(r["regressed"]))]
    print("| " + " | ".join(name for name, _ in cols) + " |")
    print("|" + "---|" * len(cols))
    for r in results:
        print("| " + " | ".join(str(get(r)) for _, get in cols) + " |")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="学習前評価 -> 学習 -> 学習後評価")
    p_run.add_argument("--epochs", type=int, help="num_train_epochs(既定 20)")
    p_run.add_argument("--lr", type=float, help="学習率(既定 2e-4)")
    p_run.add_argument("--lora-r", type=int, help="LoRA の次元(既定 16)")
    p_run.add_argument("--data", choices=list(DATA_MODULES), default="course", help="使う FAQ データ(既定 course。実装は hitorajio / hitorajio_more)")
    p_run.add_argument("--seed", type=int, help="乱数 seed(既定 42)。ゆらぎの確認用で、変更項目には数えない")
    p_run.add_argument("--out", help="出力先(既定 runs/<tag>)")
    p_run.add_argument("--n-test", type=int, default=0, help="評価に使う test の先頭件数(0 なら全 21 問。smoke test 用)")
    p_run.add_argument("--max-steps", type=int, default=-1, help="学習ステップ数の上限(-1 なら epochs どおり。smoke test 用)")
    p_run.set_defaults(func=run)

    p_cmp = sub.add_parser("compare", help="result.json を並べて表にする")
    p_cmp.add_argument("results", nargs="+", help="result.json のパス(2 つ以上)")
    p_cmp.set_defaults(func=compare)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
