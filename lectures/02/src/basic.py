# 第2回 演習: ノートブック(lora_format_tuning.ipynb)と同じ架空 FAQ・採点・設定を、スクリプトにして実行する。
# 学習前に test 21 問を評価 -> LoRA で学習 -> 学習後に評価し、result.json にまとめる。
#
# Usage (Colab, T4):
#   1. %pip uninstall -y -q torchao  →  %pip install -q "transformers==5.18.0" "peft==0.21.1" "trl==1.14.2" "accelerate==1.15.0" "datasets==5.1.0"
#   2. !python basic.py   (T4 で学習 4〜5 分 + 評価 2 回)
# Usage (local):  uv run basic.py
# 設定を変えるときは下の「設定」の定数を書き換える。結果の表との対応は answer.md の「再現方法」を参照。
# Outputs: RUNS_DIR/epochs=E-lora_r=R-lr=X-seed=S/ に result.json, eval_before.json, eval_after.json, adapter/
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

# ---- 設定 ----
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
BATCH_SIZE = 4
GRAD_ACCUM = 2
N_TEST = 0                   # 評価に使う test の先頭件数(0 なら全 21 問。動作確認用)
MAX_STEPS = -1               # 学習ステップ数の上限(-1 なら num_train_epochs どおり。動作確認用)
RUNS_DIR = "runs"

# ---- 架空 FAQ(講座運営) ----
FAQ = {
    "deadline": "課題の提出期限は 9 月 30 日 23:59（日本時間）です。",
    "destination": "提出先は学習ポータルの「課題」欄です。",
    "format": "提出形式は PDF 1 ファイルで、ファイル名は 学籍番号_課題番号.pdf です。",
    "office_hours": "質問は毎週火曜 17:00〜18:00 のオフィスアワーで受け付けます。",
    "late": "期限後の提出は、期限から 48 時間以内なら減点のうえ受け付けます。",
}
FAQ_TEXT = "\n".join(f"- {v}" for v in FAQ.values())

GROUNDED = {
    "deadline": (["課題の提出期限はいつですか", "いつまでに出せばいいですか", "締切を教えてください", "提出の期限は？"], ["9 月 30 日", "23:59"]),
    "destination": (["提出先はどこですか", "どこにアップロードしますか", "課題はどこから出しますか", "提出場所は？"], ["学習ポータル", "課題"]),
    "format": (["提出形式を教えてください", "ファイル名の決まりはありますか", "何のファイル形式で出しますか", "PDF 以外でも出せますか"], ["PDF"]),
    "office_hours": (["質問はいつできますか", "オフィスアワーはいつですか", "先生に質問する時間はありますか", "相談できる時間帯は？"], ["火曜", "17:00"]),
    "late": (["遅れて出したらどうなりますか", "期限を過ぎた場合は受け付けますか", "遅延提出のルールは？", "締切後の扱いを教えてください"], ["48 時間", "減点"]),
}
UNGROUNDED = [
    "再提出はできますか", "欠席した回の扱いを教えてください", "他の受講者の点数は見られますか",
    "成績はいつ出ますか", "提出期限は延長できますか", "グループでの提出は可能ですか",
    "課題の配点はどうなっていますか", "講義の録画は公開されますか",
]
TEST_EXTRA_GROUNDED = {
    "deadline": ["提出期限の日付と時刻を教えてください", "課題の締切は何月何日の何時ですか"],
    "destination": ["課題の提出はどこで行いますか", "提出物を出す先を教えてください"],
    "format": ["提出するファイルの形式は何ですか", "提出ファイルは何形式で、いくつ出しますか"],
    "office_hours": ["オフィスアワーの曜日と時間を教えてください", "質問を受け付けてもらえるのは何曜日の何時からですか"],
    "late": ["提出が締切に間に合わなかったらどうなりますか", "期限後に提出した課題は受理されますか"],
}
TEST_EXTRA_UNGROUNDED = [
    "課題の文字数に制限はありますか", "提出した課題へのフィードバックはもらえますか",
    "参考文献の書き方に指定はありますか", "オフィスアワーは事前予約が必要ですか",
]
UNGROUNDED_ANSWER = "この FAQ には記載がありません。運営に確認が必要です。"
GROUNDED_TEMPLATES_IN_TRAIN = 2

KNOWLEDGE_QUESTIONS = [
    "この講座の最終課題の締切はいつですか",
    "担当教員の名前は何ですか",
    "次回の講義のテーマは何ですか",
    "課題の提出期限はいつですか",
]

SYSTEM = "あなたは講座 FAQ の回答係です。与えられた FAQ だけを根拠に、必ず JSON 1 行で答えてください。形式: {\"answer\": 文字列, \"needs_review\": 真偽値}。FAQ に記載が無いことは answer に『確認が必要』と書き、needs_review を true にします。"
TRAIN_EXTRA_UNGROUNDED = []


def make_example(question: str, answer: str, needs_review: bool) -> dict:
    target = json.dumps({"answer": answer, "needs_review": needs_review}, ensure_ascii=False)
    return {"question": question, "faq": FAQ_TEXT, "system": SYSTEM, "target": target, "needs_review": needs_review}


def build_datasets():
    train, valid, test, gold = [], [], [], {}
    for key, (templates, must_include) in GROUNDED.items():
        n = GROUNDED_TEMPLATES_IN_TRAIN
        for q in templates[:n]: train.append(make_example(q, FAQ[key], False))
        for q in templates[n:n + 1]: valid.append(make_example(q, FAQ[key], False))
        for q in templates[n + 1:n + 2] + TEST_EXTRA_GROUNDED[key]:
            test.append(make_example(q, FAQ[key], False))
            gold[q] = {"needs_review": False, "must_include": must_include}
    for i, q in enumerate(UNGROUNDED):
        ex = make_example(q, UNGROUNDED_ANSWER, True)
        bucket = [train, train, valid, test][i % 4]
        bucket.append(ex)
        if bucket is test: gold[q] = {"needs_review": True, "must_include": ["確認"]}
    for q in TEST_EXTRA_UNGROUNDED:
        test.append(make_example(q, UNGROUNDED_ANSWER, True))
        gold[q] = {"needs_review": True, "must_include": ["確認"]}
    for q in TRAIN_EXTRA_UNGROUNDED:
        train.append(make_example(q, UNGROUNDED_ANSWER, True))

    all_questions = [r["question"] for r in train + valid + test]
    assert len(set(all_questions)) == len(all_questions), "質問が train / valid / test で重複しています"
    return train, valid, test, gold


def build_messages(example):
    user = f"FAQ:\n{example['faq']}\n\n質問: {example['question']}"
    return [{"role": "system", "content": example["system"]}, {"role": "user", "content": user}]


def generate(tok, model, example):
    prompt = tok.apply_chat_template(build_messages(example), tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=CONFIG["max_new_tokens"], pad_token_id=tok.pad_token_id, **CONFIG["gen"])
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


def evaluate(tok, model, label, test, gold, out_dir):
    rows = []
    t0 = time.time()
    for ex in test:
        raw = generate(tok, model, ex)
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


def run():
    tag = f"epochs={CONFIG['num_train_epochs']}-lora_r={CONFIG['lora_r']}-lr={CONFIG['lr']:g}-seed={CONFIG['seed']}"
    out_dir = Path(RUNS_DIR) / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    train, valid, test, gold = build_datasets()
    print(f"train={len(train)} valid={len(valid)} test={len(test)} | tag={tag} | out={out_dir}", flush=True)
    if N_TEST > 0: test = test[:N_TEST]

    tok = AutoTokenizer.from_pretrained(CONFIG["base_model"], revision=CONFIG["revision"])
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(CONFIG["base_model"], revision=CONFIG["revision"], dtype=torch.float32).to(device)
    model.config.use_cache = True
    print("loaded:", CONFIG["base_model"], "| device:", device, "| 重みのメモリ(GB):", round(model.get_memory_footprint() / 1e9, 2), flush=True)

    before_summary, before_rows = evaluate(tok, model, "before", test, gold, out_dir)

    def to_chat(example):
        return {"messages": build_messages(example) + [{"role": "assistant", "content": example["target"]}]}

    train_ds = Dataset.from_list([to_chat(e) for e in train])
    valid_ds = Dataset.from_list([to_chat(e) for e in valid])
    model.config.use_cache = False
    lora = LoraConfig(r=CONFIG["lora_r"], lora_alpha=CONFIG["lora_alpha"], lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    logging_steps = min(10, MAX_STEPS) if MAX_STEPS > 0 else 10
    sft = SFTConfig(output_dir=str(out_dir / "ckpt"), num_train_epochs=CONFIG["num_train_epochs"], max_steps=MAX_STEPS,
                    per_device_train_batch_size=BATCH_SIZE, gradient_accumulation_steps=GRAD_ACCUM, learning_rate=CONFIG["lr"],
                    lr_scheduler_type="cosine", warmup_steps=CONFIG["warmup_steps"], logging_steps=logging_steps,
                    save_strategy="no", max_length=CONFIG["max_seq_len"], packing=False, report_to="none", seed=CONFIG["seed"])
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

    after_summary, after_rows = evaluate(tok, tuned, "after", test, gold, out_dir)
    regressed = [b["question"] for b, a in zip(before_rows, after_rows) if b["both_ok"] and not a["both_ok"]]
    print("退行:", regressed, flush=True)

    knowledge_rows = []
    for q in KNOWLEDGE_QUESTIONS:
        ex = make_example(q, "", True)
        with tuned.disable_adapter():
            before_raw = generate(tok, tuned, ex)
        after_raw = generate(tok, tuned, ex)
        knowledge_rows.append({"質問": q, "FAQ にある": q in gold, "学習前の分類": classify(before_raw), "学習前の出力": before_raw,
                               "学習後の分類": classify(after_raw), "学習後の出力": after_raw})

    result = {
        "tag": tag,
        "config": CONFIG,
        "versions": {m.__name__: m.__version__ for m in [transformers, peft, trl, torch]},
        "method": "LoRA(量子化なし)",
        "train": {"final_loss": final_loss, "train_seconds": train_seconds, "steps": trainer.state.global_step,
                  "epochs": round(trainer.state.epoch, 2), "max_steps_override": MAX_STEPS, "n_test": len(test)},
        "before": before_summary, "after": after_summary,
        "regressed": regressed,
        "knowledge": knowledge_rows,
        "executed_training_myself": True,
    }
    (out_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("saved:", out_dir / "result.json", flush=True)


if __name__ == "__main__":
    run()
