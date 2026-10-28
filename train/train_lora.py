"""LoRA fine-tune on GemMaroc's Darija data (PLAN.md Phase 2b, Track A, steps A5/A6).

Recipe (GemMaroc paper): LoRA on the instruction-tuned model, lr 1e-4, 3 epochs,
max length 2048, bf16; r32/alpha64 for their 4B model. Not stated in the paper and
chosen here: loss on the assistant replies only (as in TULU's own training), linear
warmup 3% then cosine decay, effective batch 16, LoRA on attention + MLP of the
language layers only (vision/audio towers stay untouched; we train text-only).

Data, rebuilt with the model's own chat template from the raw conversations (the
datasets' ready-made `text` column uses the Gemma 2/3 format, wrong for Gemma 4):
  tulu          GemMaroc/TULU-3-50k-darija-english (44,725; ~45.5M tokens/epoch)   run 2
  tulu+atlas    + GemMaroc/AtlasChat-data-100k-randomlysampled-darija (~44M)       (planned run 3, dropped)
  --data-file   a ready-made mix from train/atlas_mix.py build (jsonl with `messages`); its
                held-out <name>-val.jsonl gives the eval loss. Run 3 = data/train_mix/atlas-tutor.jsonl
                with Atlas-Chat settings: --r 256 --alpha 128 --lr 5e-5 --epochs 1 --save-steps 250

Local (8 GB laptop GPU) = small slices to find bugs: --qlora loads the frozen
weights in 4 bit, keeps Gemma 4's per-layer embedding tables on the CPU.
Rented GPU = the real runs: bf16, no --qlora.

  .venv-train\\python.exe train\\train_lora.py --model google/gemma-4-E2B-it --qlora --limit 300 --max-steps 30 --max-len 1024 --out data\\train_runs\\e2b-smoke
  python train/train_lora.py --model google/gemma-4-E4B-it --data tulu --batch 4 --accum 4 --keep 0 --out /workspace/runs/e4b-tulu   (rented GPU, see train/RUNPOD.md)
"""
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import torch
import transformers as tf
from huggingface_hub import snapshot_download
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

DATASETS = {
    "tulu": ("GemMaroc/TULU-3-50k-darija-english", "dialogs"),
    "atlas": ("GemMaroc/AtlasChat-data-100k-randomlysampled-darija", "messages"),
}
MIXES = {"tulu": ["tulu"], "tulu+atlas": ["tulu", "atlas"]}
# attention + MLP projections of the language layers; excludes the vision/audio
# towers and Gemma 4's per-layer-embedding projections
LORA_TARGETS = r".*language_model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)"


def load_conversations(name: str, split: str = "train") -> list[list[dict]]:
    repo, col = DATASETS[name]
    path = snapshot_download(repo, repo_type="dataset", allow_patterns=f"data/{split}-*.parquet")
    files = sorted(Path(path, "data").glob(f"{split}-*.parquet"))
    if not files:
        return []
    rows = pa.concat_tables([pq.read_table(f, columns=[col]) for f in files]).column(col).to_pylist()
    return [[{"role": m["role"], "content": m["content"]} for m in conv] for conv in rows]


def load_jsonl(path: Path) -> list[list[dict]]:
    convs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                convs.append([{"role": m["role"], "content": m["content"]} for m in json.loads(line)["messages"]])
    return convs


def tokenize(tok, conv: list[dict], max_len: int) -> dict | None:
    """Token ids of the whole conversation; labels only on assistant replies
    (-100 = ignored by the loss). Each reply's span is found by rendering the
    conversation up to it (with the generation prompt) and up to its end."""
    full = tok.apply_chat_template(conv, tokenize=False)
    ids = tok(full, add_special_tokens=False)["input_ids"]  # template already has <bos>
    labels = [-100] * len(ids)
    for i, m in enumerate(conv):
        if m["role"] != "assistant":
            continue
        start = len(tok(tok.apply_chat_template(conv[:i], tokenize=False, add_generation_prompt=True),
                        add_special_tokens=False)["input_ids"])
        end = len(tok(tok.apply_chat_template(conv[:i + 1], tokenize=False),
                      add_special_tokens=False)["input_ids"])
        if ids[:start] != tok(tok.apply_chat_template(conv[:i], tokenize=False, add_generation_prompt=True),
                              add_special_tokens=False)["input_ids"]:
            return None  # template not prefix-consistent here: skip rather than mislabel
        labels[start:end] = ids[start:end]
    ids, labels = ids[:max_len], labels[:max_len]
    if all(l == -100 for l in labels):
        return None
    return {"input_ids": ids, "labels": labels}


def load_model(args):
    kw = {"dtype": torch.bfloat16}
    if args.attn:
        kw["attn_implementation"] = args.attn
    if args.qlora:
        kw["quantization_config"] = tf.BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True, llm_int8_enable_fp32_cpu_offload=True,
            llm_int8_skip_modules=["embed_tokens_per_layer", "vision_tower", "audio_tower", "lm_head"])
        # Gemma 4's per-layer embedding tables (2.3-2.8B params) are lookups: keep them
        # (and the unused towers) in system RAM, everything else on the GPU
        kw["device_map"] = {"model.language_model.embed_tokens_per_layer": "cpu",
                            "model.vision_tower": "cpu", "model.audio_tower": "cpu", "": 0}
    else:
        kw["device_map"] = {"": 0}
    model = tf.AutoModelForCausalLM.from_pretrained(args.model, **kw)
    if args.qlora:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    elif not args.no_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    print("attention:", model.config._attn_implementation, "| gradient checkpointing:", not args.no_checkpointing)
    model.enable_input_require_grads()
    lora = LoraConfig(r=args.r, lora_alpha=args.alpha, lora_dropout=0.05, target_modules=LORA_TARGETS,
                      task_type="CAUSAL_LM")
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    return model


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="Hugging Face repo of the instruction-tuned model")
    ap.add_argument("--data", choices=MIXES, default="tulu")
    ap.add_argument("--data-file", help="jsonl mix from train/atlas_mix.py (overrides --data); "
                    "eval loss on the <name>-val.jsonl next to it")
    ap.add_argument("--limit", type=int, default=0, help="use only N training conversations (0 = all)")
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--max-steps", type=int, default=-1, help="stop after N optimizer steps (smoke tests)")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--r", type=int, default=32)
    ap.add_argument("--alpha", type=int, default=64)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--batch", type=int, default=1, help="per-device batch size")
    ap.add_argument("--accum", type=int, default=16, help="gradient accumulation (effective batch = batch x accum)")
    ap.add_argument("--qlora", action="store_true", help="4-bit frozen weights (8 GB local GPU)")
    ap.add_argument("--no-checkpointing", action="store_true",
                    help="keep all activations instead of recomputing them (faster, needs more GPU memory)")
    ap.add_argument("--attn", default=None, help="attention implementation (sdpa, eager, flex_attention; default: model's choice)")
    ap.add_argument("--eval-n", type=int, default=200, help="validation conversations for eval loss")
    ap.add_argument("--save-steps", type=int, default=500)
    ap.add_argument("--keep", type=int, default=3, help="checkpoints to keep (0 = all; rented runs keep all to score early ones)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    tok = tf.AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if args.data_file:
        data_file = Path(args.data_file)
        convs = load_jsonl(data_file)
        val_convs = load_jsonl(data_file.with_name(data_file.stem + "-val.jsonl"))
    else:
        convs = [c for name in MIXES[args.data] for c in load_conversations(name)]
        val_convs = load_conversations("tulu", "validation")
    random.Random(args.seed).shuffle(convs)
    if args.limit:
        convs = convs[:args.limit]
    t0 = time.perf_counter()
    train = [x for x in (tokenize(tok, c, args.max_len) for c in convs) if x]
    val = [x for x in (tokenize(tok, c, args.max_len) for c in val_convs[:args.eval_n]) if x]
    n_tok = sum(len(x["input_ids"]) for x in train)
    n_lab = sum(sum(l != -100 for l in x["labels"]) for x in train)
    print(f"train {len(train)}/{len(convs)} conversations ({len(convs) - len(train)} skipped), {n_tok/1e6:.2f}M tokens, "
          f"{n_lab/1e6:.2f}M in the loss; val {len(val)}; tokenized in {time.perf_counter() - t0:.0f} s", flush=True)
    sample = train[0]
    print("sample start:", repr(tok.decode(sample["input_ids"][:40])))
    first = next(i for i, l in enumerate(sample["labels"]) if l != -100)
    print("loss starts at:", repr(tok.decode(sample["input_ids"][first:first + 20])))

    model = load_model(args)
    targs = tf.TrainingArguments(
        output_dir=str(out), num_train_epochs=args.epochs, max_steps=args.max_steps,
        per_device_train_batch_size=args.batch, per_device_eval_batch_size=args.batch,
        gradient_accumulation_steps=args.accum, learning_rate=args.lr, lr_scheduler_type="cosine",
        warmup_steps=0.03, bf16=True, logging_steps=5, eval_strategy="steps",  # float < 1 = ratio (transformers 5)
        eval_steps=args.save_steps, save_strategy="steps", save_steps=args.save_steps, save_total_limit=args.keep or None,
        train_sampling_strategy="group_by_length", report_to="none", seed=args.seed, dataloader_num_workers=0,
        # plain AdamW: LoRA trains ~50-100M parameters, so its state is only a few hundred MB.
        # (paged_adamw_8bit hung for 13+ min writing optimizer.pt at the first checkpoint)
        remove_unused_columns=False, optim="adamw_torch")
    trainer = tf.Trainer(model=model, args=targs, train_dataset=train, eval_dataset=val,
                         data_collator=tf.DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100))
    t0 = time.perf_counter()
    trainer.train(resume_from_checkpoint=any(out.glob("checkpoint-*")) or None)
    secs = time.perf_counter() - t0
    final = trainer.evaluate()
    model.save_pretrained(out / "adapter")
    tok.save_pretrained(out / "adapter")
    steps = trainer.state.global_step
    summary = {"args": vars(args), "train_conversations": len(train), "train_tokens": n_tok, "loss_tokens": n_lab,
               "steps": steps, "seconds": round(secs), "final_eval_loss": final.get("eval_loss"),
               "log": trainer.state.log_history}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    seen = min(1.0, steps * args.batch * args.accum / max(len(train), 1)) * n_tok
    print(f"done: {steps} steps in {secs/60:.1f} min, ~{seen / secs:.0f} tokens/s; eval loss {final.get('eval_loss'):.3f}; "
          f"peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GB")


if __name__ == "__main__":
    main()
