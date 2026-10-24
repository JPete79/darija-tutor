"""Trained LoRA adapter -> merged model -> GGUF -> Q4_K_M -> Ollama (PLAN.md Track A, A5).

Steps: merge the adapter into the bf16 base model on the CPU (system RAM: ~10 GB for
E2B, ~16 GB for E4B), save it in Hugging Face format, convert it with llama.cpp's
converter from the same release as our llama.cpp binaries (b11565), quantize to
Q4_K_M with llama-quantize, and register it in Ollama.

--no-adapter runs the same pipeline on the released model, so "before vs after
training" compares two models quantized the same way (our unsloth GGUFs use a
calibrated imatrix quantization; ours is plain Q4_K_M).

  .venv-train\\python.exe train\\export_gguf.py --base google/gemma-4-E2B-it --adapter data\\train_runs\\e2b-smoke\\adapter --name e2b-smoke
  .venv-train\\python.exe train\\export_gguf.py --base google/gemma-4-E4B-it --no-adapter --name gemma-4-e4b-ours
"""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LLAMA = ROOT / "models" / "llama.cpp"
CONVERTER = ROOT / "models" / "llama.cpp-src" / "llama.cpp-b11565"
OUT = ROOT / "models" / "trained"


def run(cmd: list, **kw) -> None:
    print(">", " ".join(map(str, cmd)), flush=True)
    subprocess.run([str(c) for c in cmd], check=True, **kw)


def merge(base: str, adapter: str | None, out: Path) -> None:
    import torch
    import transformers as tf
    from huggingface_hub import snapshot_download
    model = tf.AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16, device_map="cpu")
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    model.save_pretrained(out, safe_serialization=True)
    # tokenizer, chat template, processor configs: copied unchanged from the base model
    src = Path(snapshot_download(base))
    for f in src.iterdir():
        if f.is_file() and not f.name.endswith(".safetensors") and f.name not in ("config.json", "model.safetensors.index.json"):
            shutil.copy(f, out / f.name)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True, help="Hugging Face repo of the model the adapter was trained on")
    ap.add_argument("--adapter", help="adapter folder written by train_lora.py")
    ap.add_argument("--no-adapter", action="store_true", help="export the released model itself")
    ap.add_argument("--name", required=True, help="Ollama model name (also the file name)")
    ap.add_argument("--quant", default="Q4_K_M")
    ap.add_argument("--keep", action="store_true", help="keep the merged HF folder and the bf16 GGUF (~10-16 GB)")
    ap.add_argument("--reuse", action="store_true",
                    help="reuse a merged folder / bf16 GGUF kept with --keep (default: delete leftovers and redo)")
    args = ap.parse_args()
    if not args.adapter and not args.no_adapter:
        sys.exit("give --adapter, or --no-adapter to export the released model")
    sys.stdout.reconfigure(encoding="utf-8")

    OUT.mkdir(parents=True, exist_ok=True)
    merged = OUT / f"{args.name}-hf"
    bf16 = OUT / f"{args.name}-bf16.gguf"
    final = OUT / f"{args.name}-{args.quant}.gguf"
    # Leftovers from an interrupted export can look complete but be corrupt (a crash on 2026-10-24
    # left a bf16 GGUF with a valid header whose data never reached the disk), so redo by default.
    if not args.reuse:
        if merged.exists():
            print(f"removing leftover {merged.name}")
            shutil.rmtree(merged)
        if bf16.exists():
            print(f"removing leftover {bf16.name}")
            bf16.unlink()
    if not merged.exists():
        merge(args.base, args.adapter, merged)
    if not bf16.exists():
        env = os.environ | {"PYTHONPATH": str(CONVERTER / "gguf-py")}
        run([sys.executable, CONVERTER / "convert_hf_to_gguf.py", merged, "--outtype", "bf16", "--outfile", bf16], env=env)
    run([LLAMA / "llama-quantize.exe", bf16, final, args.quant])

    modelfile = ROOT / "modelfiles" / f"{args.name}.Modelfile"
    modelfile.write_text(f"FROM ../models/trained/{final.name}\nPARAMETER num_ctx 2048\nPARAMETER num_gpu 99\n",
                         encoding="utf-8")
    run(["ollama", "create", args.name, "-f", modelfile.name], cwd=modelfile.parent)
    if not args.keep:
        shutil.rmtree(merged)
        bf16.unlink()
    print(f"done: Ollama model '{args.name}' from {final}")


if __name__ == "__main__":
    main()
