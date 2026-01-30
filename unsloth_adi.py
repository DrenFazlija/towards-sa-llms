import pandas as pd
from pathlib import Path
import json
import re
from tqdm import tqdm
import argparse
import torch

from unsloth import FastLanguageModel
from transformers import TextStreamer
from peft import PeftModel 


def looks_like_adapter_dir(p: Path) -> bool:
    """Heuristic: local folder contains PEFT adapter artifacts."""
    return (p / "adapter_config.json").exists() and (
        (p / "adapter_model.safetensors").exists() or (p / "adapter_model.bin").exists()
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Unsloth model on a batch of inputs.")
    parser.add_argument("--inputs", type=str, required=True, help="Path to the input CSV file")
    parser.add_argument("--continue_at", type=int, default=0, help="Line number to continue processing from")
    parser.add_argument("--to_drop", type=str, default="../2025_datasets/0/to_drop.txt", help="Path to the to_drop.txt file")

    # Base model (actual weights)
    parser.add_argument("--model", type=str, default="unsloth/qwen3-8b-unsloth-bnb-4bit", help="Base model id")

    # NEW: Public HF adapter repo id (or local adapter dir)
    parser.add_argument(
        "--adapter_repo",
        type=str,
        default=None,
        help="HF repo id of LoRA adapter (e.g., 'SisWiss/qwen3-8b-sa-lora') OR a local adapter directory.",
    )

    parser.add_argument("--temperature", type=float, default=0.6, help="Temperature for the model")
    parser.add_argument("--max_new_tokens", type=int, default=2048, help="Max new tokens to generate")
    parser.add_argument("--input_batch_exists", type=bool, default=False, help="Whether the input batch exists")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save outputs")

    # Legacy options for loading checkpoints / local adapters
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to a specific checkpoint directory")
    parser.add_argument("--trainer_output_dir", type=str, default=None, help="Training output_dir containing checkpoints")
    parser.add_argument("--checkpoint_step", type=int, default=None, help="Checkpoint step to load")
    parser.add_argument("--adapter_dir", type=str, default=None, help="Local directory with LoRA adapters (legacy)")

    parser.add_argument("--identifier", type=str, default=None, help="Identifier to use for the output directory")

    args = parser.parse_args()

    # Resolve source
    # - If checkpoint_* is provided: load that directory as a full model *or* an adapter if it looks like one.
    # - Otherwise: load base model and optionally attach adapter (--adapter_repo or --adapter_dir).
    load_source = None
    if args.checkpoint_path:
        load_source = args.checkpoint_path
    elif args.trainer_output_dir and args.checkpoint_step is not None:
        load_source = str(Path(args.trainer_output_dir) / f"checkpoint-{args.checkpoint_step}")
    elif args.adapter_dir:
        # legacy local adapter
        pass
    else:
        # base-only mode (or base + adapter_repo)
        pass

    inputs = Path(args.inputs)
    pattern = r"----- USER QUERY -----\n([\s\S]*?)\n----- END USER QUERY -----\n"
    pat = re.compile(pattern)

    df = pd.read_csv(inputs, sep=chr(30))
    if Path(args.to_drop).exists():
        to_drop = Path(args.to_drop).read_text().split("\n")
        print(f"Dropping ids from to_drop.txt! Amount: {len(to_drop)}")
        df = df[~df["id"].isin(to_drop)]

    # Build output directory name
    base_model_name = args.model.split("/")[-1] if args.model else (Path(load_source).name if load_source else "model")
    step_suffix = None
    if args.checkpoint_path:
        m = re.search(r"checkpoint-(\d+)", args.checkpoint_path)
        if m:
            step_suffix = m.group(1)
    elif args.trainer_output_dir and args.checkpoint_step is not None:
        step_suffix = str(args.checkpoint_step)

    model_name = f"{base_model_name}-steps{step_suffix}" if step_suffix else base_model_name
    if args.identifier:
        model_name = f"{model_name}-{args.identifier}"

    if args.output_dir:
        model_path = Path(args.output_dir) / f"models/{model_name}"
    else:
        model_path = Path(inputs.parent / f"models/{model_name}")
    print(model_path)

    batch_input_path = model_path / "batch_input.jsonl"
    batch_output_path = model_path / "batch_output.jsonl"
    clean_output_path = model_path / "batch_output.csv"

    model_path.mkdir(parents=True, exist_ok=True)

    print("Getting the batch input from", batch_input_path)

    if not args.input_batch_exists:
        with open(batch_input_path, "w") as f:
            for _, e in tqdm(df.iterrows(), total=len(df)):
                line = {"custom_id": e["id"], "input": e["input"]}
                f.write(json.dumps(line, separators=(",", ":")) + "\n")

    print("Saving the batch output to", batch_output_path)
    print("Saving the clean output to", clean_output_path)

    # -------------------------
    # Model loading
    # -------------------------
    adapter_source = args.adapter_repo or args.adapter_dir  # allow either

    # Case A: explicit checkpoint load requested
    if load_source is not None:
        p = Path(load_source)
        if p.exists() and looks_like_adapter_dir(p):
            # The "checkpoint" is actually an adapter directory: load base + attach adapter
            print(f"Loading BASE model from: {args.model}")
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=args.model,
                max_seq_length=3048,
                load_in_4bit=True,
                load_in_8bit=False,
                full_finetuning=False,
            )
            print(f"Attaching ADAPTER from local dir: {load_source}")
            model = PeftModel.from_pretrained(model, load_source)
        else:
            # Treat as full model checkpoint directory
            print(f"Loading FULL model from: {load_source}")
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=load_source,
                max_seq_length=3048,
                load_in_4bit=True,
                load_in_8bit=False,
                full_finetuning=False,
            )

    # Case B: normal eval mode: base model (+ optional public adapter)
    else:
        print(f"Loading BASE model from: {args.model}")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.model,
            max_seq_length=3048,
            load_in_4bit=True,
            load_in_8bit=False,
            full_finetuning=False,
        )

        if adapter_source:
            # If it's a local path and exists -> attach local adapter
            # Else assume it's an HF repo id like "SisWiss/qwen3-8b-sa-lora"
            if Path(adapter_source).exists():
                print(f"Attaching ADAPTER from local dir: {adapter_source}")
                model = PeftModel.from_pretrained(model, adapter_source)
            else:
                print(f"Attaching ADAPTER from HF Hub: {adapter_source}")
                model = PeftModel.from_pretrained(model, adapter_source)

    # Unsloth inference optimization (safe to call after adapter attachment)
    FastLanguageModel.for_inference(model)
    model.eval()

    # -------------------------
    # Inference loop
    # -------------------------
    continue_at = args.continue_at
    total_lines = sum(1 for _ in open(batch_input_path, "r"))

    with open(batch_input_path, "r") as f:
        for i, line in enumerate(tqdm(f, total=total_lines)):
            if i < continue_at:
                continue

            request = json.loads(line)
            text = request["input"]

            user_query = re.findall(pat, text)
            if not user_query:
                print("No query found in:", request["custom_id"])
                continue
            user_query = user_query[0]

            system_prompt = re.sub(
                r"----- USER QUERY -----\n(.*)\n----- END USER QUERY -----\n",
                "",
                text,
                flags=re.DOTALL,
            )

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query},
            ]

            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )

            inputs_tok = tokenizer(prompt, return_tensors="pt").to(model.device)
            streamer = TextStreamer(tokenizer, skip_prompt=True)

            output = model.generate(
                **inputs_tok,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=0.95,
                top_k=20,
                streamer=streamer,
            )

            # Keep your original behavior (full decode). If you want response-only,
            # slice off the prompt tokens: output[0][inputs_tok["input_ids"].shape[-1]:]
            decoded = tokenizer.decode(output[0], skip_special_tokens=True)

            response_dict = {"id": request["custom_id"], "response": decoded}
            with open(batch_output_path, "a") as out_f:
                out_f.write(json.dumps(response_dict, separators=(",", ":")) + "\n")

    print("Output saved to", batch_output_path)
    assert batch_output_path.exists(), "Output file failed to save"
