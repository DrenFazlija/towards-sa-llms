import pandas as pd
from pathlib import Path
import json
import re
from tqdm import tqdm
import argparse
import torch
from unsloth import FastLanguageModel
from transformers import TextStreamer

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Unsloth model on a batch of inputs.')
    parser.add_argument('--inputs', type=str, required=True, help='Path to the input CSV file')
    parser.add_argument('--continue_at', type=int, default=0, help='Line number to continue processing from')
    parser.add_argument('--to_drop', type=str, default='../2025_datasets/0/to_drop.txt', help='Path to the to_drop.txt file')
    parser.add_argument('--model', type=str, default='unsloth/Qwen3-14B', help='Model name')
    parser.add_argument('--temperature', type=float, default=0.6, help='Temperature for the model')
    parser.add_argument('--max_new_tokens', type=int, default=2048, help='Max new tokens to generate')
    parser.add_argument('--input_batch_exists', type=bool, default=False, help='Whether the input batch exists')
    parser.add_argument('--output_dir', type=str, default=None, help='Directory to save outputs')
    parser.add_argument('--checkpoint_path', type=str, default=None, help='Path to a specific checkpoint directory (e.g., trainer_output_xxx/checkpoint-200)')
    parser.add_argument('--trainer_output_dir', type=str, default=None, help='Training output_dir containing checkpoints (e.g., trainer_output_xxx)')
    parser.add_argument('--checkpoint_step', type=int, default=None, help='Checkpoint step to load from trainer_output_dir (e.g., 200 for checkpoint-200)')
    parser.add_argument('--adapter_dir', type=str, default=None, help='Directory with LoRA adapters (output_dir from finetuning)')
    parser.add_argument('--identifier', type=str, default=None, help='Identifier to use for the output directory')
    args = parser.parse_args()

    # Resolve which source to load for the model (checkpoint directory or base model)
    if args.checkpoint_path:
        load_source = args.checkpoint_path
    elif args.trainer_output_dir and args.checkpoint_step is not None:
        load_source = str(Path(args.trainer_output_dir) / f"checkpoint-{args.checkpoint_step}")
    elif args.adapter_dir:
        load_source = args.adapter_dir
    else:
        load_source = args.model

    inputs = Path(args.inputs)
    pattern = r'----- USER QUERY -----\n([\s\S]*?)\n----- END USER QUERY -----\n'
    pat = re.compile(pattern)

    df = pd.read_csv(inputs, sep=chr(30))
    if Path(args.to_drop).exists():
        to_drop = Path(args.to_drop).read_text().split('\n')
        print(f"Dropping ids from to_drop.txt! Amount: {len(to_drop)}")
        df = df[~df['id'].isin(to_drop)]

    # Build a descriptive directory name like: model_name-steps200 when using checkpoints
    base_model_name = args.model.split('/')[-1] if args.model else Path(load_source).name
    step_suffix = None
    if args.checkpoint_path:
        m = re.search(r'checkpoint-(\d+)', args.checkpoint_path)
        if m:
            step_suffix = m.group(1)
    elif args.trainer_output_dir and args.checkpoint_step is not None:
        step_suffix = str(args.checkpoint_step)

    if step_suffix:
        model_name = f"{base_model_name}-steps{step_suffix}"
    else:
        model_name = base_model_name
    if args.identifier:
        model_name = f"{model_name}-{args.identifier}"
    if args.output_dir:
        model_path = Path(args.output_dir) / f'models/{model_name}'
    else:
        model_path = Path(inputs.parent / f'models/{model_name}')
    print(model_path)

    batch_input_path = model_path / 'batch_input.jsonl'
    batch_output_path = model_path / 'batch_output.jsonl'
    clean_output_path = model_path / 'batch_output.csv'

    if not model_path.exists():
        model_path.mkdir(parents=True)

    print('Getting the batch input from', batch_input_path)

    # Prepare batch input if not already present
    if not args.input_batch_exists:
        with open(batch_input_path, 'w') as f:
            for i, e in tqdm(df.iterrows(), total=len(df)):
                line = {
                    "custom_id": e['id'],
                    "input": e['input']
                }
                f.write(json.dumps(line, separators=(',', ':')))
                f.write('\n')

    print('Saving the batch output to', batch_output_path)
    print('Saving the clean output to', clean_output_path)

    # Load Unsloth model and tokenizer
    print(f"Loading model from: {load_source}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=load_source,
        max_seq_length=3048,
        load_in_4bit=True,
        load_in_8bit=False,
        full_finetuning=False,
    )

    continue_at = args.continue_at
    with open(batch_input_path, 'r') as f:
        for i, line in enumerate(tqdm(f, total=sum(1 for _ in open(batch_input_path)))):
            if i < continue_at:
                continue
            request = json.loads(line)
            text = request['input']
            # Extract user query and system prompt
            user_query = re.findall(pat, text)
            if not user_query:
                print('No query found in:', request['custom_id'])
                continue
            user_query = user_query[0]
            system_prompt = re.sub(r'----- USER QUERY -----\n(.*)\n----- END USER QUERY -----\n', '', text)
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query}
            ]
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
            # Run inference
            inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
            streamer = TextStreamer(tokenizer, skip_prompt=True)
            output = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=0.95, top_k=20,
                streamer=streamer,
            )
            # Decode output
            decoded = tokenizer.decode(output[0], skip_special_tokens=True)
            # Save output
            response_dict = {
                'id': request['custom_id'],
                'response': decoded
            }
            with open(batch_output_path, 'a') as out_f:
                out_f.write(json.dumps(response_dict, separators=(',', ':')))
                out_f.write('\n')
    print('Output saved to', batch_output_path)
    assert batch_output_path.exists(), 'Output file failed to save'
