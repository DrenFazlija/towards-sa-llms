from unsloth import FastLanguageModel
import torch
import pandas as pd
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--model", type=str, default="unsloth/Qwen3-14B")
parser.add_argument("--local_name", type=str, default="lora_model_max_steps")
parser.add_argument("--resume_from_checkpoint", action="store_true", help="Resume training from the latest checkpoint")
# Critical-difference knobs (defaults match previous hard-coded values)
parser.add_argument("--learning_rate", type=float, default=2e-5)
parser.add_argument("--lora_alpha", type=int, default=32)
parser.add_argument("--weight_decay", type=float, default=0.01)
parser.add_argument("--lr_scheduler_type", type=str, default="linear")
parser.add_argument("--warmup_steps", type=int, default=5)
args = parser.parse_args()

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = args.model,
    max_seq_length = 3048,   
    load_in_4bit = True,     
    load_in_8bit = False,    
    full_finetuning = False, 
)


model = FastLanguageModel.get_peft_model(
    model,
    r = 32,           # Choose any number > 0! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = args.lora_alpha,  # Best to choose alpha = rank or rank*2
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,   # We support rank stabilized LoRA
    loftq_config = None,  # And LoftQ
)

from datasets import Dataset
processed = Dataset.from_pandas(pd.read_csv("processed_dataset.csv"))

from trl import SFTTrainer, SFTConfig
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = processed,
    eval_dataset = None, # Can set up evaluation!
    args = SFTConfig(
        dataset_text_field = "text",
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4, # Use GA to mimic batch size!
        warmup_steps = args.warmup_steps,
        num_train_epochs = 1, # Set this for 1 full training run.
        max_steps = -1,
        learning_rate = args.learning_rate, # Reduce to 2e-5 for long training runs
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = args.weight_decay,
        lr_scheduler_type = args.lr_scheduler_type,
        seed = 3407,
        report_to = "none", # Use this for WandB etc
        # Add these checkpointing parameters:
        save_steps = 200,           # Save every 100 steps
        #save_total_limit = 3,       # Keep only the last 3 checkpoints
        save_strategy = "steps",    # Save based on steps rather than epochs
    ),
)

trainer_stats = trainer.train(resume_from_checkpoint = args.resume_from_checkpoint)

print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(
    f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training."
)

model.save_pretrained(args.local_name)  # Local saving
tokenizer.save_pretrained(args.local_name)


