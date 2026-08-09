# Towards Sensitivity-Aware Language Models

This repository contains the scripts for fine-tuning and running inference with Qwen3 models using Unsloth used in the AISTATS 2026 publication "Towards Sensitivity-Aware Language Models".

> [!NOTE]
Feel free to reach out to us if you have any issues with the codebase. We'll happily provide you with our local setup and files.

## Prerequisites

- Conda (or Miniconda/Mamba)
- CUDA-capable GPU

### Setting up the Conda Environment

**1. Create Conda Environment**
```bash 
conda create -n sallms python=3.11.13
```

**2. Activate Conda Environment and Install Packages**
```bash
cd towards-sa-llms # Change to the Repository Directory
conda activate sallms
conda install pip # necessary for some conda versions
pip install -r requirements.txt
```

## Scripts

### 1. Fine-tuning: `qwen3_finetuning_vldb.py`

This script fine-tunes a Qwen3 model using LoRA (Low-Rank Adaptation) on a processed dataset.

#### Requirements
- `processed_dataset.csv` must exist in the working directory with a `text` column
- Please download the csv-file from the following Zenodo repository: https://doi.org/10.5281/zenodo.18412000


#### Usage

**Basic usage:**
```bash
python qwen3_finetuning_vldb.py
```

**With custom arguments:**
```bash
python qwen3_finetuning_vldb.py \
    --model unsloth/Qwen3-14B \
    --local_name my_lora_model \
    --learning_rate 2e-5 \
    --lora_alpha 32 \
    --weight_decay 0.01 \
    --lr_scheduler_type linear \
    --warmup_steps 5
```

**Fine-tuning with Qwen3-8B base model:**
```bash
python qwen3_finetuning_vldb.py \
    --model unsloth/Qwen3-8B \
    --local_name qwen3_8b_custom
```

**Resume from checkpoint:**
```bash
python qwen3_finetuning_vldb.py --resume_from_checkpoint
```

#### Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model` | str | `unsloth/Qwen3-14B` | Base model to fine-tune |
| `--local_name` | str | `lora_model_max_steps` | Directory name to save the fine-tuned model |
| `--resume_from_checkpoint` | flag | False | Resume training from the latest checkpoint |
| `--learning_rate` | float | `2e-5` | Learning rate for training |
| `--lora_alpha` | int | `32` | LoRA alpha parameter (typically rank or rank*2) |
| `--weight_decay` | float | `0.01` | Weight decay for optimizer |
| `--lr_scheduler_type` | str | `linear` | Learning rate scheduler type |
| `--warmup_steps` | int | `5` | Number of warmup steps |

#### Output

- The fine-tuned model and tokenizer are saved to the directory specified by `--local_name`
- Training checkpoints are saved every 200 steps in the default trainer output directory

---

### 2. Inference: `unsloth_adi.py`

This script runs batch inference on a pre-prepared ADI test dataset using either:
- a **base model**, or
- a **base model + LoRA adapter** (local folder **or** public Hugging Face Hub repo), or
- a **full checkpoint directory** (if you have one).

> **Scope note:** The released LoRA adapters are specialized for the **ADI toy environment** and should not be interpreted as general-purpose “enterprise security” models.

#### Requirements
- Test CSV file must be prepared beforehand with columns: `id` and `input`
- The input CSV must use ASCII character 30 (Record Separator) as the delimiter
- The input format should include `----- USER QUERY -----` markers for query extraction

#### Public LoRA adapters (Hugging Face Hub)

Pre-trained adapters are hosted on the HF Hub and can be loaded directly:

- **Qwen3-8B (4-bit base):** `SisWiss/qwen3-8b-sa-lora`
- **Qwen3-14B (4-bit base):** `SisWiss/qwen3-14b-sa-lora`


#### Usage

**A) Basic usage (base model only):**
```bash
python unsloth_adi.py \
  --inputs path/to/test_data.csv \
  --model unsloth/qwen3-8b-unsloth-bnb-4bit
````

**B) Using the public LoRA adapter from the Hub (recommended):**

```bash
python unsloth_adi.py \
  --inputs path/to/test_data.csv \
  --model unsloth/qwen3-8b-unsloth-bnb-4bit \
  --adapter_repo SisWiss/qwen3-8b-sa-lora \
  --identifier sa-lora
```

Why `--identifier`?
By default, output directories are named only by the base model. If you evaluate both base-only and base+adapter on the same input file, add `--identifier` to avoid overwriting outputs.

**Output directory for the command above:**

* Defaults to the input CSV’s parent directory, i.e. `path/to/models/`
* With `--identifier sa-lora`, the final folder becomes:

  * `path/to/models/qwen3-8b-unsloth-bnb-4bit-sa-lora/`

**C) Using a local LoRA adapter directory (e.g., from your own fine-tuning run):**

```bash
python unsloth_adi.py \
  --inputs path/to/test_data.csv \
  --model unsloth/qwen3-8b-unsloth-bnb-4bit \
  --adapter_repo /path/to/lora_adapter_dir \
  --identifier local-adapter
```

> Note: `--adapter_repo` accepts either a Hub repo id (e.g. `SisWiss/qwen3-8b-sa-lora`) **or** a local path.
> (The legacy `--adapter_dir` flag still exists for backward compatibility.)

**D) Using a specific checkpoint directory:**

```bash
python unsloth_adi.py \
  --inputs path/to/test_data.csv \
  --checkpoint_path trainer_output_xxx/checkpoint-200
```

**E) Using trainer output directory + checkpoint step:**

```bash
python unsloth_adi.py \
  --inputs path/to/test_data.csv \
  --trainer_output_dir trainer_output_xxx \
  --checkpoint_step 200
```

**F) With custom output directory and identifier:**

```bash
python unsloth_adi.py \
  --inputs path/to/test_data.csv \
  --model unsloth/qwen3-8b-unsloth-bnb-4bit \
  --adapter_repo SisWiss/qwen3-8b-sa-lora \
  --output_dir /path/to/output \
  --identifier experiment1
```

**G) Continue from a specific line (useful for resuming):**

```bash
python unsloth_adi.py \
  --inputs path/to/test_data.csv \
  --model unsloth/qwen3-8b-unsloth-bnb-4bit \
  --adapter_repo SisWiss/qwen3-8b-sa-lora \
  --identifier sa-lora \
  --continue_at 100
```

#### Arguments

| Argument               | Type  | Default                             | Description                                                                                                          |
| ---------------------- | ----- | ----------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| `--inputs`             | str   | **required**                        | Path to the input CSV file (test data prepared beforehand)                                                           |
| `--continue_at`        | int   | `0`                                 | Line number to continue processing from (for resuming)                                                               |
| `--to_drop`            | str   | `../2025_datasets/0/to_drop.txt`    | Path to file containing IDs to exclude                                                                               |
| `--model`              | str   | `unsloth/qwen3-8b-unsloth-bnb-4bit` | **Base model id** to load (used unless a full checkpoint is loaded)                                                  |
| `--adapter_repo`       | str   | `None`                              | LoRA adapter source: either a **HF Hub repo id** (e.g., `SisWiss/qwen3-8b-sa-lora`) or a **local adapter directory** |
| `--temperature`        | float | `0.6`                               | Sampling temperature for generation                                                                                  |
| `--max_new_tokens`     | int   | `2048`                              | Maximum number of tokens to generate                                                                                 |
| `--input_batch_exists` | bool  | `False`                             | Whether the input batch JSONL already exists                                                                         |
| `--output_dir`         | str   | `None`                              | Custom directory to save outputs (defaults to input file's parent)                                                   |
| `--checkpoint_path`    | str   | `None`                              | Path to a specific checkpoint directory (e.g., `trainer_output_xxx/checkpoint-200`)                                  |
| `--trainer_output_dir` | str   | `None`                              | Training output directory containing checkpoints                                                                     |
| `--checkpoint_step`    | int   | `None`                              | Checkpoint step number (used with `--trainer_output_dir`)                                                            |
| `--adapter_dir`        | str   | `None`                              | Legacy: local directory with LoRA adapters (prefer `--adapter_repo`)                                                 |
| `--identifier`         | str   | `None`                              | Additional identifier appended to the output directory name (recommended when comparing runs)                        |

#### Model Loading Priority

The script loads models in the following priority order:

1. `--checkpoint_path` (if specified)
2. `--trainer_output_dir` + `--checkpoint_step` (if both specified)
3. Base model `--model`, then attach adapter from:

   * `--adapter_repo` (Hub id or local path), otherwise
   * `--adapter_dir` (legacy local path)
4. `--model` only (base model)

#### Output

The script creates output files in `{output_dir}/models/{model_name}/`:

* `batch_input.jsonl`: preprocessed input data
* `batch_output.jsonl`: raw model outputs
* `batch_output.csv`: clean output CSV (if you post-process it)

The `model_name` is constructed from:

* base model name (e.g., `qwen3-8b-unsloth-bnb-4bit`)
* checkpoint step (if using checkpoints)
* `--identifier` (if provided)

---

## Example Workflow

1. **Training data (`processed_dataset.csv`)**
   - The `processed_dataset.csv` in our [Zenodo repository](https://doi.org/10.5281/zenodo.18412000) is already prepared and ready to use. It builds upon the original data from the ACCESS DENIED INC experiments and contains the correct outputs of the models assessed in the original paper.
   - **Data composition**: Roughly 75% of the examples are chain-of-thought (CoT) outputs and the remaining 25% are non-CoT outputs.

2. **Fine-tune the model:**
   ```bash
   python qwen3_finetuning_vldb.py --local_name my_finetuned_model
   ```

3. **Prepare test data for inference:**
   - **Setup ADI codebase**: Clone and install the [ACCESS DENIED INC](https://github.com/DrenFazlija/AccessDeniedInc) repository.
   - **Modify `adult_transformation.py`**: In the ACCESS DENIED INC repo, open `adult_transformation.py` and change line 217 to `seed = 3` (it was `0` in the original version).
   - **Generate questionnaires**: Run `questionnaire.py` three times with different seeds to generate the test datasets:
     ```bash
     python questionnaire.py --seed 0
     python questionnaire.py --seed 1
     python questionnaire.py --seed 2
     ```
   - The resulting questionnaire data files are the test inputs that will be passed to `unsloth_adi.py`. These CSV files should have `id` and `input` columns (using ASCII character 30 as the delimiter) and contain `----- USER QUERY -----` markers in the input format.

4. **Run inference:**
   ```bash
   python unsloth_adi.py \
       --inputs path/to/test_data.csv \
       --adapter_dir my_finetuned_model
   ```

## Downstream Task Assessment

We evaluate downstream performance using the **LM-Evaluation-Harness** framework [`EleutherAI/lm-evaluation-harness`](https://github.com/EleutherAI/lm-evaluation-harness/tree/main) on several standard benchmarks.

### Environment for AutoEval

- We provide a separate conda environment file: `autoeval_env.yml`.
- Create and activate this environment (from the lm-eval codebase or this repo, depending on where you keep the env file):

```bash
conda env create -f autoeval_env.yml
conda activate autoeval
```

Then install and set up `lm-evaluation-harness` following the official instructions in [`EleutherAI/lm-evaluation-harness`](https://github.com/EleutherAI/lm-evaluation-harness/tree/main).

### Reproducing Our Downstream Comparisons

Below we show the commands used to compare the **base Qwen3-8B** model and our **LoRA-adapted Qwen3-8B (`qwen3_8b_4bit`)** on several tasks. All commands assume:
- You have `lm_eval` on your `PATH` (from `lm-evaluation-harness`).
- You run them from a directory where `qwen3_8b_4bit/` is available (this repo).

#### BBH (Big-Bench Hard)

```bash
# Baseline Qwen3-8B (optional, for comparison)
lm_eval --model hf \
    --model_args pretrained=unsloth/Qwen3-8B,trust_remote_code=True,load_in_4bit=True \
    --tasks bbh \
    --batch_size auto \
    --output_path llm-autoeval_results/bbh_qwen3-8b

# Qwen3-8B with our LoRA adapter
lm_eval --model hf \
    --model_args pretrained=unsloth/Qwen3-8B,trust_remote_code=True,load_in_4bit=True,peft=qwen3_8b_4bit \
    --tasks bbh \
    --batch_size auto \
    --output_path llm-autoeval_results/bbh_qwen3-8b-lora-final
```

#### IfEval

```bash
# Baseline Qwen3-8B (optional, for comparison)
lm_eval --model hf \
    --model_args pretrained=unsloth/Qwen3-8B,trust_remote_code=True,load_in_4bit=True \
    --tasks ifeval \
    --batch_size auto \
    --output_path llm-autoeval_results/ifeval_qwen3-8b

# Qwen3-8B with our LoRA adapter
lm_eval --model hf \
    --model_args pretrained=unsloth/Qwen3-8B,trust_remote_code=True,load_in_4bit=True,peft=qwen3_8b_4bit \
    --tasks ifeval \
    --batch_size auto \
    --output_path llm-autoeval_results/ifeval_qwen3-8b-lora-final
```

#### GSM8K-Platinum

```bash
# Baseline Qwen3-8B (optional, for comparison)
lm_eval --model hf \
    --model_args pretrained=unsloth/Qwen3-8B,trust_remote_code=True,load_in_4bit=True \
    --tasks gsm8k_platinum \
    --batch_size auto \
    --output_path llm-autoeval_results/gsm8k_platinum_qwen3-8b

# Qwen3-8B with our LoRA adapter
lm_eval --model hf \
    --model_args pretrained=unsloth/Qwen3-8B,trust_remote_code=True,load_in_4bit=True,peft=qwen3_8b_4bit \
    --tasks gsm8k_platinum \
    --batch_size auto \
    --output_path llm-autoeval_results/gsm8k_platinum_qwen3-8b-lora-final
```

## Citation
If you use this code in your research, please cite our paper:
```bibtex
@inproceedings{
fazlija2026towards,
title={Towards Sensitivity-Aware Language Models},
author={Dren Fazlija and Iyiola E. Olatunji and Daniel Kudenko and Sandipan Sikdar},
booktitle={The 29th International Conference on Artificial Intelligence and Statistics},
year={2026},
}
```
