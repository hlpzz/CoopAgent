# CoopAgent

**CoopAgent: Training Multi-Agent Cooperation from Single-Agent Execution Trajectories** (EMNLP 2026)

CoopAgent learns multi-agent cooperation from successful single-agent atomic-task trajectories. Training consists of supervised fine-tuning (SFT), followed by Perturbation-Aligned Direct Preference Optimization (PADPO).

This repository contains the data-construction and training code. Minecraft task generation and trajectory collection rely on [VillagerAgent/VillagerBench](https://github.com/cnsdqd-dyb/VillagerAgent-Minecraft-multiagent-framework).

## Repository structure

```text
CoopAgent/
|-- data_pipeline/
|   |-- build_sft_data.py
|   |-- build_preference_data.py
|   `-- prompt_templates.py
|-- training/
|   |-- train_sft.py
|   |-- train_padpo.py
|   `-- padpo_trainer.py
|-- requirements.txt
`-- README.md
```

## Quickstart

The workflow has two stages: collect trajectories and construct datasets, then run SFT and PADPO training. Separate Python environments are recommended for VillagerAgent and CoopAgent.

### 1. Generate single-agent trajectories

First, clone VillagerAgent and complete its environment, Minecraft 1.19.2 server, and API setup. Verify that its Quickstart works before continuing.

```bash
git clone https://github.com/cnsdqd-dyb/VillagerAgent-Minecraft-multiagent-framework.git
cd VillagerAgent-Minecraft-multiagent-framework

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
npm install
python js_setup.py
```

Use the activation command appropriate for your operating system. VillagerAgent also requires Node.js, Java, a reachable Minecraft server, and suitable player permissions.

Create `API_KEY_LIST` in the VillagerAgent root:

```json
{
  "AGENT_KEY": ["your_api_key_here"]
}
```

Do not commit this file. If necessary, update the API endpoint and model settings in VillagerAgent's `config.py` and `start_with_config.py`.

Generate single-agent meta-task configurations:

```bash
python config.py \
  --task meta \
  --meta_task_num 1000 \
  --api_model qwen3-next-80b-a3b-instruct \
  --host 127.0.0.1 \
  --port 25565 \
  --agent_num 1
```

Adjust the model, task count, host, and port as needed. The example creates:

```text
qwen3_next_80b_a3b_instruct_launch_config_meta.json
```

Set the launch-configuration path in `start_with_config.py` to the generated JSON file, then run:

```bash
python start_with_config.py
```

Results are written under `result/<task_name>/`. The SFT builder retains tasks with `score.json`, `Alice_history.json`, and `config.json` whose numeric score is at least `50`.

### 2. Construct the SFT and PADPO datasets

Create a separate environment for this repository:

```bash
git clone https://github.com/hlpzz/CoopAgent.git
cd CoopAgent

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

In `data_pipeline/build_sft_data.py`, set:

```python
RESULTS_DIR = Path("/path/to/VillagerAgent-Minecraft-multiagent-framework/result")
SFT_OUTPUT_PATH = Path("/path/to/CoopAgent/outputs/sft_dataset.json")
TRAJECTORY_OUTPUT_PATH = Path("/path/to/CoopAgent/outputs/trajectory_data.json")
```

Then build the SFT data and processed trajectories:

```bash
python -m data_pipeline.build_sft_data
```

Next, configure `data_pipeline/build_preference_data.py`:

```python
VILLAGERAGENT_ROOT = Path("/path/to/VillagerAgent-Minecraft-multiagent-framework")
INPUT_DATA_PATH = Path("/path/to/CoopAgent/outputs/trajectory_data.json")
OUTPUT_DATA_PATH = Path("/path/to/CoopAgent/outputs/preference_dataset.json")
NOISE_CACHE_PATH = Path("/path/to/CoopAgent/outputs/noise_cache.json")

API_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
PERTURBATION_MODEL = "qwen3-next-80b-a3b-instruct"
PERTURBATION_MODE = "task_related"
```

`task_related` is used by the main method; `random` is retained for the random-perturbation ablation and reads `data/blocks.json` from VillagerAgent.

Set the API key and build the preference data:

```bash
export DASHSCOPE_API_KEY=your_api_key_here
python -m data_pipeline.build_preference_data
```

The output contains `prompt`, `noise_prompt`, `chosen`, and `rejected`. Generated perturbations are cached at `NOISE_CACHE_PATH`, allowing interrupted runs to resume without repeating completed API calls.

### 3. Train CoopAgent

Prepare a Hugging Face causal language model. A local model directory must include its weights, configuration, and tokenizer files. Use the same base model for both training stages.

For SFT, set these values in `training/train_sft.py`:

```python
MODEL_PATH = "path/to/base_model_or_huggingface_model_id"
TRAIN_DATA_PATH = "path/to/sft_dataset.json"
OUTPUT_DIR = "path/to/sft_output"
```

Run:

```bash
python -m training.train_sft
```

For PADPO, set these values in `training/train_padpo.py`:

```python
MODEL_PATH = "path/to/the_same_base_model"
ADAPTER_PATH = "path/to/sft_output/checkpoint-N"
TRAIN_DATA_PATH = "path/to/preference_dataset.json"
OUTPUT_DIR = "path/to/padpo_output"
```

`ADAPTER_PATH` must point to the SFT LoRA checkpoint containing `adapter_config.json`. Run:

```bash
python -m training.train_padpo
```

For multi-GPU training, replace either command with, for example:

```bash
torchrun --nproc_per_node=8 -m training.train_sft
torchrun --nproc_per_node=8 -m training.train_padpo
```

Adjust batch size, sequence length, precision, distributed settings, and `save_steps` for your hardware and dataset size.

## Acknowledgements

CoopAgent builds on [VillagerAgent/VillagerBench](https://github.com/cnsdqd-dyb/VillagerAgent-Minecraft-multiagent-framework) for Minecraft task generation, environment execution, and evaluation.
