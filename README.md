# CoopAgent

CoopAgent is a two-stage training framework for improving multi-agent cooperation using only single-agent execution trajectories. It first learns atomic agent behavior through LoRA supervised fine-tuning (SFT), then applies Perturbation-Aligned Direct Preference Optimization (PADPO) to keep action preferences stable under task-related observation changes.

This repository contains only the CoopAgent data-construction and training code. Minecraft execution, atomic-task generation, trajectory collection, and evaluation are provided by the external VillagerAgent/VillagerBench environment.

## Repository layout

```text
CoopAgent/
├── data_pipeline/
│   ├── augment_preference_data.py
│   ├── build_training_data.py
│   └── perturbation_prompts.py
├── training/
│   ├── padpo_trainer.py
│   ├── train_padpo_qwen2_5.py
│   └── train_sft_qwen2_5.py
├── docs/
│   ├── DATA_PIPELINE.md
│   └── KNOWN_RISKS.md
└── requirements.txt
```

## Installation boundary

CoopAgent intentionally does not vendor VillagerAgent or VillagerBench. There are two separate setup steps:

1. Install VillagerAgent/VillagerBench to create Minecraft tasks and collect trajectories.
2. Install this repository's Python dependencies to construct datasets and train CoopAgent.

The local compatibility audit for this release used VillagerAgent commit `7edb5c6659f3ab02f850844ca78d5761a85dff5e`. Pin this revision until compatibility with a newer revision has been verified.

## Quickstart

### 1. Install VillagerAgent/VillagerBench

Follow the upstream [VillagerAgent repository](https://github.com/cnsdqd-dyb/VillagerAgent-Minecraft-multiagent-framework) first. Its setup includes:

- Python 3.8 or newer;
- Node.js and npm;
- a reachable Minecraft 1.19.2 server;
- the upstream Python and npm dependencies;
- `npm install` and `python js_setup.py`;
- an `API_KEY_LIST` file in the VillagerAgent root, as required by its Quickstart.

For a pinned checkout:

```bash
git clone https://github.com/cnsdqd-dyb/VillagerAgent-Minecraft-multiagent-framework.git
cd VillagerAgent-Minecraft-multiagent-framework
git checkout 7edb5c6659f3ab02f850844ca78d5761a85dff5e
pip install -r requirements.txt
npm install
python js_setup.py
```

Start the Minecraft server and verify the upstream minimal example before continuing. CoopAgent does not replace or diagnose the Minecraft-side setup.

### 2. Install CoopAgent dependencies

From the CoopAgent repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 3. Configure the data pipeline

The data pipeline needs the installed VillagerAgent repository and an API key for perturbation generation. The same DashScope/Qwen key may be used for VillagerAgent and CoopAgent, but CoopAgent reads it from an environment variable rather than from a tracked file.

Linux or macOS:

```bash
export VILLAGERAGENT_ROOT=/absolute/path/to/VillagerAgent-Minecraft-multiagent-framework
export DASHSCOPE_API_KEY=your_key_here
```

Windows PowerShell:

```powershell
$env:VILLAGERAGENT_ROOT = "C:\path\to\VillagerAgent-Minecraft-multiagent-framework"
$env:DASHSCOPE_API_KEY = "your_key_here"
```

Never commit either the upstream `API_KEY_LIST` file or a local `.env` file.

### 4. Generate and execute single-agent atomic tasks

Run task generation from the VillagerAgent root:

```bash
python config.py \
  --task meta \
  --meta_task_num 1000 \
  --api_model qwen3-next-80b-a3b-instruct \
  --host localhost \
  --port 25565 \
  --agent_num 1
```

This creates a launch-config JSON file. Set `CONFIG_PATH` near the top of VillagerAgent's `start_with_config.py` to that file, then execute:

```bash
python start_with_config.py
```

Completed runs are written under VillagerAgent's `result/<task_name>/` directories. CoopAgent expects each usable run to contain at least:

- `score.json`;
- `Alice_history.json`;
- `config.json`.

See [docs/DATA_PIPELINE.md](docs/DATA_PIPELINE.md) for the dataset schemas and the current execution constraints.

### 5. Train CoopAgent

The training scripts are lightweight wrappers around Transformers, PEFT, and TRL. The current revision still stores model, dataset, adapter, and output paths as constants inside the scripts. Update those paths before execution.

Stage 1, LoRA SFT:

```bash
python -m training.train_sft_qwen2_5
```

Stage 2, PADPO initialized from the SFT adapter:

```bash
python -m training.train_padpo_qwen2_5
```

The training scripts were originally executed on a GPU server and are not expected to run on a typical local machine without adapting the model paths, CUDA environment, and distributed-training configuration.

## Current status

The repository structure and import paths have been cleaned, but the command-line interfaces and paper-exact data flow are still being consolidated. Read [docs/KNOWN_RISKS.md](docs/KNOWN_RISKS.md) before treating this revision as reproducible release code.

