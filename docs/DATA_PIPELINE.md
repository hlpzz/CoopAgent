# Data construction pipeline

## Scope

CoopAgent does not implement the Minecraft environment. It consumes single-agent atomic-task trajectories produced by VillagerAgent/VillagerBench and converts them into:

- SFT records with `input` and `output` fields;
- PADPO preference records with `prompt`, `noise_prompt`, `chosen`, and `rejected` fields.

## Prerequisites

Before using this pipeline:

1. Install and validate VillagerAgent/VillagerBench.
2. Start a compatible Minecraft 1.19.2 server.
3. Configure the upstream `API_KEY_LIST` file according to the VillagerAgent Quickstart.
4. Set `VILLAGERAGENT_ROOT` to the VillagerAgent checkout.
5. Set `DASHSCOPE_API_KEY` for CoopAgent perturbation generation.

The compatibility audit used VillagerAgent commit `7edb5c6659f3ab02f850844ca78d5761a85dff5e`.

## Stage A: atomic-task configuration

VillagerAgent's `config.py --task meta` samples single-agent atomic tasks and writes a launch-config JSON file. The task taxonomy covers digging, crafting, placing, equipping, movement, and interaction actions.

Example:

```bash
cd "$VILLAGERAGENT_ROOT"
python config.py \
  --task meta \
  --meta_task_num 1000 \
  --api_model qwen3-next-80b-a3b-instruct \
  --host localhost \
  --port 25565 \
  --agent_num 1
```

Set `CONFIG_PATH` in `start_with_config.py` to the generated file.

## Stage B: trajectory collection

From the VillagerAgent root, run:

```bash
python start_with_config.py
```

For every executed task, VillagerAgent writes a result directory. The CoopAgent converters currently depend on the following files and field conventions:

| File | Required content |
| --- | --- |
| `score.json` | A numeric `score` used to select successful executions. |
| `Alice_history.json` | The single-agent interaction chains and action feedback. |
| `config.json` | `task_scenario` and `evaluation_arg` metadata used to determine relevant tools. |

The current converter treats a trajectory as usable when `score >= 50`. Confirm that this threshold is equivalent to the intended binary-success rule for the task set being released.

## Stage C: SFT construction

`data_pipeline/build_training_data.py` contains the SFT conversion functions. The intended processing is:

1. scan successful VillagerAgent result directories;
2. load the action history and task configuration;
3. replace fixed player names with sampled names;
4. remove or collapse unsuitable repeated actions;
5. convert every retained step into an `input`/`output` training record.

SFT schema:

```json
{
  "input": "formatted task, tools, observation, and history",
  "output": "the next action"
}
```

The SFT branch exists in the current source, but the module's present `__main__` block executes the preference-data branch. A command-line switch must be added before the public release so users do not need to edit source comments.

## Stage D: preference-pair construction

For each action in a successful trajectory, the converter determines whether the action is both successful and relevant to the atomic task. A failed or irrelevant action becomes `rejected`; the first subsequent successful, relevant action becomes `chosen`.

Preference schema before augmentation:

```json
{
  "prompt": "the prompt at the rejected action",
  "chosen": "the later successful relevant action",
  "rejected": "the failed or irrelevant action"
}
```

The task-to-tool mapping is currently embedded directly in the converter. Changes to VillagerAgent tool names or task metadata may therefore break extraction silently.

## Stage E: perturbation augmentation

`data_pipeline/perturbation_prompts.py` defines:

- a perturbation-generation prompt that produces an ADD or REMOVE operation;
- a perturbation-injection prompt that changes only the environment observation.

`data_pipeline/augment_preference_data.py` applies perturbations and emits:

```json
{
  "prompt": "original prompt",
  "noise_prompt": "task-related perturbed prompt",
  "chosen": "preferred action",
  "rejected": "dispreferred action"
}
```

The paper method uses task-related two-step perturbation generation and injection. The current script still defaults to a random perturbation configuration in `add_noise()`. Switch this default only after adding an explicit CLI option such as `--perturbation-type task-related|random`; otherwise the paper method and random-perturbation ablation are easy to confuse.

## Current execution constraint

The converters still use the process working directory for result discovery and intermediate filenames. Until path arguments are implemented, the current preference builder must be launched from the VillagerAgent result directory with the CoopAgent root on `PYTHONPATH`.

Linux or macOS example:

```bash
cd "$VILLAGERAGENT_ROOT/result"
PYTHONPATH=/absolute/path/to/CoopAgent \
  python -m data_pipeline.build_training_data
```

This is a transitional interface. The planned public interface should accept explicit `--result-dir`, `--output`, `--mode`, `--seed`, and `--perturbation-type` arguments.

