# Known risks before public release

This document records limitations identified during the initial repository audit. Items in the first section should be resolved before declaring the repository paper-reproducible.

## Release blockers

| Risk | Why it matters | Recommended mitigation |
| --- | --- | --- |
| The augmentation script currently defaults to random perturbations. | The paper's main method uses task-related two-step perturbations; users may accidentally reproduce the ablation instead. | Add a required `--perturbation-type` option and make the paper configuration explicit. |
| Training paths are hard-coded to the original server filesystem. | Model, dataset, adapter, log, and output paths do not exist for other users. | Replace constants with dataclass or argparse configuration and provide checked example configs. |
| The SFT and preference branches are selected by commented code. | Reproduction currently requires source editing and can produce the wrong dataset. | Add explicit subcommands or `--mode sft|preference`. |
| Dependency versions are not yet pinned. | TRL, Transformers, and PEFT APIs change frequently, particularly custom `DPOTrainer` hooks and data collators. | Export the successful server environment and publish a tested lock file or exact version constraints. |
| Reported dataset counts have not been regenerated from the cleaned public pipeline. | Local artifacts do not yet provide a single verified path to the paper's 1,039 trajectories, 6,380 SFT examples, and 995 preference pairs. | Regenerate from scratch, record checksums/counts, and add an automated dataset validation command. |
| The Qwen2.5 PADPO launcher refers to a dataset filename that is not present in the repository. | The documented training command cannot run without an externally prepared file. | Standardize output filenames between the data builder and trainer, then provide a small schema-valid example. |

## Implementation-alignment risks

- The PADPO trainer detaches the original-prompt advantage before applying the consistency term. Consequently, the regularizer's gradient flows through the perturbed-prompt branch only, whereas the paper equation is written as a symmetric squared difference. Confirm that this is the intended released implementation or change the code and rerun the relevant experiments.
- The task-relevant tool mapping is hard-coded. Any VillagerAgent tool rename, result-schema change, or new atomic task can change which actions become positive and negative pairs.
- The converter uses `score >= 50` as the success criterion. This must be validated against the binary success definition used for the released dataset.
- The current code contains both a one-step perturbation path and a two-step generation/injection path. Retain one canonical implementation for the main method and place ablations behind named options.

## Reproducibility and operational risks

- Name replacement, tool-order shuffling, task generation, and perturbation sampling are stochastic but do not currently expose or record a global seed.
- Dataset discovery depends on the current working directory and fixed intermediate filenames such as `temp_output.json`.
- Training was performed on a GPU server. CUDA, GPU memory, distributed launch settings, and tested hardware are not yet documented.
- The custom trainer relies on internal TRL behavior and may fail with newer releases even if imports succeed.
- Minecraft execution is nondeterministic and also depends on server state, world configuration, network latency, and external LLM responses.
- Perturbation augmentation makes paid external API calls. Retry loops can increase cost and may partially overwrite intermediate state after interruption.

## Security and release-management risks

- VillagerAgent uses an `API_KEY_LIST` file, while CoopAgent uses `DASHSCOPE_API_KEY`. The separation is safer for this repository but can confuse users. Documentation and example files must never contain real credentials.
- Raw trajectories may contain prompts, model outputs, player names, host information, or other environment metadata. Review and sanitize any dataset before publishing it.
- Model checkpoints and generated datasets can be very large. Do not commit them directly to Git; publish them through an appropriate model/dataset host with checksums and licenses.
- The repository does not yet include a LICENSE, citation metadata, or an explicit statement describing the license boundary with VillagerAgent. Resolve these before public release.

