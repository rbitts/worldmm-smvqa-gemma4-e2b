## Development And Training Workflow

- This host is for codebase development only.
- Until training starts, write and verify code locally, but do not run real training, evaluation, or model downloads on this host.
- Company compute resources are accessed through the bastion/head node.
- Real training, evaluation, dataset/model download, checkpointing, and large artifact generation must run only on company resources: 10 nodes / H100 x80.
- Training and evaluation code/configs must assume DDP/multi-node execution by default where applicable, so company GPU resources can be used fully instead of single-process fallbacks.

## Local Host Rules

- Use this host for:
  - editing code
  - small syntax/type checks
  - unit tests with tiny mock data
  - config/script preparation
  - result inspection
- Do not use this host for:
  - downloading production models
  - downloading large datasets
  - running real training
  - running real evaluation
  - storing company checkpoints, weights, or sensitive artifacts

## Remote Resource Rules

- Access company resources only through the bastion/head node.
- Run model download, training, evaluation, and large-scale preprocessing remotely.
- Store checkpoints, logs, datasets, model weights, and eval artifacts on approved company storage.
- Pull back only lightweight results needed for review:
  - metrics
  - logs
  - plots
  - summaries
  - small sample outputs
- Never copy full datasets, model weights, or sensitive artifacts back to this host unless explicitly approved.

## Agent Workflow

When a task involves training or evaluation:

1. Prepare code, configs, and launch scripts locally.
2. Validate locally only with tiny smoke tests or dry-runs.
3. Provide or execute the remote DDP/multi-node command through the bastion/head node when training/evaluation can use distributed execution.
4. Monitor remote logs and job status.
5. Collect only result summaries back to this host.
6. Report final metrics, artifact paths, failed jobs, and next action.

## Safety Rules

- Do not commit secrets, tokens, SSH keys, internal URLs, or credentials.
- Do not hardcode company paths when an environment variable or config value is enough.
- Ask before launching expensive, long-running, or multi-node jobs.
- Ask before deleting remote checkpoints, datasets, logs, or experiment outputs.
- Prefer reproducible scripts over manual shell history.

## Expected Deliverables

For training/evaluation work, final response must include:

- local code/config changed
- remote command used
- remote job ID or process reference
- result artifact path on company storage
- key metrics or failure reason
- what was not copied locally
