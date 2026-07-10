## Development And Training Workflow

- This host is for codebase development only.
- Until training starts, write and verify code locally, but do not run real training, evaluation, or model downloads on this host.
- Company compute resources are accessed through the bastion/head node.
- Real training, evaluation, dataset/model download, checkpointing, and large artifact generation must run only on company resources: 10 nodes / H100 x80.
- Training and evaluation code/configs must assume DDP/multi-node execution by default where applicable, so company GPU resources can be used fully instead of single-process fallbacks.

## Company Remote Environment

### Access And Network

- Bastion / ProxyJump host: `sr-gpu-bastion`
  - Access uses the company network and certificate-based ProxyJump configuration.
  - Reuse the operator's existing SSH configuration; never commit certificates,
    private keys, tokens, or SSH options containing credentials.
- Slurm submission and control head node: `sr-gpu-head`
- All compute access must pass through the bastion and head node.

### Compute And Slurm

- GPU nodes: `gpu-vtt-queue-st-p5-node-[1-10]`
- GPU capacity: 8 x H100 80GB per node, 80 GPUs total.
- Slurm partition: `gpu-vtt-queue`
- Slurm binaries: `/opt/slurm/bin/`
- CPU preprocessing and parallel merge nodes:
  `cpu-prepro-queue-02-dy-m6i-node-[1-6]`
- Use GPU nodes for model inference/training and GPU-bound memory construction.
- Use CPU nodes for ingest, validation, retrieval preprocessing, JSONL merge,
  metric aggregation, and report generation where GPU execution is unnecessary.

### Data And Repository

- Prepared dataset root:
  `/groups/VTteam/datasets/SuperMemory-VQA/ingested/`
- Required prepared files already present:
  - `sources.jsonl`
  - `questions.jsonl`
  - `labels.jsonl`
- Remote code repository:
  `/repo/VTteam/bongh.park/worldmm-smvqa-gemma4-e2b`
- Python virtual environment:
  `/repo/VTteam/bongh.park/worldmm-smvqa-gemma4-e2b/.venv`
- Remote Python version: 3.13.

### Models And Outputs

- Gemma model path: `/repo/VTteam/bongh.park/gemma-4-e2b-it`
- Output base path: `/repo/VTteam/bongh.park/outputs`
- Every run must use a unique run-scoped output directory such as
  `/repo/VTteam/bongh.park/outputs/$WORLDMM_RUN_ID`; do not write multiple
  experiments directly into the output base path.
- Keep datasets, model files, embeddings, evidence packs, predictions, logs,
  metrics, diagnostics, and checkpoints on company storage.

### Runtime Environment Mapping

- `BASTION_HOST=sr-gpu-bastion`
- `HEAD_NODE=sr-gpu-head`
- `SMVQA_DATA_ROOT=/groups/VTteam/datasets/SuperMemory-VQA/ingested/`
- `WORLDMM_REMOTE_REPO=/repo/VTteam/bongh.park/worldmm-smvqa-gemma4-e2b`
- `GEMMA_MODEL_PATH=/repo/VTteam/bongh.park/gemma-4-e2b-it`
- `WORLDMM_OUTPUT_ROOT=/repo/VTteam/bongh.park/outputs/$WORLDMM_RUN_ID`
- `WORLDMM_REMOTE_NODES=10`
- `WORLDMM_GPUS_PER_NODE=8`
- Activate the remote virtual environment before invoking project commands:
  `source /repo/VTteam/bongh.park/worldmm-smvqa-gemma4-e2b/.venv/bin/activate`

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
