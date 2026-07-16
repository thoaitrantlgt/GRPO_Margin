# Adaptive Midpoint GRPO

Reference implementation of Boundary-Margin reweighting on top of Hugging Face TRL `GRPOTrainer`.

The implementation keeps TRL's generation, reward aggregation, group normalization, clipping and loss intact. It only
reweights the advantages returned by TRL. The default experiment uses eight completions per prompt.

## Method

For accuracy rewards `a_i` in one completion group:

```text
p = mean(a_i)
g = 4 * p * (1 - p)
b = 0.5 * (max(r) + min(r))
q_i = exp(-abs(r_i - b) / boundary_bandwidth)
w_i = gate_floor + (1 - gate_floor) * g + alpha * g * q_i
A_i = clip(w_i * A_i_GRPO, -advantage_clip, advantage_clip)
```

The default combined reward is `accuracy + 0.1 * format`. Groups with fewer than two valid accuracy rewards and
zero-variance groups fall back to plain GRPO.

## Hardware profiles

- Local GTX 1650 4 GB: pure unit tests and code development only.
- Training: Linux, CUDA and one NVIDIA GPU with 24 GB VRAM.
- Default model: `Qwen/Qwen2.5-1.5B-Instruct` at pinned revision
  `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`.
- Default rollout group: `num_generations=8` with `gradient_accumulation_steps=8`.
- vLLM is optional for Linux GPU training. Distributed training remains out of scope for v1.

## Installation

Install a PyTorch CUDA build compatible with the remote machine first, then:

```bash
cd code_grpo_margin
python -m venv .venv
source .venv/bin/activate
pip install -e ".[train,test]"
```

For vLLM-accelerated GRPO generation on Linux:

```bash
pip uninstall -y vllm
pip install -e ".[train,test,vllm]"
```

The code pins `trl==1.6.0`. `BoundaryMarginGRPOTrainer` intentionally fails with another TRL version because it uses
the private reward and generation hooks.

## Dataset pipeline

Smoke profile:

```bash
python -m bm_grpo.data.prepare --config configs/data/smoke.yaml
python -m bm_grpo.data.audit --manifest data/processed/smoke/manifest.json
```

Full GSM8K and paper profiles:

```bash
python -m bm_grpo.data.prepare --config configs/data/gsm8k.yaml
python -m bm_grpo.data.audit --manifest data/processed/gsm8k/manifest.json

python -m bm_grpo.data.prepare --config configs/data/paper.yaml
python -m bm_grpo.data.audit --manifest data/processed/paper/manifest.json
```

The pipeline pins every dataset revision, normalizes all datasets to the same conversational schema, verifies gold
answers, removes overlength prompts, performs exact and MinHash near-duplicate filtering, materializes Parquet and
writes checksums to `manifest.json`.

## Training

Smoke test on a 24 GB GPU:

```bash
accelerate launch \
  --config_file configs/accelerate/single_gpu.yaml \
  -m bm_grpo.train \
  --config configs/train/smoke_boundary.yaml
```

GSM8K controlled comparison:

```bash
accelerate launch --config_file configs/accelerate/single_gpu.yaml \
  -m bm_grpo.train --config configs/train/gsm8k_grpo.yaml

accelerate launch --config_file configs/accelerate/single_gpu.yaml \
  -m bm_grpo.train --config configs/train/gsm8k_boundary.yaml
```

Resume a paper run:

```bash
accelerate launch --config_file configs/accelerate/single_gpu.yaml \
  -m bm_grpo.train \
  --config configs/train/paper_boundary_seed42.yaml
```

If `outputs/<name>/resolved_config.yaml` matches the current config, `bm_grpo.train` automatically resumes from the
latest `checkpoint-*` directory in that output folder. You can still override the checkpoint manually with
`--resume-from` if you really need to.

Every run stores its resolved config, Python/package/CUDA/GPU environment, adapter checkpoints, completion tables and
training metrics.

Training completions are written under `outputs/<run>/completions/`. To inspect recent parses:

```bash
python - <<'PY'
import pandas as pd
path = "outputs/paper_qwen3_4b_instruct_2507_boundary_seed42/completions/completions_00001.parquet"
df = pd.read_parquet(path)
print(df[["completion", "parsed_answer", "format_valid"]].head(4).to_string())
PY
```

To smoke-test generation outside TRL before a run:

```bash
python -m bm_grpo.smoke_generate --config configs/train/paper_qwen3_4b_boundary_seed42.yaml
```

## Experiment matrix

Inspect the commands without launching jobs:

```bash
python -m bm_grpo.experiments --matrix configs/experiments/main.yaml --dry-run
python -m bm_grpo.experiments --matrix configs/experiments/ablations.yaml --dry-run
```

Run the three-seed GRPO/Boundary-Margin matrix:

```bash
python -m bm_grpo.experiments --matrix configs/experiments/main.yaml
```

Completed runs are skipped when `train_metrics.json` exists. Pass `--force` to rerun them.

## Train baseline and compare automatically

The paired runner verifies that GRPO and Boundary-Margin use the same model, data, rewards, seed, optimizer, group
size and rollout budget. It then trains baseline first, trains Boundary-Margin, evaluates both checkpoints and writes
`comparison.json` plus `comparison.md`.

If either output folder already contains checkpoints and a matching resolved config, each train stage resumes from the
latest checkpoint automatically before continuing.

Inspect the smoke commands:

```bash
python -m bm_grpo.compare --config configs/compare/smoke.yaml --dry-run
```

Run the complete smoke comparison on a 24 GB GPU:

```bash
python -m bm_grpo.compare --config configs/compare/smoke.yaml
```

Run the 500-step GSM8K MVP comparison:

```bash
python -m bm_grpo.compare --config configs/compare/gsm8k.yaml
```

Run the paper seed-42 comparison after preparing the paper dataset:

```bash
python -m bm_grpo.compare --config configs/compare/paper_seed42.yaml
```

Run the Qwen3-4B Instruct 2507 paper comparison:

```bash
pip install -e ".[train,test]"
python -m bm_grpo.data.prepare --config configs/data/paper_qwen3_4b.yaml
python -m bm_grpo.data.audit --manifest data/processed/paper_qwen3_4b_instruct_2507/manifest.json
python -m bm_grpo.compare --config configs/compare/paper_qwen3_4b_seed42.yaml
```

The Qwen3 data config also materializes MathArena AIME 2025 and AIME 2026 eval sets:

```text
data/processed/paper_qwen3_4b_instruct_2507/aime25.parquet
data/processed/paper_qwen3_4b_instruct_2507/aime26.parquet
```

To run only the missing Qwen3 ablations for seed 42, including train/resume, evaluation on GSM8K, MATH-500,
AIME24, AIME25 and AIME26, and a final summary report:

```bash
python -m bm_grpo.experiments --matrix configs/experiments/ablations_qwen3_4b.yaml
```

This matrix is organized around the paper table:

```text
standard_grpo
gate_only
midpoint_only
full_bm_grpo
```

The existing GRPO and Full AM-GRPO checkpoints are mapped into the matrix but marked `skip_train: true` and
`skip_eval: true`, so the command above only runs Gate-only and Midpoint-only. If their old metrics exist, they are
still included in the generated summary table.

If all ablation checkpoints already exist and you only want to run/re-run evaluation plus the summary report, use:

```bash
python -m bm_grpo.experiments --matrix configs/experiments/ablations_qwen3_4b.yaml --eval-only
```

In eval-only mode, variants without an existing `final_adapter` or `checkpoint-*` are skipped by default. Add
`--strict-missing-checkpoints` if missing checkpoints should be treated as an error.

The ablation runner writes generated configs under `configs/experiments/generated/ablations_qwen3_4b/`, per-run
outputs under `outputs/paper_qwen3_4b_*`, and the aggregate report to
`outputs/ablations/ablations_qwen3_4b/ablation_summary.md` plus a root-level `ablation_summary.md` copy.

The Qwen3-4B Instruct 2507 train configs use the Transformers generation path by default because the current
TRL/vLLM rollout path produced whitespace-only completions in smoke tests. They pass
`chat_template_kwargs: {enable_thinking: false}` and conservative sampling (`temperature: 0.7`, `top_p: 0.9`,
`top_k: 50`, `min_p: 0.02`, `repetition_penalty: 1.08`). Re-enable vLLM only after a direct vLLM smoke test produces
non-empty `\boxed{...}` answers.

If training and evaluation already finished, rebuild only the comparison report:

```bash
python -m bm_grpo.compare --config configs/compare/smoke.yaml --report-only
```

Outputs are written under `outputs/comparisons/<pair-name>/`. Positive evaluation deltas mean Boundary-Margin is
better; runtime and memory deltas are reported as raw method minus baseline.

## Evaluation

```bash
python -m bm_grpo.evaluate \
  --config configs/eval/paper.yaml \
  --checkpoint outputs/paper_boundary_seed42/final_adapter
```

Evaluate the untrained/base Qwen model without loading any LoRA adapter:

```bash
python -m bm_grpo.evaluate \
  --config configs/eval/paper_qwen3_4b.yaml \
  --base-only \
  --output-dir outputs/qwen3_4b_base_eval
```

Evaluation writes completion-level Parquet and reports greedy pass@1, sampled pass@4, completion accuracy, format rate
and bootstrap 95% confidence intervals for GSM8K, MATH-500 and AIME24.

For faster evaluation on larger GPUs, increase `generation.*.batch_size` in the eval config. You can add a global
`limit: 100` or a per-dataset mapping such as `gsm8k: {path: data/processed/paper/gsm8k_test.parquet, limit: 100}` for
quick checks.

## Tests

```bash
pytest -q
```

The local suite covers config validation, group-size-eight margin behavior, NaN/fallback paths, advantage clipping,
boxed-answer parsing, dataset adapters, matrix expansion and the TRL version guard. Actual QLoRA smoke training remains
a GPU integration gate and is not run on the 4 GB development machine.
