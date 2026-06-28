# GRPO vs Boundary-Margin: paper_seed42

Positive evaluation delta means Boundary-Margin is better. Training runtime/memory deltas are raw method minus baseline.

## Training

| Metric | GRPO | Boundary-Margin | Delta | Relative |
|---|---:|---:|---:|---:|
| train_loss | 0.000000 | 0.000100 | 0.000100 | 437434.33% |
| train_runtime | 128383.737600 | 44470.660700 | -83913.076900 | -65.36% |
| train_samples_per_second | 0.031000 | 0.090000 | 0.059000 | 190.32% |
| train_steps_per_second | 0.004000 | 0.011000 | 0.007000 | 175.00% |
| peak_gpu_memory_bytes | 11684948480.000000 | 11749511680.000000 | 64563200.000000 | 0.55% |

## Evaluation

### aime24/pass1

| Metric | GRPO | Boundary-Margin | Delta | Relative |
|---|---:|---:|---:|---:|
| pass_at_k | 0.000000 | 0.033333 | 0.033333 | N/A |
| completion_accuracy | 0.000000 | 0.033333 | 0.033333 | N/A |
| format_rate | 0.900000 | 0.833333 | -0.066667 | -7.41% |
| parse_rate | 0.900000 | 0.833333 | -0.066667 | -7.41% |

### aime24/pass4

| Metric | GRPO | Boundary-Margin | Delta | Relative |
|---|---:|---:|---:|---:|
| pass_at_k | 0.033333 | 0.033333 | 0.000000 | 0.00% |
| completion_accuracy | 0.008333 | 0.008333 | 0.000000 | 0.00% |
| format_rate | 0.958333 | 0.966667 | 0.008333 | 0.87% |
| parse_rate | 0.975000 | 0.966667 | -0.008333 | -0.85% |

### gsm8k/pass1

| Metric | GRPO | Boundary-Margin | Delta | Relative |
|---|---:|---:|---:|---:|
| pass_at_k | 0.656558 | 0.626232 | -0.030326 | -4.62% |
| completion_accuracy | 0.656558 | 0.626232 | -0.030326 | -4.62% |
| format_rate | 0.777104 | 0.767248 | -0.009856 | -1.27% |
| parse_rate | 0.977255 | 0.964367 | -0.012889 | -1.32% |

### gsm8k/pass4

| Metric | GRPO | Boundary-Margin | Delta | Relative |
|---|---:|---:|---:|---:|
| pass_at_k | 0.802123 | 0.783169 | -0.018954 | -2.36% |
| completion_accuracy | 0.547195 | 0.512130 | -0.035064 | -6.41% |
| format_rate | 0.728203 | 0.699204 | -0.028999 | -3.98% |
| parse_rate | 0.930819 | 0.908453 | -0.022365 | -2.40% |

### math500/pass1

| Metric | GRPO | Boundary-Margin | Delta | Relative |
|---|---:|---:|---:|---:|
| pass_at_k | 0.419283 | 0.430493 | 0.011211 | 2.67% |
| completion_accuracy | 0.419283 | 0.430493 | 0.011211 | 2.67% |
| format_rate | 0.903587 | 0.923767 | 0.020179 | 2.23% |
| parse_rate | 0.932735 | 0.952915 | 0.020179 | 2.16% |

### math500/pass4

| Metric | GRPO | Boundary-Margin | Delta | Relative |
|---|---:|---:|---:|---:|
| pass_at_k | 0.582960 | 0.596413 | 0.013453 | 2.31% |
| completion_accuracy | 0.366031 | 0.380045 | 0.014013 | 3.83% |
| format_rate | 0.939462 | 0.948430 | 0.008969 | 0.95% |
| parse_rate | 0.978139 | 0.980381 | 0.002242 | 0.23% |
