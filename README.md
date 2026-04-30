# kld-sweep

A Python script to compare GGUF quantizations of a model against a baseline (BF16 or F16) using KL Divergence and Perplexity, powered by [llama.cpp](https://github.com/ggerganov/llama.cpp).

## What it does

For each `.gguf` file in a directory, the script runs `llama-perplexity` and measures how much the quantized model's output distribution diverges from the full precision baseline. Results are saved to a CSV and a dual-axis scatter plot (KLD vs MDL_norm / model size) is generated. A ranked markdown report is also produced.

Logits are generated once from the BF16/F16 model and reused for all quants. The sweep resumes automatically if interrupted -- already completed entries in the CSV are skipped.

Compatible with **mainline llama.cpp** and **[ik_llama](https://github.com/ikawrakow/ik_llama.cpp)**. ik_llama returns non-zero exit codes on success -- the script parses output first and uses results if valid, regardless of exit code.

## Metrics

| Metric | Description |
|---|---|
| **PPL** | Perplexity of the quantized model |
| **KLD** | Mean KL Divergence vs the baseline |
| **KLD 99.9%** | 99.9th percentile KLD -- captures tail divergence |
| **BPW** | Bits per weight from model load metadata |
| **MDL_norm** | `Size_GiB x 8 + log2(PPL)` -- bits/token amortised over 1B tokens |

Results are sorted by MDL_norm (lower is better). This metric balances model size against quality in a single number.

## Requirements

Python 3.10+

```
pip install pandas matplotlib adjustText
```

A llama.cpp build with `llama-perplexity` -- download the latest release for your platform from [github.com/ggerganov/llama.cpp/releases](https://github.com/ggerganov/llama.cpp/releases).

## Usage

```
python kld_sweep.py \
--exe /path/to/llama-perplexity \
--baseline /path/to/model-BF16.gguf \
--quants /path/to/quants/ \
--dataset /path/to/dataset.txt \
--output /path/to/output/ \
--model-name MyModel \
--args="-t 7 -c 512 -ngl 99"
```

**Windows:**
```
python kld_sweep.py ^
--exe C:\llamacpp\llama-perplexity.exe ^
--baseline C:\models\MyModel-BF16.gguf ^
--quants C:\models\quants\ ^
--dataset C:\datasets\mydataset.txt ^
--output C:\results\ ^
--model-name MyModel ^
--args="-t 7 -c 512 -ngl 36"
```

**When the baseline doesn't fit in VRAM with the same settings as the quants:**
```
python kld_sweep.py ^
--exe C:\llamacpp\llama-perplexity.exe ^
--baseline C:\models\MyModel-BF16.gguf ^
--quants C:\models\quants\ ^
--dataset C:\datasets\mydataset.txt ^
--output C:\results\ ^
--model-name MyModel ^
--args="-t 7 -c 512 -ngl 99" ^
--args-baseline="-t 7 -c 512 -ngl 20"
```

> **Note:** Always use `--args="-t 7 ..."` with the `=` sign -- the value contains spaces and will cause an error without it.

## Arguments

| Argument | Required | Description |
|---|---|---|
| `--exe` | Yes | Path to `llama-perplexity` binary |
| `--baseline` | Yes | Path to baseline GGUF -- BF16 or F16 (first shard if split) |
| `--quants` | Yes | Directory containing quant GGUFs (searched recursively) |
| `--dataset` | Yes | Plain text evaluation dataset |
| `--output` | Yes | Output directory for report, plots, log, and logits |
| `--model-name` | No | Short name used in filenames and plot titles |
| `--logits` | No | Path to existing logits file -- auto-generated in `--output` if not provided, reused on resume |
| `--args` | No | Extra flags for llama-perplexity for quant evaluation (default: `-t 7 -c 512 -ngl 99`) |
| `--args-baseline` | No | Extra flags for llama-perplexity for baseline logits generation only -- falls back to `--args` if not provided. Use when the baseline needs different VRAM settings than the quants. |

## Output

| File | Description |
|---|---|
| `{model}_report.md` | Ranked results table sorted by MDL_norm |
| `mdl_kld_plot_{model}.png` | KLD vs MDL_norm dual-axis scatter plot |
| `{model}_results.csv` | 8-column CSV with all metrics |
| `{model}.log` | Sweep log -- errors, warnings, skipped quants |
| `{model}-logits.bin` | BF16 logits (reused on resume) |
| `{model}-logits.bin.meta` | Logits metadata sidecar -- tracks which dataset generated the logits |

### CSV schema

```
Quantization, Size_GiB, PPL_Score, KLD_Score, MDL_norm, Num_Tokens, KLD_99, BPW
```
---

## Plot styling

- Uses the TOL qualitative colour palette (not default matplotlib)
- Labels are repelled from each other but dots stay in place (no point offset)

## Dataset

Any plain UTF-8 text file works. A minimum of ~50,000 characters is recommended for meaningful results -- more chunks means tighter confidence intervals.

As a rule of thumb: 25+ chunks at `-c 512` gives credible separation between quants. 100+ chunks is better for tight comparisons.

For instruct models, wrapping the dataset in the model's chat template (e.g. ChatML) gives more representative results than plain text, since the model's distribution is calibrated to that format.

A sample dataset is not included in this repository -- see [datasets_README.md](https://github.com/cmhamiche/kld-sweep/blob/main/datasets_README.md) for recommendations.

To build a custom dataset tailored to specific languages, tasks, or quantization use cases, see **[kld-sweep-dataset](https://github.com/cmhamiche/kld-sweep-dataset)** -- a companion tool that assembles and optionally chat-wraps evaluation and imatrix calibration datasets from the [eaddario/imatrix-calibration](https://huggingface.co/datasets/eaddario/imatrix-calibration) corpus.

## Notes

- The baseline model can be in the same directory as the quants -- it will be detected and excluded automatically. Sibling shards of the baseline are also excluded.
- Quantization names are extracted from GGUF metadata (`general.name`) -- the model name portion is stripped from the filename, leaving just the quant type (e.g. `IQ4_XS` instead of `Qwen3.5-0.8B-IQ4_XS`). Falls back to the full filename stem if metadata is unreadable.
- Split-shard baselines are supported -- point `--baseline` to the first shard only.
- If logits generation is interrupted (Ctrl+C, crash), the partial file is automatically detected on the next run and the script prompts you to regenerate.
- If you change `--dataset` between runs, the script detects the mismatch and asks whether to regenerate the logits.
- ERROR entries (crashed or unparseable runs) are retried automatically on the next run.
- The `--quants` directory is searched recursively -- subdirectories are included.

## Troubleshooting

See [FAQ.md](FAQ.md) for a full list of error codes and fixes.
