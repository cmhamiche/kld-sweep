# kld-sweep

A cross-platform Python script to evaluate and compare GGUF quantizations of a model against its BF16/F16 baseline using KL Divergence and Perplexity, powered by [llama.cpp](https://github.com/ggerganov/llama.cpp).

---

## What it does

For each `.gguf` file in a directory, the script runs `llama-perplexity` and measures how much the quantized model's output distribution diverges from the full precision baseline. Results are saved to a CSV and two plots are generated — KLD vs size and PPL vs size. An efficiency ranking (Euclidean distance from the ideal 0,0 point) is also computed.

Logits are generated once from the BF16/F16 model and reused for all quants. The sweep resumes automatically if interrupted — already completed entries in the CSV are skipped.

Compatible with **mainline llama.cpp** and **[ik_llama](https://github.com/ikawrakow/ik_llama.cpp)**.

---

## Requirements

Python 3.10+

```
pip install pandas matplotlib adjustText scipy
```

A llama.cpp build with `llama-perplexity` — download the latest release for your platform from [github.com/ggerganov/llama.cpp/releases](https://github.com/ggerganov/llama.cpp/releases).

---

## Usage

```
python kld_sweep.py \
  --exe    /path/to/llama-perplexity \
  --bf16   /path/to/model-BF16.gguf \
  --quants /path/to/quants/ \
  --dataset /path/to/dataset.txt \
  --output /path/to/output/ \
  --model-name MyModel \
  --args="-t 7 -c 512 -ngl 99"
```

**Windows:**
```
python kld_sweep.py ^
  --exe    C:\llamacpp\llama-perplexity.exe ^
  --bf16   C:\models\MyModel-BF16.gguf ^
  --quants C:\models\quants\ ^
  --dataset C:\datasets\mydataset.txt ^
  --output  C:\results\ ^
  --model-name MyModel ^
  --args="-t 7 -c 512 -ngl 36"
```

> **Note:** Always use `--args="-t 7 ..."` with the `=` sign — the value contains spaces and will cause an error without it.

---

## Arguments

| Argument | Required | Description |
|---|---|---|
| `--exe` | Yes | Path to `llama-perplexity` binary |
| `--bf16` | Yes | Path to BF16/F16 GGUF (first shard if split) |
| `--quants` | Yes | Directory containing quant GGUFs |
| `--dataset` | Yes | Plain text evaluation dataset |
| `--output` | Yes | Output directory for CSV, plots, and logits |
| `--model-name` | No | Short name used in filenames and plot titles |
| `--logits` | No | Path to existing logits file — auto-generated in `--output` if not provided, reused on resume |
| `--args` | No | Extra flags for llama-perplexity (default: `-t 7 -c 4096 -ngl 99`) |

---

## Output

| File | Description |
|---|---|
| `{model}_results.csv` | PPL and KLD for each quant, sorted by size |
| `kld_plot_{model}.png` | KLD vs model size scatter plot |
| `ppl_plot_{model}.png` | Perplexity vs model size scatter plot |
| `{model}_efficiency.txt` | Efficiency ranking — Euclidean distance from (0,0) |
| `{model}-logits.bin` | BF16 logits (reused on resume) |
| `{model}-logits.bin.meta` | Logits metadata sidecar — tracks which dataset generated the logits |

---

## Dataset

Any plain UTF-8 text file works. A minimum of ~50,000 characters is recommended for meaningful results — more chunks means tighter confidence intervals.

As a rule of thumb: 25+ chunks at `-c 512` gives credible separation between quants. 100+ chunks is better for tight comparisons.

For instruct models, wrapping the dataset in the model's chat template (e.g. ChatML) gives more representative results than plain text, since the model's distribution is calibrated to that format.

A sample dataset is not included in this repository — see [[datasets/README.md](datasets/README.md)](https://github.com/cmhamiche/kld-sweep/blob/main/datasets_README.md) for recommendations.

---

## Notes

- The BF16/F16 model can be in the same directory as the quants — it will be detected and excluded automatically.
- Split-shard BF16 models are supported — point `--bf16` to the first shard only.
- If logits generation is interrupted (Ctrl+C, crash), the partial file is automatically detected on the next run and the script prompts you to regenerate.
- If you change `--dataset` between runs, the script should detects the mismatch and asks whether to regenerate the logits.
- Results from ERROR entries (crashed or unparseable runs) are retried automatically on the next run.

---

## Troubleshooting

See [FAQ.md](FAQ.md) for a full list of error codes and fixes.
