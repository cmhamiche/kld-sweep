# Datasets

Large text files are not committed to this repository. This directory is a placeholder.

## Building a custom dataset

The recommended way to build an evaluation or imatrix calibration dataset is **[kld-sweep-dataset](https://github.com/cmhamiche/kld-sweep-dataset)**, a companion tool that:

- Sources data from [eaddario/imatrix-calibration](https://huggingface.co/datasets/eaddario/imatrix-calibration) (tools / math / code / multilingual text)
- Downloads only the files you need on demand
- Wraps samples in the model's chat template extracted directly from the GGUF
- Targets a specific chunk count at `-c 512` and estimates the actual llama-perplexity chunk output

## Manual options

**Wikitext2** — a standard NLP benchmark, plain prose, good neutral baseline.
Download via the llama.cpp helper script:
```bash
bash https://raw.githubusercontent.com/ggerganov/llama.cpp/master/scripts/get-wikitext-2.sh
```
Use `wikitext-2-raw/wiki.test.raw`.

**Custom corpus** — for instruct models, a chat-template-wrapped dataset gives more representative results than plain text. Assemble content from diverse domains (science, code, multilingual, etc.), wrap each sample in the model's chat template, and aim for 50–100 chunks at `-c 512` for eval, 500–1000 for imatrix calibration.

## Format

Plain UTF-8 text. Samples separated by double newlines. No special formatting required beyond whatever chat template wrapping you choose to apply.
