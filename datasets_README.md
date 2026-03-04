# Datasets

Large text files are not committed to this repository. This directory is a placeholder.

## Recommendations

**Wikitext2** — a standard NLP benchmark, plain prose, good neutral baseline.
Download via the llama.cpp helper script:
```bash
bash https://raw.githubusercontent.com/ggerganov/llama.cpp/master/scripts/get-wikitext-2.sh
```
Use `wikitext-2-raw/wiki.test.raw` — 72 chunks at `-c 4096`.

**Custom corpus** — for instruct models, a chat-template-wrapped dataset gives more representative results than plain text. Assemble content from diverse domains (science, code, multilingual, etc.), wrap consecutive line pairs in the model's chat template, and aim for 25+ chunks at your chosen context size.

## Format

Plain UTF-8 text. No special formatting required beyond whatever chat template wrapping you choose to apply.
