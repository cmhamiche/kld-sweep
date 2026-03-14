# FAQ & Troubleshooting

---

## E01 — File or directory not found

**Message:** `[ERROR E01] <label> not found: <path>`

**Causes:**
- Typo in a path argument
- The file exists but with a different extension or casing (Windows paths are case-insensitive, Linux/macOS are not)
- The BF16 model is split across multiple shards — point `--bf16` to the **first shard only** (e.g. `model-00001-of-00003.gguf`), llama.cpp will find the rest automatically

**Fix:** Double-check all paths. On Linux/macOS use `ls` to verify exact filenames.

---

## E02 — Binary not executable (Linux/macOS only)

**Message:** `[ERROR E02] llama-perplexity is not executable`

**Cause:** The downloaded binary does not have execute permission.

**Fix:**
```bash
chmod +x /path/to/llama-perplexity
```

---

## E02b — Wrong executable

**Message:** `[ERROR E02] Unexpected executable name: <name>`

**Cause:** The binary passed to `--exe` does not contain "perplexity" in its name — e.g. `llama-quantize`, `llama-cli`, `llama-server`.

**Fix:** Point `--exe` to `llama-perplexity` (Windows: `llama-perplexity.exe`). If you renamed the binary intentionally, the check will still trigger — rename it back or make sure it is actually the perplexity tool.

---

## E03 — Logits generation failed

**Message:** `[ERROR E03] Logits generation failed (exit code N)`

**Causes:**
- Not enough RAM to load the BF16 model. A 27B model in BF16 requires ~54 GB RAM. Check your available memory.
- `-ngl` set too high — more layers offloaded to GPU than VRAM can hold. Lower `-ngl` in `--args` (e.g. `-ngl 11` to mostly use CPU).
- Corrupted BF16 GGUF file — re-download the model.
- Wrong path to the binary.

**Note:** Any partial logits file is automatically deleted so the next run starts clean.

---

## E04 — Logits file is partial, corrupt, or from a different dataset

**Messages:**
- `[ERROR E04] Logits file exists but is suspiciously small` — generation was interrupted early
- `[ERROR E04] Logits file exists but has no metadata sidecar` — generation was interrupted before completing (file looks large enough but was never finished)
- `[ERROR E04] Logits were generated from a different dataset` — you changed the `--dataset` argument but the old logits file is still present

**Cause:** Any of the above means the existing logits file cannot be trusted for the current run.

**Fix:** Delete the logits file (and its `.meta` sidecar if present) and re-run:
```
# Windows
del "C:\path\to\logits.bin"
del "C:\path\to\logits.bin.meta"

# Linux/macOS
rm /path/to/logits.bin /path/to/logits.bin.meta
```
The script will regenerate both from scratch.

**Note:** The `.meta` sidecar is a small JSON file written next to the logits after successful generation. It records which dataset was used. If you interrupt generation with Ctrl+C, no `.meta` file is written, so the script knows the logits are incomplete on the next run.

---

## E05 — No GGUF files found

**Message:** `[ERROR E05] No .gguf files found in: <path>`

**Causes:**
- Wrong directory path
- Files have a different extension (`.gguf` expected, lowercase)
- The directory contains subdirectories but no `.gguf` files at the top level — the script does not recurse into subdirectories

**Fix:** Make sure all GGUF quant files are directly inside the `--quants` directory.

---

## E06 — PPL/KLD parse failed

**Message:** `[WARN W06] Could not parse PPL/KLD from output for: <filename>`

**Causes:**
- llama.cpp output format changed in a newer build — the regex expects `Mean PPL(Q)` and `Mean KLD:` which are mainline llama.cpp format. ik_llama or other forks may differ.
- The run completed but produced `nan` values (CUDA memory error, driver issue)
- The model file is corrupt

**What happens:** The result is recorded as `ERROR` in the CSV and the sweep continues. Re-run the script to retry ERROR entries after fixing the underlying issue.

**If using ik_llama:** Check the output format manually and open an issue on the repository.

---

## E07 — llama-perplexity crashed during sweep

**Message:** `[WARN W07] llama-perplexity returned exit code N for: <filename>`

**Causes:**
- Out of VRAM — lower `-ngl` in `--args`
- CUDA illegal memory access — GPU instability, overclocking, or driver issue. Check GPU temperatures and reset overclocks.
- The quant file is corrupted — re-download it.

**What happens:** The result is recorded as `ERROR` and the sweep continues. Fix the issue and re-run to retry.

**Note on `nan` results:** If you see `nan ± nan` in the output, this is almost always a GPU memory issue, not a software bug.

---

## E08 — Dataset too small

**Message:** `[ERROR E08] Dataset file is too small`

**Cause:** The dataset file is empty or nearly empty (under 1 KB).

**Fix:** Make sure the dataset is a plain UTF-8 text file with substantial content. A minimum of ~50,000 characters is recommended for meaningful KLD evaluation. See the included sample dataset for reference formatting.

---

## E09 — CSV corrupt or unexpected format

**Message:** `[ERROR E09] CSV has unexpected columns` or `Could not read existing CSV`

**Causes:**
- The CSV was manually edited and columns were renamed or reordered
- A different tool wrote a CSV with the same name to the output directory
- File system corruption

**Fix:** Delete or rename the existing CSV and re-run. The sweep will start from scratch but completed results will be regenerated.

---

## E10 — Plot failed

**Message:** `[WARN W10] <plot type> failed: <reason>`

**Causes:**
- Not enough valid results (all entries are ERROR) — fix E06/E07 first
- `adjustText` not installed — run `pip install adjustText`
- Only 1 valid result — efficiency score requires at least 2 for normalization
- For dumbbell plot: no overlapping quants between primary and secondary CSVs — make sure both sweeps ran against the same set of GGUF files

**What happens:** A warning is printed, the script continues, and other outputs are still generated.

---

## General tips

**Resuming an interrupted sweep:**
Just re-run with the same arguments. Any entry with a valid KLD score in the CSV is skipped automatically. Only ERROR entries and new files are processed.

**Retrying ERROR entries:**
Re-run the script as-is. ERROR entries are treated as incomplete and will be retried.

**The script shuts down my computer — how do I prevent that?**
Add `--no-shutdown` to your command. The automatic shutdown is opt-in by default but can be disabled.

**My quant filenames don't have a `_` separator (e.g. `TheBloke.Q4_K_M.gguf`):**
Labels and colours are assigned based on the first token before `_`. If your files use `.` as separator the label stripping may be imperfect — the full filename will still appear correctly in the CSV and efficiency report.

**`--args`: expected one argument error**
The `--args` value contains spaces and must be quoted. Always use the `=` form to be safe:
```
--args="-t 7 -c 4096 -ngl 36"
```
Without the `=`, some shells split the value on spaces before argparse sees it.

---

**Which llama.cpp build should I use?**
Use mainline llama.cpp releases from https://github.com/ggerganov/llama.cpp/releases.
The output parsing regex targets mainline format. ik_llama and other forks may produce different output — if parsing fails (E06), check the raw output format and open an issue.

**What context size and -ngl should I use?**
A good starting point is `-c 512 -ngl 99` (full GPU offload). If your VRAM is limited, lower `-ngl`. For BF16 logits generation you will likely need a much lower `-ngl` (e.g. `-ngl 11`) since the full model needs to fit in RAM+VRAM combined.
