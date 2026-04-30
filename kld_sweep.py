"""
kld_sweep.py — Cross-platform KLD evaluation sweep for GGUF quantizations.

Usage:
    python kld_sweep.py --exe /path/to/llama-perplexity \
                        --baseline /path/to/model-baseline.gguf \
                        --quants /path/to/quants/ \
                        --dataset /path/to/dataset.txt \
                        --output /path/to/output/ \
                        [--logits /path/to/logits.bin] \
                        [--args "-t 7 -c 512 -ngl 999 -cmoe"] \
                        [--args-baseline "-t 7 -c 512 -ngl 20"] \
                        [--model-name MyModel]

Resume: already completed entries in the CSV are skipped automatically.

Dependencies: pandas, matplotlib, adjustText
pip install pandas matplotlib adjustText
"""

import argparse
import csv
import json
import math
import os
import re
import struct
import subprocess
import sys
import time
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from adjustText import adjust_text

GGUF_TYPE_SIZES = {
    0: 1,
    1: 1,
    2: 2,
    3: 2,
    4: 4,
    5: 4,
    6: 4,
    7: 4,
    8: None,
    9: None,
    10: 8,
    11: 8,
    12: 8,
}

# ---------------------------------------------------------------------------
# Error codes — referenced in FAQ.md
# ---------------------------------------------------------------------------
ERR_PATH_NOT_FOUND   = 1   # A required file or directory does not exist
ERR_NOT_EXECUTABLE   = 2   # llama-perplexity binary is not executable
ERR_LOGITS_FAILED    = 3   # Logits generation subprocess returned non-zero
ERR_LOGITS_PARTIAL   = 4   # Logits file exists but is suspiciously small
ERR_NO_QUANTS        = 5   # No .gguf files found in quant directory
ERR_PARSE_FAILED     = 6   # Could not parse PPL/KLD from llama-perplexity output
ERR_SUBPROCESS       = 7   # llama-perplexity crashed or returned non-zero during sweep
ERR_DATASET_EMPTY    = 8   # Dataset file is empty or too small
ERR_CSV_CORRUPT      = 9   # Existing CSV could not be parsed
ERR_PLOT_FAILED = 10 # Plotting failed (e.g. not enough valid results)
ERR_LOG_FAILED = 11 # Log or report file could not be written

# Minimum expected logits file size: 1 MB — anything smaller is likely partial/corrupt
LOGITS_MIN_BYTES = 1 * 1024 * 1024

_RE_SHARD = re.compile(r"-(\d{5})-of-(\d{5})$")

# ---------------------------------------------------------------------------
# Colours — extended palette, auto-assigned to unknown quantizers
# ---------------------------------------------------------------------------
PALETTE = [
    "#4477AA", "#EE6677", "#228833", "#CCBB44",
    "#66CCEE", "#AA3377", "#BBBBBB",
]

MARKERS = [
    "o", "D", "s", "^", "v", "P", "X", "p", "h", "<", ">", "H", "d",
]

BG_COLOR   = "#F5F5DC"
GRID_COLOR = "#D3D3D3"

plt.rcParams.update({
    "figure.facecolor": BG_COLOR,
    "axes.facecolor":   BG_COLOR,
    "axes.edgecolor":   "#333333",
    "axes.labelcolor":  "#333333",
    "text.color":       "#333333",
    "xtick.color":      "#333333",
    "ytick.color":      "#333333",
    "font.family":      "sans-serif",
    "font.size":        10,
})

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def read_gguf_name(path: Path) -> str | None:
    try:
        with open(path, "rb") as f:
            magic = f.read(4)
            if magic != b"GGUF":
                return None
            version = struct.unpack("<I", f.read(4))[0]
            if version < 2:
                return None
            _tensor_count = struct.unpack("<Q", f.read(8))[0]
            kv_count = struct.unpack("<Q", f.read(8))[0]
            for _ in range(kv_count):
                key_len = struct.unpack("<Q", f.read(8))[0]
                key = f.read(key_len).decode("utf-8")
                vtype = struct.unpack("<I", f.read(4))[0]
                if key == "general.name" and vtype == 8:
                    str_len = struct.unpack("<Q", f.read(8))[0]
                    return f.read(str_len).decode("utf-8")
                if vtype == 8:
                    str_len = struct.unpack("<Q", f.read(8))[0]
                    f.seek(str_len, 1)
                elif vtype == 9:
                    elem_type = struct.unpack("<I", f.read(4))[0]
                    elem_count = struct.unpack("<Q", f.read(8))[0]
                    for _ in range(elem_count):
                        if elem_type == 8:
                            el = struct.unpack("<Q", f.read(8))[0]
                            f.seek(el, 1)
                        elif elem_type in GGUF_TYPE_SIZES and GGUF_TYPE_SIZES[elem_type] is not None:
                            f.seek(GGUF_TYPE_SIZES[elem_type], 1)
                        else:
                            return None
                elif vtype in GGUF_TYPE_SIZES and GGUF_TYPE_SIZES[vtype] is not None:
                    f.seek(GGUF_TYPE_SIZES[vtype], 1)
                else:
                    return None
    except Exception:
        return None
    return None

def quant_name_from_file(path: Path) -> str:
    gguf_name = read_gguf_name(path)
    stem = path.stem
    # Strip shard suffix (e.g. -00001-of-00002) — all shards share the same name
    shard = _RE_SHARD.search(stem)
    if shard:
        stem = stem[:shard.start()]
    if not gguf_name:
        return stem
    normalized = gguf_name.replace(" ", "-")
    if stem.lower().startswith(normalized.lower()):
        remainder = stem[len(normalized):]
        cleaned = remainder.lstrip("-_ .")
        return cleaned if cleaned else stem
    elif normalized.lower() in stem.lower():
        idx = stem.lower().index(normalized.lower())
        suffix = stem[idx + len(normalized):]
        cleaned = suffix.lstrip("-_ .")
        return cleaned if cleaned else stem
    else:
        return stem

def fatal(msg: str, code: int):
    """Print a clear error and exit with a documented code."""
    print(f"\n[ERROR E{code:02d}] {msg}", file=sys.stderr)
    print(f"  -> See FAQ.md entry E{code:02d} for troubleshooting.", file=sys.stderr)
    sys.exit(code)


def warn(msg: str, code: int):
    """Print a non-fatal warning with its FAQ reference."""
    print(f"\n[WARN W{code:02d}] {msg}", file=sys.stderr)
    print(f" -> See FAQ.md entry E{code:02d} for details.", file=sys.stderr)


class SweepLogger:
    def __init__(self, log_path: Path, model_name: str):
        self.log_path = log_path
        self.model_name = model_name
        self._file = open(log_path, "w", encoding="utf-8")
        self._write_header()

    def _write_header(self):
        self._file.write(f"# kld-sweep log — {self.model_name}\n\n")
        self._file.flush()

    def log(self, msg: str):
        self._file.write(msg + "\n")
        self._file.flush()

    def log_skip(self, name: str):
        self.log(f"- [SKIP] {name} — already in results")

    def log_error(self, name: str, detail: str):
        self.log(f"- [ERROR] {name}: {detail}")

    def log_result(self, name: str, ppl: str, kld: str, size: float, mdl_norm, num_tokens: int, kld_99: str = "", bpw: str = ""):
        parts = f"- [OK] {name} — PPL: {ppl} | KLD: {kld} | Size: {size:.3f} GiB | MDL_norm: {mdl_norm:.3f} | Tokens: {num_tokens}"
        if kld_99:
            parts += f" | KLD_99: {kld_99}"
        if bpw:
            parts += f" | BPW: {bpw}"
        self.log(parts)

    def close(self):
        self._file.close()


def parse_args():
    p = argparse.ArgumentParser(description="KLD sweep for GGUF quants.")
    p.add_argument("--exe", required=True, help="Path to llama-perplexity binary")
    p.add_argument("--baseline", required=True, help="Path to baseline GGUF (BF16 or F16 — first shard if split)")
    p.add_argument("--quants", required=True, help="Directory containing quant GGUFs")
    p.add_argument("--dataset", required=True, help="Primary evaluation dataset (.txt)")
    p.add_argument("--output", required=True, help="Output directory for report, plots, log, and logits")
    p.add_argument("--logits", default=None, help="Path to existing logits file (optional — auto-generated in --output if not provided, reused on resume)")
    p.add_argument("--args", default="-t 7 -c 512 -ngl 99",
        help="Extra flags passed to llama-perplexity for quant evaluation. Must be quoted: --args=\"-t 7 -c 512 -ngl 36\"")
    p.add_argument("--args-baseline", default=None,
        help="Extra flags passed to llama-perplexity for baseline logits generation only. Falls back to --args if not provided. Useful when the baseline does not fit in VRAM with the same settings as the quants.")
    p.add_argument("--model-name", default=None, help="Short model name used in plot titles")

    return p.parse_args()


def gib(path: Path) -> float:
    return path.stat().st_size / (1024 ** 3)


def shard_base(stem: str) -> str | None:
    """Return the prefix before the shard suffix, or None if not a shard."""
    m = _RE_SHARD.search(stem)
    return stem[:m.start()] if m else None


def shard_total_gib(first_shard: Path, quant_dir: Path) -> float:
    """Sum file sizes of all shards sharing the same base prefix as first_shard."""
    base = shard_base(first_shard.stem)
    if base is None:
        return gib(first_shard)
    total = 0.0
    for f in quant_dir.glob("**/*.gguf"):
        if shard_base(f.stem) == base:
            total += f.stat().st_size
    return total / (1024 ** 3)


def validate_paths(exe: Path, baseline: Path, dataset: Path, quant_dir: Path) -> int:
    """Check all required inputs exist and are usable."""
    for p, label in [
        (baseline,  "Baseline model"),
        (dataset,   "Dataset"),
        (quant_dir, "Quant directory"),
    ]:
        if not p.exists():
            fatal(f"{label} not found: {p}", ERR_PATH_NOT_FOUND)

    if not exe.exists():
        fatal(
            f"llama-perplexity binary not found: {exe}\n"
            "  Download llama.cpp from https://github.com/ggerganov/llama.cpp/releases\n"
            "  and point --exe to llama-perplexity (or llama-perplexity.exe on Windows).",
            ERR_PATH_NOT_FOUND,
        )

    if not os.access(exe, os.X_OK) and sys.platform != "win32":
        fatal(
            f"llama-perplexity is not executable: {exe}\n"
            f"  Run: chmod +x {exe}",
            ERR_NOT_EXECUTABLE,
        )

    if "perplexity" not in exe.name.lower():
        fatal(
            f"Unexpected executable name: {exe.name}\n"
            "  --exe should point to llama-perplexity (or llama-perplexity.exe on Windows).\n"
            f"  Got: {exe.name}\n"
            "  If you renamed the binary, this check can be ignored but make sure it is the right tool.",
            ERR_NOT_EXECUTABLE,
        )

    if not quant_dir.is_dir():
        fatal(f"Quant path is not a directory: {quant_dir}", ERR_PATH_NOT_FOUND)

    dataset_size = dataset.stat().st_size
    if dataset_size < 1024:
        fatal(
            f"Dataset file is too small ({dataset_size} bytes): {dataset}\n"
            "  Make sure the dataset is a plain text file with sufficient content.",
            ERR_DATASET_EMPTY,
        )
    return dataset_size


def logits_meta_path(logits: Path) -> Path:
    return logits.with_suffix(".bin.meta")


def write_logits_meta(logits: Path, dataset: Path, dataset_size: int):
    """Write sidecar metadata after successful logits generation."""
    meta = {"dataset": str(dataset.resolve()), "dataset_size": dataset_size}
    logits_meta_path(logits).write_text(json.dumps(meta), encoding="utf-8")


def check_logits(logits: Path, dataset: Path, dataset_size: int):
    """Check logits file is complete and was generated from the current dataset."""
    size = logits.stat().st_size
    if size < LOGITS_MIN_BYTES:
        fatal(
            f"Logits file exists but is suspiciously small ({size} bytes):\n"
            f"  {logits}\n"
            "  This usually means a previous logits generation was interrupted.\n"
            "  Delete it and re-run:\n"
            f"  Windows: del \"{logits}\"\n"
            f"  Linux/macOS: rm \"{logits}\"",
            ERR_LOGITS_PARTIAL,
        )
    meta_path = logits_meta_path(logits)
    if not meta_path.exists():
        fatal(
            f"Logits file exists but has no metadata sidecar — generation was likely interrupted:\n"
            f"  {logits}\n"
            "  Delete it and re-run:\n"
            f"  Windows: del \"{logits}\"\n"
            f"  Linux/macOS: rm \"{logits}\"",
            ERR_LOGITS_PARTIAL,
        )
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        recorded_size = meta.get("dataset_size")
        current_size = dataset_size
        if recorded_size != current_size:
            print(
                f"\n[WARN] Logits were generated from a different dataset."
                f"\n  Logits dataset : {meta.get('dataset', 'unknown')} ({recorded_size} bytes)"
                f"\n  Current dataset: {dataset} ({current_size} bytes)"
            )
            try:
                answer = input("  Delete existing logits and regenerate? [y/N]: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                answer = "n"
            if answer == "y":
                logits.unlink()
                logits_meta_path(logits).unlink(missing_ok=True)
                print("[logits] Deleted. Will regenerate.")
                return False
            else:
                fatal(
                    "Logits dataset mismatch — cannot continue.\n"
                    "  Delete the logits file manually and re-run:\n"
                    f"  Windows: del \"{logits}\"\n"
                    f"  Linux/macOS: rm \"{logits}\"",
                    ERR_LOGITS_PARTIAL,
                )
    except Exception as e:
        fatal(
            f"Could not read logits metadata: {meta_path}\n  Reason: {e}\n"
            "  Delete both files and re-run:\n"
            f"  Windows: del \"{logits}\" & del \"{meta_path}\"\n"
            f"  Linux/macOS: rm \"{logits}\" \"{meta_path}\"",
            ERR_LOGITS_PARTIAL,
        )
    return True


def run(cmd: list, label: str) -> tuple:
    """
    Run a subprocess, stream output to console, return (full_output, returncode).
    Does NOT raise on non-zero -- caller decides what to do.
    """
    print(f"\n>>> {label}", flush=True)
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except FileNotFoundError:
        fatal(
            f"Could not launch binary: {cmd[0]}\n"
            "  Check that the path is correct and the file exists.",
            ERR_PATH_NOT_FOUND,
        )
    except PermissionError:
        fatal(
            f"Permission denied when launching: {cmd[0]}\n"
            f"  On Linux/macOS run: chmod +x {cmd[0]}",
            ERR_NOT_EXECUTABLE,
        )

    lines = []
    for line in proc.stdout:
        print(line, end="", flush=True)
        lines.append(line)
    proc.wait()
    return "".join(lines), proc.returncode


def generate_logits(exe: Path, baseline: Path, dataset: Path, logits: Path, extra: list, dataset_size: int):
    """Generate baseline logits file. Cleans up partial file on failure."""
    size_gib = gib(baseline)
    print(f"\n[logits] Generating from {baseline.name} ({size_gib:.2f} GiB) — ETA will appear below...")
    cmd = (
        [str(exe), "-m", str(baseline), "-f", str(dataset)]
        + extra
        + ["--kl-divergence-base", str(logits)]
    )
    raw, rc = run(cmd, f"Generating logits -> {logits.name}")

    if rc != 0:
        if not logits.exists() or logits.stat().st_size < LOGITS_MIN_BYTES:
            if logits.exists():
                logits.unlink()
            print(f"[logits] Partial logits file deleted: {logits}")
            fatal(
                f"Logits generation failed (exit code {rc}).\n"
                " Common causes:\n"
                " - Not enough RAM to load the BF16 model\n"
                " - Incorrect --args flags (check -ngl value for your VRAM)\n"
                " - Corrupted BF16 GGUF file\n"
                " Last 20 lines of output:\n"
                + "\n".join(raw.splitlines()[-20:]),
                ERR_LOGITS_FAILED,
            )
        print(f"[logits] Note: exit code {rc} but logits file looks valid — continuing")

    if not logits.exists() or logits.stat().st_size < LOGITS_MIN_BYTES:
        if logits.exists():
            logits.unlink()
        fatal(
            "Logits generation appeared to succeed (exit 0) but output file is missing or empty.\n"
            "  Check available disk space.",
            ERR_LOGITS_PARTIAL,
        )

    write_logits_meta(logits, dataset, dataset_size)
    print(f"[logits] Generated: {logits} ({gib(logits):.2f} GiB)")


_RE_TOKENS = re.compile(
    r"prompt eval time\s*=\s*[\d.]+\s*ms\s*/\s*(\d+)\s+tokens",
    re.IGNORECASE,
)


def parse_output(raw: str) -> tuple:
    ppl = kld = "ERROR"
    kld_99 = ""
    bpw = ""
    num_tokens = 0
    m = re.search(r"Mean PPL\(Q\)\s*:\s*([0-9]+\.[0-9]+)", raw)
    if m:
        ppl = m.group(1)
    m = re.search(r"Mean\s+KLD:\s+([0-9]+\.[0-9]+)", raw)
    if m:
        kld = m.group(1)
    m = re.search(r"99\.9%\s+KLD:\s+([0-9]+\.[0-9]+)", raw)
    if m:
        kld_99 = m.group(1)
    m = re.search(r"model size\s*=.*\((\d+\.\d+)\s+BPW\)", raw)
    if m:
        bpw = m.group(1)
    m = _RE_TOKENS.search(raw)
    if m:
        num_tokens = int(m.group(1))
    return ppl, kld, kld_99, bpw, num_tokens


def load_csv(path: Path) -> dict:
    results = {}
    if not path.exists():
        return results
    try:
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            old4 = ["Quantization", "Size_GiB", "PPL_Score", "KLD_Score"]
            old6 = ["Quantization", "Size_GiB", "PPL_Score", "KLD_Score", "MDL_norm", "Num_Tokens"]
            cur = ["Quantization", "Size_GiB", "PPL_Score", "KLD_Score", "MDL_norm", "Num_Tokens", "KLD_99", "BPW"]
            if reader.fieldnames == cur:
                for row in reader:
                    results[row["Quantization"]] = row
            elif reader.fieldnames == old6:
                for row in reader:
                    row["KLD_99"] = ""
                    row["BPW"] = ""
                    results[row["Quantization"]] = row
                warn("CSV was 6-column format — KLD_99 and BPW columns added (empty). Re-run to populate.", ERR_CSV_CORRUPT)
            elif reader.fieldnames == old4:
                for row in reader:
                    try:
                        size = float(row["Size_GiB"])
                        ppl = float(row["PPL_Score"])
                        row["MDL_norm"] = size * 8.0 + math.log2(max(ppl, 1e-9))
                    except (ValueError, ZeroDivisionError):
                        row["MDL_norm"] = ""
                    row["Num_Tokens"] = 0
                    row["KLD_99"] = ""
                    row["BPW"] = ""
                    results[row["Quantization"]] = row
                warn("CSV was old 4-column format — MDL_norm auto-computed, KLD_99/BPW/Num_Tokens empty. Re-run to populate.", ERR_CSV_CORRUPT)
            else:
                fatal(
                    f"CSV has unexpected columns: {reader.fieldnames}\n"
                    f" Expected: {cur}\n"
                    f" File: {path}\n"
                    " Rename or delete the file to start fresh.",
                    ERR_CSV_CORRUPT,
                )
    except Exception as e:
        fatal(
            f"Could not read existing CSV: {path}\n Reason: {e}\n"
            " If the file is corrupt, delete it and re-run.",
            ERR_CSV_CORRUPT,
        )
    return results


def save_csv(path: Path, results: dict):
    rows = list(results.values())
    fieldnames = ["Quantization", "Size_GiB", "PPL_Score", "KLD_Score", "MDL_norm", "Num_Tokens", "KLD_99", "BPW"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------------------
# Source / label helpers
# ---------------------------------------------------------------------------

def get_source_styles(df: pd.DataFrame) -> tuple[dict, dict, dict]:
    """Auto-assign TOL colours + marker shapes to quantizer prefixes and build a name->prefix lookup."""
    prefixes = {}
    for name in df["Quantization"]:
        prefix = name.split("_")[0] if "_" in name else name.split(".")[0]
        prefixes.setdefault(prefix, None)
    color_map = {prefix: PALETTE[i % len(PALETTE)] for i, prefix in enumerate(prefixes)}
    marker_map = {prefix: MARKERS[i % len(MARKERS)] for i, prefix in enumerate(prefixes)}
    source_map = {}
    for name in df["Quantization"]:
        prefix = name.split("_")[0] if "_" in name else name.split(".")[0]
        source_map[name] = prefix
    return color_map, marker_map, source_map


def get_source(name: str, source_map: dict) -> str:
    return source_map.get(name, "other")


def get_label(name: str, model_name: str) -> str:
    parts = name.split("_", 1)
    if len(parts) == 2:
        name = parts[1]
    if model_name:
        name = name.replace(model_name + "-", "").replace(model_name, "")
    return name.strip("-_.")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def mdl_kld_plot(df: pd.DataFrame, out_path: Path, color_map: dict, marker_map: dict, model_name: str):
    if df.empty:
        warn("No valid data for MDL-KLD plot -- skipping.", ERR_PLOT_FAILED)
        return
    try:
        fig, ax = plt.subplots(figsize=(15, 9), dpi=200)
        ax2 = ax.twiny()

        kld_raw = df["KLD_Score"].values.astype(float)
        kld_floor = np.where(kld_raw <= 0, 1e-9, kld_raw)

        texts = []
        for _, row in df.iterrows():
            src = row["Source"]
            c = color_map.get(src, "#888888")
            mk = marker_map.get(src, "o")
            kld_val = max(float(row["KLD_Score"]), 1e-9)
            ax.scatter(row["MDL_norm"], kld_val,
                       color=c, marker=mk, s=75, alpha=0.9, zorder=10,
                       edgecolors="white", linewidths=0.5)
            ax2.scatter(row["Size_GiB"], kld_val,
                        color=c, marker=mk, s=0, alpha=0)
            texts.append(ax.text(row["MDL_norm"], kld_val,
                                 row["Label"], fontsize=8, alpha=0.95, zorder=11))

        adjust_text(texts, ax=ax,
                    force_text=(0.6, 0.9), force_points=(0, 0),
                    arrowprops=dict(arrowstyle="-", color="gray", lw=0.4, alpha=0.5),
                    lim=200)

        ax.set_yscale("log")
        ax.set_xlabel("MDL_norm (bits/token, 1B amortised)", fontsize=12, fontweight="bold")
        ax.set_ylabel("KL Divergence (log scale, lower is better)", fontsize=12, fontweight="bold")
        ax2.set_xlabel("Model Size (GiB)", fontsize=12, fontweight="bold")
        ax.grid(True, color=GRID_COLOR, linestyle="--", linewidth=0.5, alpha=0.6)
        ax.set_title(f"{model_name} — KLD vs MDL_norm", fontsize=14, fontweight="bold", pad=15)

        legend_handles = [
            mlines.Line2D([], [], marker=marker_map.get(s, "o"), color="w",
                          markerfacecolor=c, markersize=9, label=s)
            for s, c in color_map.items()
        ]
        ax.legend(handles=legend_handles, loc="upper right",
                  facecolor="#FFFFFF", edgecolor="#CCCCCC", fontsize=9)

        plt.tight_layout()
        plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close()
        print(f"Saved: {out_path}")
    except Exception as e:
        warn(f"MDL-KLD plot failed: {e}", ERR_PLOT_FAILED)


def generate_markdown_report(df: pd.DataFrame, model_name: str, out_path: Path):
    if df.empty:
        warn("No valid data for markdown report -- skipping.", ERR_LOG_FAILED)
        return
    lines = []
    lines.append(f"# KLD-Sweep Report — {model_name}")
    lines.append("")
    lines.append("MDL_norm = Size_GiB × 8 + log₂(PPL) [bits/token, amortised over 1B tokens]")
    lines.append("")
    lines.append("## Results (sorted by MDL_norm)")
    lines.append("")
    lines.append("| Rank | Quantization | Size (GiB) | BPW | PPL | KLD | KLD 99.9% | MDL_norm |")
    lines.append("|------|--------------|------------|-----|-----|-----|-----------|----------|")
    df_sorted = df.sort_values("MDL_norm").reset_index(drop=True)
    for i, (_, row) in enumerate(df_sorted.iterrows(), 1):
        bpw = f"{row['BPW']:.3f}" if pd.notna(row.get("BPW")) and row.get("BPW") != "" else "—"
        kld99 = f"{row['KLD_99']:.6f}" if pd.notna(row.get("KLD_99")) and row.get("KLD_99") != "" else "—"
        lines.append(
            f"| {i} | {row['Quantization']} | {row['Size_GiB']:.3f} | "
            f"{bpw} | {row['PPL_Score']:.4f} | {row['KLD_Score']:.6f} | {kld99} | {row['MDL_norm']:.3f} |"
        )
    lines.append("")
    report = "\n".join(lines)
    print(report)
    try:
        out_path.write_text(report, encoding="utf-8")
        print(f"\nSaved: {out_path}")
    except Exception as e:
        warn(f"Could not write markdown report: {e}", ERR_LOG_FAILED)


def backfill_metadata(exe: Path, quant_files: list, extra: list, results: dict, csv_path: Path, dataset: Path):
    """Dry-run llama-perplexity on quants missing BPW to capture metadata from model load."""
    needed = []
    for qf in quant_files:
        name = quant_name_from_file(qf)
        entry = results.get(name)
        if entry and not entry.get("BPW"):
            needed.append((qf, name))
    if not needed:
        return

    print(f"\n[backfill] {len(needed)} quant(s) missing BPW — running dry passes to capture metadata...")
    for qf, name in needed:
        cmd = [str(exe), "-m", str(qf), "-f", str(dataset), "-c", "1", "-t", "1"] + extra
        raw, rc = run(cmd, f"Backfill BPW: {name}")
        bpw = ""
        m = re.search(r"model size\s*=.*\((\d+\.\d+)\s+BPW\)", raw)
        if m:
            bpw = m.group(1)
        if bpw:
            results[name]["BPW"] = bpw
            print(f"  -> {name}: BPW={bpw}")
        else:
            print(f"  -> {name}: BPW not found in output")
        save_csv(csv_path, results)
        time.sleep(1)


# ---------------------------------------------------------------------------
# Sweep runner
# ---------------------------------------------------------------------------

def run_sweep(exe: Path, quant_files: list, dataset: Path, logits: Path,
              extra: list, csv_path: Path, label: str,
              existing_results: dict | None = None, logger: SweepLogger | None = None,
              quant_dir: Path | None = None) -> dict:
    results = existing_results if existing_results is not None else load_csv(csv_path)
    errors = []

    for qf in quant_files:
        name = quant_name_from_file(qf)
        size = round(shard_total_gib(qf, quant_dir) if quant_dir and shard_base(qf.stem) else gib(qf), 3)

        existing = results.get(name)
        if existing and existing.get("KLD_Score") not in (None, "", "ERROR"):
            print(f"[skip{label}] {name}")
            if logger:
                logger.log_skip(name)
            continue

        cmd = (
            [str(exe), "-m", str(qf), "-f", str(dataset)]
            + extra
            + ["--kl-divergence", "--kl-divergence-base", str(logits)]
        )
        raw, rc = run(cmd, f"Testing{label}: {name} ({size} GiB)")

        ppl, kld, kld_99, bpw, num_tokens = parse_output(raw)

        if ppl == "ERROR" or kld == "ERROR":
            rc_info = f" (exit code {rc})" if rc != 0 else ""
            warn(
                f"Could not parse PPL/KLD from output for: {qf.name}{rc_info}\n"
                " Last 20 lines of output:\n"
                + "\n".join(raw.splitlines()[-20:]) + "\n"
                " This may mean llama-perplexity output format has changed.\n"
                " Result recorded as ERROR -- sweep will continue.",
                ERR_PARSE_FAILED,
            )
            results[name] = {
                "Quantization": name, "Size_GiB": size,
                "PPL_Score": "ERROR", "KLD_Score": "ERROR",
                "MDL_norm": "ERROR", "Num_Tokens": 0,
                "KLD_99": "", "BPW": ""}
            errors.append(name)
            if logger:
                logger.log_error(name, f"parse failed (exit code {rc})" if rc != 0 else "could not parse PPL/KLD from output")
            save_csv(csv_path, results)
            time.sleep(1)
            continue

        if rc != 0:
            print(f" [note] exit code {rc} but output parsed OK — using result")

        print(f"-> PPL: {ppl} | KLD: {kld} | KLD_99: {kld_99 or 'N/A'} | BPW: {bpw or 'N/A'}")
        try:
            mdl_norm = size * 8.0 + math.log2(max(float(ppl), 1e-9))
        except (ValueError, ZeroDivisionError):
            mdl_norm = "ERROR"
        results[name] = {
            "Quantization": name, "Size_GiB": size,
            "PPL_Score": ppl, "KLD_Score": kld,
            "MDL_norm": mdl_norm, "Num_Tokens": num_tokens,
            "KLD_99": kld_99, "BPW": bpw}
        save_csv(csv_path, results)
        time.sleep(1)
        if logger:
            logger.log_result(name, ppl, kld, size, mdl_norm, num_tokens, kld_99=kld_99, bpw=bpw)

    if errors:
        print(f"\n[!] {len(errors)} file(s) recorded as ERROR — see log file for details.")
        print(" Re-run the script to retry ERROR entries.")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    exe        = Path(args.exe)
    baseline   = Path(args.baseline)
    quant_dir  = Path(args.quants)
    dataset    = Path(args.dataset)
    out_dir    = Path(args.output)
    extra      = args.args.split()
    extra_base = args.args_baseline.split() if args.args_baseline else extra
    model_name = args.model_name or baseline.stem

    out_dir.mkdir(parents=True, exist_ok=True)

    logits   = Path(args.logits) if args.logits else out_dir / f"{model_name}-logits.bin"
    csv_path = out_dir / f"{model_name}_results.csv"
    log_path = out_dir / f"{model_name}.log"
    report_path = out_dir / f"{model_name}_report.md"

    # --- Validate ---
    dataset_size = validate_paths(exe, baseline, dataset, quant_dir)

    # --- Find quants ---
    # Always exclude the baseline file itself.
    # For split-shard baselines (e.g. -00001-of-00002), also exclude other shards by prefix.
    # Prefix matching is intentionally limited to shard patterns to avoid accidentally
    # excluding quants that share a name fragment with the baseline (e.g. a BF16 baseline
    # vs BF16 quants).
    # Non-first shards (-00002-of-NNNNN, etc.) are excluded — llama-perplexity loads them
    # automatically from the first shard.
    baseline_stem = baseline.stem
    is_shard = bool(_RE_SHARD.search(baseline_stem))
    baseline_shard_prefix = shard_base(baseline_stem) if is_shard else None

    # recursive=True walks all subdirectories — no path/name hardcoding needed
    all_gguf = sorted(quant_dir.glob("**/*.gguf"), key=lambda f: f.stat().st_size)
    quant_files = []
    skipped = []
    baseline_resolved = baseline.resolve()
    for f in all_gguf:
        is_baseline_file = f.resolve() == baseline_resolved
        is_baseline_shard = baseline_shard_prefix and shard_base(f.stem).lower() == baseline_shard_prefix.lower()
        m = _RE_SHARD.search(f.stem)
        is_nonfirst_shard = m is not None and m.group(1) != "00001"
        if is_baseline_file or is_baseline_shard or is_nonfirst_shard:
            skipped.append(f.name)
        else:
            quant_files.append(f)
    if skipped:
        print(f"\n[quants] Excluded: {', '.join(skipped)}")
    if not quant_files:
        fatal(
            f"No .gguf files found in: {quant_dir}\n"
            " Make sure the path is correct and the files have a .gguf extension.",
            ERR_NO_QUANTS,
        )
    print(f"\nFound {len(quant_files)} quant file(s) in {quant_dir}")

    # --- Logits ---
    if extra_base != extra:
        print(f"[logits] Using --args-baseline for logits generation: {' '.join(extra_base)}")

    need_generate = True
    if logits.exists():
        if check_logits(logits, dataset, dataset_size):
            print(f"[logits] Using existing logits: {logits} ({gib(logits):.2f} GiB)")
            need_generate = False
    if need_generate:
        generate_logits(exe, baseline, dataset, logits, extra_base, dataset_size)

    # --- Primary sweep ---
    existing = load_csv(csv_path)
    logger = SweepLogger(log_path, model_name)
    try:
        results = run_sweep(exe, quant_files, dataset, logits, extra, csv_path, "", existing_results=existing, logger=logger, quant_dir=quant_dir)
        save_csv(csv_path, results)
    finally:
        logger.close()
    print(f"\nPrimary results saved to {csv_path}")
    print(f"Log saved to {log_path}")

    # --- Backfill BPW for entries missing it ---
    backfill_metadata(exe, quant_files, extra, results, csv_path, dataset)

    # --- Build DataFrame ---
    rows = []
    for r in results.values():
        try:
            kld = float(r["KLD_Score"])
            ppl = float(r["PPL_Score"])
            size = float(r["Size_GiB"])
            mdl_norm = float(r["MDL_norm"]) if r.get("MDL_norm") not in (None, "", "ERROR") else size * 8.0 + math.log2(max(ppl, 1e-9))
        except (ValueError, TypeError):
            continue
        kld_99 = ""
        if r.get("KLD_99") not in (None, "", "ERROR"):
            try:
                kld_99 = float(r["KLD_99"])
            except (ValueError, TypeError):
                kld_99 = ""
        bpw = ""
        if r.get("BPW") not in (None, "", "ERROR"):
            try:
                bpw = float(r["BPW"])
            except (ValueError, TypeError):
                bpw = ""
        rows.append({"Quantization": r["Quantization"], "Size_GiB": size, "PPL_Score": ppl, "KLD_Score": kld, "MDL_norm": mdl_norm, "KLD_99": kld_99, "BPW": bpw})
    df = pd.DataFrame(rows)

    if df.empty:
        fatal(
            "No valid results after filtering ERROR entries.\n"
            " Check that llama-perplexity ran correctly and --args flags are appropriate for your hardware.",
            ERR_PARSE_FAILED,
        )

    color_map, marker_map, source_map = get_source_styles(df)
    df["Source"] = df["Quantization"].apply(lambda n: get_source(n, source_map))
    df["Label"] = df["Quantization"].apply(lambda n: get_label(n, model_name))
    df = df.sort_values("MDL_norm")

    # --- Scatter plots ---
    mdl_kld_plot(df, out_dir / f"mdl_kld_plot_{model_name}.png", color_map, marker_map, model_name)

    # --- Efficiency report ---
    generate_markdown_report(df, model_name, report_path)

    print("\nAll done.")




if __name__ == "__main__":
    main()
