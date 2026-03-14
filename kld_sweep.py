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

Dependencies: pandas, matplotlib, adjustText, scipy
    pip install pandas matplotlib adjustText scipy
"""

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from adjustText import adjust_text

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
ERR_PLOT_FAILED      = 10  # Plotting failed (e.g. not enough valid results)

# Minimum expected logits file size: 1 MB — anything smaller is likely partial/corrupt
LOGITS_MIN_BYTES = 1 * 1024 * 1024

# ---------------------------------------------------------------------------
# Colours — extended palette, auto-assigned to unknown quantizers
# ---------------------------------------------------------------------------
PALETTE = [
    "#2ECC71", "#E74C3C", "#9B59B6", "#F39C12",
    "#3498DB", "#1ABC9C", "#E67E22", "#C0392B",
    "#8E44AD", "#16A085", "#D35400", "#2980B9",
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

def fatal(msg: str, code: int):
    """Print a clear error and exit with a documented code."""
    print(f"\n[ERROR E{code:02d}] {msg}", file=sys.stderr)
    print(f"  -> See FAQ.md entry E{code:02d} for troubleshooting.", file=sys.stderr)
    sys.exit(code)


def warn(msg: str, code: int):
    """Print a non-fatal warning with its FAQ reference."""
    print(f"\n[WARN  W{code:02d}] {msg}", file=sys.stderr)
    print(f"  -> See FAQ.md entry E{code:02d} for details.", file=sys.stderr)


def parse_args():
    p = argparse.ArgumentParser(description="KLD sweep for GGUF quants.")
    p.add_argument("--exe",           required=True,  help="Path to llama-perplexity binary")
    p.add_argument("--baseline",      required=True,  help="Path to baseline GGUF (BF16, F16, Q8_0 — first shard if split)")
    p.add_argument("--quants",        required=True,  help="Directory containing quant GGUFs")
    p.add_argument("--dataset",       required=True,  help="Primary evaluation dataset (.txt)")
    p.add_argument("--output",        required=True,  help="Output directory for results and plots")
    p.add_argument("--logits",        default=None,   help="Path to existing logits file (optional — auto-generated in --output if not provided, reused on resume)")
    p.add_argument("--args",          default="-t 7 -c 512 -ngl 99",
                                                      help="Extra flags passed to llama-perplexity for quant evaluation. Must be quoted: --args=\"-t 7 -c 512 -ngl 36\"")
    p.add_argument("--args-baseline", default=None,
                                                      help="Extra flags passed to llama-perplexity for baseline logits generation only. Falls back to --args if not provided. Useful when the baseline does not fit in VRAM with the same settings as the quants.")
    p.add_argument("--model-name",    default=None,   help="Short model name used in plot titles")

    return p.parse_args()


def gib(path: Path) -> float:
    return path.stat().st_size / (1024 ** 3)


def validate_paths(exe: Path, baseline: Path, dataset: Path, quant_dir: Path):
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


def logits_meta_path(logits: Path) -> Path:
    return logits.with_suffix(".bin.meta")


def write_logits_meta(logits: Path, dataset: Path):
    """Write sidecar metadata after successful logits generation."""
    meta = {"dataset": str(dataset.resolve()), "dataset_size": dataset.stat().st_size}
    logits_meta_path(logits).write_text(json.dumps(meta), encoding="utf-8")


def check_logits(logits: Path, dataset: Path):
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
        current_size  = dataset.stat().st_size
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


def generate_logits(exe: Path, baseline: Path, dataset: Path, logits: Path, extra: list):
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
        if logits.exists():
            logits.unlink()
            print(f"[logits] Partial logits file deleted: {logits}")
        fatal(
            f"Logits generation failed (exit code {rc}).\n"
            "  Common causes:\n"
            "    - Not enough RAM to load the BF16 model\n"
            "    - Incorrect --args flags (check -ngl value for your VRAM)\n"
            "    - Corrupted BF16 GGUF file\n"
            "  Last 20 lines of output:\n"
            + "\n".join(raw.splitlines()[-20:]),
            ERR_LOGITS_FAILED,
        )

    if not logits.exists() or logits.stat().st_size < LOGITS_MIN_BYTES:
        if logits.exists():
            logits.unlink()
        fatal(
            "Logits generation appeared to succeed (exit 0) but output file is missing or empty.\n"
            "  Check available disk space.",
            ERR_LOGITS_PARTIAL,
        )

    write_logits_meta(logits, dataset)
    print(f"[logits] Generated: {logits} ({gib(logits):.2f} GiB)")


def parse_output(raw: str) -> tuple:
    """Extract PPL and KLD from llama-perplexity output.
    Compatible with mainline llama.cpp and ik_llama (output format is identical)."""
    ppl = kld = "ERROR"
    m = re.search(r"Mean PPL\(Q\)\s*:\s*([0-9]+\.[0-9]+)", raw)
    if m:
        ppl = m.group(1)
    m = re.search(r"Mean\s+KLD:\s+([0-9]+\.[0-9]+)", raw)
    if m:
        kld = m.group(1)
    return ppl, kld


def load_csv(path: Path) -> dict:
    """Load existing results as {quantization_name: row_dict}."""
    results = {}
    if not path.exists():
        return results
    try:
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            expected = ["Quantization", "Size_GiB", "PPL_Score", "KLD_Score"]
            if reader.fieldnames != expected:
                fatal(
                    f"CSV has unexpected columns: {reader.fieldnames}\n"
                    f"  Expected: {expected}\n"
                    f"  File: {path}\n"
                    "  Rename or delete the file to start fresh.",
                    ERR_CSV_CORRUPT,
                )
            for row in reader:
                results[row["Quantization"]] = row
    except Exception as e:
        fatal(
            f"Could not read existing CSV: {path}\n  Reason: {e}\n"
            "  If the file is corrupt, delete it and re-run.",
            ERR_CSV_CORRUPT,
        )
    return results


def save_csv(path: Path, results: dict):
    rows = sorted(results.values(), key=lambda r: float(r["Size_GiB"]))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["Quantization", "Size_GiB", "PPL_Score", "KLD_Score"])
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------------------
# Source / label helpers
# ---------------------------------------------------------------------------

def get_source_colors(df: pd.DataFrame) -> dict:
    """Auto-assign colours to quantizer prefixes found in filenames."""
    prefixes = {}
    for name in df["Quantization"]:
        prefix = name.split("_")[0] if "_" in name else name.split(".")[0]
        prefixes.setdefault(prefix, None)
    return {prefix: PALETTE[i % len(PALETTE)] for i, prefix in enumerate(prefixes)}


def get_source(name: str, color_map: dict) -> str:
    for prefix in color_map:
        if name.startswith(prefix):
            return prefix
    return "other"


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

def scatter_plot(df: pd.DataFrame, metric: str, ylabel: str, title: str,
                 out_path: Path, color_map: dict, model_name: str):
    if df.empty:
        warn(f"No valid data for {metric} scatter plot -- skipping.", ERR_PLOT_FAILED)
        return
    try:
        fig, ax = plt.subplots(figsize=(15, 9), dpi=200)
        texts = []
        for _, row in df.iterrows():
            c = color_map.get(row["Source"], "#888888")
            ax.scatter(row["Size_GiB"], row[metric],
                       color=c, s=75, alpha=0.9, zorder=10,
                       edgecolors="white", linewidths=0.5)
            texts.append(ax.text(row["Size_GiB"], row[metric],
                                 row["Label"], fontsize=8, alpha=0.95, zorder=11))
        adjust_text(texts, ax=ax,
                    force_text=(0.6, 0.9), force_points=(0.2, 0.2),
                    arrowprops=dict(arrowstyle="-", color="gray", lw=0.4, alpha=0.5),
                    lim=200)
        ax.set_xlabel("Model Size (GiB)", fontsize=12, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=12, fontweight="bold")
        ax.grid(True, color=GRID_COLOR, linestyle="--", linewidth=0.5, alpha=0.6)
        ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
        legend_handles = [
            mlines.Line2D([], [], marker="o", color="w",
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
        warn(f"Scatter plot failed: {e}", ERR_PLOT_FAILED)


def efficiency_report(df: pd.DataFrame, model_name: str, out_path: Path):
    if len(df) < 2:
        warn(
            f"Not enough valid results ({len(df)}) to compute efficiency scores.\n"
            "  Need at least 2 quants with valid KLD.",
            ERR_PLOT_FAILED,
        )
        return
    d = df.copy()
    s_min, s_max = d["Size_GiB"].min(), d["Size_GiB"].max()
    k_min, k_max = d["KLD_Score"].min(), d["KLD_Score"].max()
    if s_max == s_min or k_max == k_min:
        warn("All values are identical -- efficiency score normalization would divide by zero.",
             ERR_PLOT_FAILED)
        return
    d["norm_size"] = (d["Size_GiB"] - s_min) / (s_max - s_min)
    d["norm_kld"]  = (d["KLD_Score"] - k_min) / (k_max - k_min)
    d["eff"]       = np.sqrt(d["norm_size"] ** 2 + d["norm_kld"] ** 2)
    d = d.sort_values("eff").reset_index(drop=True)

    W = 105
    lines = [
        "=" * W,
        f"{'EFFICIENCY RANKINGS -- ' + model_name:^{W}}",
        f"{'Euclidean Distance from (0,0) -- lower is better':^{W}}",
        "=" * W,
        f"{'Rank':<5} {'Quantization':<48} {'Size (GiB)':<12} {'KLD':<12} {'Eff. Score'}",
        "-" * W,
    ]
    for i, (_, row) in enumerate(d.iterrows(), 1):
        prefix = ">> " if i == 1 else "   "
        lines.append(
            f"{prefix}{i:<3} {row['Quantization']:<48} "
            f"{row['Size_GiB']:<12.3f} {row['KLD_Score']:<12.6f} {row['eff']:.6f}"
        )
    lines += ["=" * W, f"WINNER: {d.iloc[0]['Quantization']}", "=" * W]
    report = "\n".join(lines)
    print(report)
    try:
        out_path.write_text(report, encoding="utf-8")
        print(f"\nSaved: {out_path}")
    except Exception as e:
        warn(f"Could not write efficiency report: {e}", ERR_PLOT_FAILED)


# ---------------------------------------------------------------------------
# Sweep runner
# ---------------------------------------------------------------------------

def run_sweep(exe: Path, quant_files: list, dataset: Path, logits: Path,
              extra: list, csv_path: Path, label: str) -> dict:
    results = load_csv(csv_path)
    errors  = []

    for qf in quant_files:
        name = qf.stem
        size = round(gib(qf), 3)

        existing = results.get(name)
        if existing and existing.get("KLD_Score") not in (None, "", "ERROR"):
            print(f"[skip{label}] {name}")
            continue

        cmd = (
            [str(exe), "-m", str(qf), "-f", str(dataset)]
            + extra
            + ["--kl-divergence", "--kl-divergence-base", str(logits)]
        )
        raw, rc = run(cmd, f"Testing{label}: {name} ({size} GiB)")

        if rc != 0:
            warn(
                f"llama-perplexity returned exit code {rc} for: {qf.name}\n"
                "  Last 10 lines:\n" + "\n".join(raw.splitlines()[-10:]) + "\n"
                "  Result recorded as ERROR -- sweep will continue.",
                ERR_SUBPROCESS,
            )
            results[name] = {"Quantization": name, "Size_GiB": size,
                             "PPL_Score": "ERROR", "KLD_Score": "ERROR"}
            errors.append(name)
            save_csv(csv_path, results)
            time.sleep(1)
            continue

        ppl, kld = parse_output(raw)

        if ppl == "ERROR" or kld == "ERROR":
            warn(
                f"Could not parse PPL/KLD from output for: {qf.name}\n"
                "  Last 20 lines of output:\n" + "\n".join(raw.splitlines()[-20:]) + "\n"
                "  This may mean llama-perplexity output format has changed.\n"
                "  Result recorded as ERROR -- sweep will continue.",
                ERR_PARSE_FAILED,
            )
            errors.append(name)

        print(f"-> PPL: {ppl} | KLD: {kld}")
        results[name] = {"Quantization": name, "Size_GiB": size,
                         "PPL_Score": ppl, "KLD_Score": kld}
        save_csv(csv_path, results)
        time.sleep(1)

    if errors:
        print(f"\n[!] {len(errors)} file(s) recorded as ERROR:")
        for e in errors:
            print(f"    - {e}")
        print("  Re-run the script to retry ERROR entries.")

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

    # --- Validate ---
    validate_paths(exe, baseline, dataset, quant_dir)

    # --- Find quants ---
    # Always exclude the baseline file itself.
    # For split-shard baselines (e.g. -00001-of-00002), also exclude other shards by prefix.
    # Prefix matching is intentionally limited to shard patterns to avoid accidentally
    # excluding quants that share a name fragment with the baseline (e.g. a Q8_0 baseline
    # vs Q8_0 quants).
    baseline_stem = baseline.stem
    is_shard = bool(re.search(r"-\d{5}-of-\d{5}", baseline_stem))
    baseline_shard_prefix = baseline_stem.split("-00")[0].lower() if is_shard else None

    all_gguf = sorted(quant_dir.glob("*.gguf"), key=lambda f: f.stat().st_size)
    quant_files = []
    skipped = []
    for f in all_gguf:
        is_baseline_file  = f.resolve() == baseline.resolve()
        is_baseline_shard = baseline_shard_prefix and baseline_shard_prefix in f.stem.lower()
        if is_baseline_file or is_baseline_shard:
            skipped.append(f.name)
        else:
            quant_files.append(f)
    if skipped:
        print(f"\n[quants] Excluded (baseline/shard detected): {', '.join(skipped)}")
    if not quant_files:
        fatal(
            f"No .gguf files found in: {quant_dir}\n"
            "  Make sure the path is correct and the files have a .gguf extension.",
            ERR_NO_QUANTS,
        )
    print(f"\nFound {len(quant_files)} quant file(s) in {quant_dir}")

    # --- Logits ---
    if extra_base != extra:
        print(f"[logits] Using --args-baseline for logits generation: {' '.join(extra_base)}")

    if logits.exists():
        if check_logits(logits, dataset):
            print(f"[logits] Using existing logits: {logits} ({gib(logits):.2f} GiB)")
        else:
            generate_logits(exe, baseline, dataset, logits, extra_base)
            time.sleep(2)
    else:
        generate_logits(exe, baseline, dataset, logits, extra_base)
        time.sleep(2)

    # --- Primary sweep ---
    run_sweep(exe, quant_files, dataset, logits, extra, csv_path, "")
    print(f"\nPrimary results saved to {csv_path}")

    # --- Build DataFrame ---
    df = pd.read_csv(csv_path)
    df = df[df["KLD_Score"] != "ERROR"].copy()
    df["KLD_Score"] = pd.to_numeric(df["KLD_Score"], errors="coerce")
    df["PPL_Score"] = pd.to_numeric(df["PPL_Score"], errors="coerce")
    df["Size_GiB"]  = pd.to_numeric(df["Size_GiB"],  errors="coerce")
    df = df.dropna(subset=["KLD_Score", "Size_GiB"])

    if df.empty:
        fatal(
            "No valid results after filtering ERROR entries.\n"
            "  Check that llama-perplexity ran correctly and --args flags are appropriate for your hardware.",
            ERR_PARSE_FAILED,
        )

    color_map = get_source_colors(df)
    df["Source"] = df["Quantization"].apply(lambda n: get_source(n, color_map))
    df["Label"]  = df["Quantization"].apply(lambda n: get_label(n, model_name))
    df = df.sort_values("Size_GiB")

    # --- Scatter plots ---
    scatter_plot(df, "KLD_Score", "KL Divergence (lower is better)",
                 f"{model_name} -- KL Divergence vs Size",
                 out_dir / f"kld_plot_{model_name}.png", color_map, model_name)

    scatter_plot(df, "PPL_Score", "Perplexity (lower is better)",
                 f"{model_name} -- Perplexity vs Size",
                 out_dir / f"ppl_plot_{model_name}.png", color_map, model_name)

    # --- Efficiency report ---
    efficiency_report(df, model_name, out_dir / f"{model_name}_efficiency.txt")

    print("\nAll done.")




if __name__ == "__main__":
    main()
