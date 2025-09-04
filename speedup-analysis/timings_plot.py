#!/usr/bin/env python3
# plot.py

# Usage: python plot.py <input_csv> <output_png>

# Example: python plot.py "parsed_results.csv" "speedup_plot.png"

import sys, os
import pandas as pd
import matplotlib.pyplot as plt

def main(csv_path: str = "speedup_results.csv", out_png: str = "speedup_plot.png"):
    # Load CSV and drop junk rows (e.g., trailing 'f' line)
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    needed = {"mesh_preset_key","mode","processes","mean"}
    df = df[[c for c in df.columns if c in needed]].copy()
    # Coerce numerics; drop bad rows
    df["processes"] = pd.to_numeric(df["processes"], errors="coerce")
    df["mean"]    = pd.to_numeric(df["mean"],    errors="coerce")
    df = df.dropna(subset=["mesh_preset_key","mode","processes","mean"])

    # Normalize mesh_preset basename and mesh name
    df["mesh_preset"] = df["mesh_preset_key"].apply(os.path.basename)
    df["mesh"]   = df["mesh_preset"].str.replace(".toml", "", regex=False)

    # Build label "sequential (1)" or "MPI (N)"
    def mk_label(row):
        m = str(row["mode"]).strip().lower()
        if m == "sequential":
            return "sequential (1)"
        return f"MPI ({int(row['processes'])})"
    df["label"] = df.apply(mk_label, axis=1)

    # Average in case there are multiple lines for the same (mesh, label)
    agg = (
        df.groupby(["mesh","label"], as_index=False)["mean"]
          .mean()
          .rename(columns={"mean":"elapsed_mean"})
    )

    # Pivot to have one curve per label
    pivot = agg.pivot(index="mesh", columns="label", values="elapsed_mean")
    
    # Sort by increasing time (using sequential)
    if "sequential (1)" in pivot.columns:
        pivot = pivot.sort_values("sequential (1)")

    # Plot
    plt.figure(figsize=(10, 5))
    for col in pivot.columns:
        plt.plot(pivot.index, pivot[col], marker="o", label=col)

    plt.xlabel("mesh (mesh_preset)")
    plt.ylabel("elapsed time [s]")
    plt.title("Timing per mesh (mean over runs)")
    plt.xticks(rotation=45, ha="right")
    plt.legend(title="mode (processes)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    print(f"Saved plot -> {out_png}")
    try:
        plt.show()
    except Exception:
        pass

if __name__ == "__main__":
    csv = sys.argv[1] if len(sys.argv) >= 2 else "parsed_results.csv"
    png = sys.argv[2] if len(sys.argv) >= 3 else "timings_plot.png"
    main(csv, png)
