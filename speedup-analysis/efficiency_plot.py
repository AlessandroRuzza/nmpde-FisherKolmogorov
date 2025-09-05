#!/usr/bin/env python3
import sys, os
import pandas as pd
import matplotlib.pyplot as plt

def main(csv_path, out_png):
    # Load CSV and keep only the necessary columns (no averages)
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)

    cols_needed = ["mesh_preset_key", "processes", "efficiency"]
    for c in cols_needed:
        if c not in df.columns:
            raise ValueError(f"Column '{c}' not found in CSV")

    # Numeric casting
    df["processes"] = pd.to_numeric(df["processes"], errors="coerce")
    df["efficiency"] = pd.to_numeric(df["efficiency"], errors="coerce")

    # Clean invalid rows
    df = df.dropna(subset=["mesh_preset_key", "processes", "efficiency"])

    # Mesh label: basename without .toml
    df["mesh"] = df["mesh_preset_key"].apply(os.path.basename).str.replace(".toml", "", regex=False)

    df = df.drop_duplicates(subset=["mesh", "processes"], keep="first")
    pivot = df.pivot(index="processes", columns="mesh", values="efficiency")

    # Sort by process
    pivot = pivot.sort_index()

    # Plot
    plt.figure(figsize=(10, 5))
    for col in pivot.columns:
        plt.plot(pivot.index, pivot[col], marker="o", label=col)

    plt.xlabel("processes")
    plt.ylabel("efficiency vs sequential (1 process)")
    plt.title("Efficiency by processes")
    plt.xticks(pivot.index.tolist())
    plt.legend(title="mesh", ncol=2)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    print(f"Saved plot -> {out_png}")
    try:
        plt.show()
    except Exception:
        pass

if __name__ == "__main__":
    csv = sys.argv[1] if len(sys.argv) >= 2 else "parsed_results.csv"
    png = sys.argv[2] if len(sys.argv) >= 3 else "efficiency_vs_processes.png"
    main(csv, png)
