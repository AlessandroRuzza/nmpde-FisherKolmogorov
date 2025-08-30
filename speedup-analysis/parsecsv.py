#!/usr/bin/env python3
# parsecsv.py

# python3 parsecsv.py -i ../speedup_results.csv -o parsed_results.csv 

import argparse
import os
import pandas as pd

def main():
    ap = argparse.ArgumentParser(description="Average multiple runs from speedup_results.csv")
    ap.add_argument("--input", "-i", default="speedup_results.csv", help="Path to speedup_results.csv")
    ap.add_argument("--output", "-o", default="parsed_results.csv", help="Output CSV path (default: <input> with _avg suffix)")
    ap.add_argument("--keep-path", action="store_true",
                    help="Keep full config path instead of just basename")
    args = ap.parse_args()

    in_csv = args.input
    out_csv = args.output or os.path.splitext(in_csv)[0] + "_avg.csv"

    df = pd.read_csv(in_csv, on_bad_lines="skip")

    # Normalise columns
    expected_cols = {"config","mode","processes","run","elapsed_s"}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing columns: {missing}")

    # Conversions & cleaning
    df["processes"] = pd.to_numeric(df["processes"], errors="coerce")
    df["run"]     = pd.to_numeric(df["run"], errors="coerce")
    df["elapsed_s"] = pd.to_numeric(df["elapsed_s"], errors="coerce")
    df = df.dropna(subset=["config","mode","processes","run","elapsed_s"])

    # Only use file name
    if args.keep_path:
        df["config_key"] = df["config"]
    else:
        df["config_key"] = df["config"].apply(lambda p: os.path.basename(str(p)))

    # Aggregate
    grp_cols = ["config_key","mode","processes"]
    agg = (df.groupby(grp_cols)["elapsed_s"]
             .agg(mean="mean", std="std", count="count")
             .reset_index())

    # Calculate speedup vs sequential 1-process 
    ref = (agg[(agg["mode"]=="sequential") & (agg["processes"]==1)]
             .loc[:, ["config_key","mean"]]
             .rename(columns={"mean":"seq1_mean"}))
    merged = agg.merge(ref, on="config_key", how="left")
    merged["speedup_vs_seq1"] = merged["seq1_mean"] / merged["mean"]
    merged["efficiency"] = merged["speedup_vs_seq1"] / merged["processes"]

    # Sort for convenience
    merged = merged.sort_values(by=["config_key","mode","processes"]).reset_index(drop=True)

    # Save
    merged.to_csv(out_csv, index=False, float_format="%.4f")
    print(f"[OK] Wrote: {out_csv}")

    # Print summary
    with pd.option_context("display.max_rows", None, "display.width", 120):
        print(merged)

if __name__ == "__main__":
    main()
