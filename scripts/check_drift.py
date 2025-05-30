#!/usr/bin/env python
import argparse
import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--baseline', required=True)
    p.add_argument('--current',  required=True)
    p.add_argument('--report',   required=True)
    args = p.parse_args()

    # 1. Load the two slices
    df_base = pd.read_parquet(args.baseline)
    df_curr = pd.read_parquet(args.current)

    # 2. Drop date columns if present
    for col in ['booking_date', 'reservation_status_date']:
        if col in df_base.columns:
            df_base = df_base.drop(columns=col)
            df_curr = df_curr.drop(columns=col)

    # 3. Filter out columns that are all-null in either slice
    common_cols = [
        col for col in df_base.columns
        if df_base[col].notna().any() and df_curr[col].notna().any()
    ]
    if not common_cols:
        print("⚠️ No valid features for drift detection after filtering. Exiting.")
        exit(0)

    df_base = df_base[common_cols]
    df_curr = df_curr[common_cols]

    # 2. Build & run an Evidently Report
    report = Report(metrics=[DataDriftPreset()])
    try:
        result = report.run(reference_data=df_base, current_data=df_curr)
    except ZeroDivisionError:
        print("⚠️ No features to evaluate for drift. Exiting.")
        exit(0)

    # 3. Save an HTML report for review
    result.save_html(args.report)

    print(f"✅ Drift report generated: {args.report}")
    exit(0)

if __name__ == "__main__":
    main()
