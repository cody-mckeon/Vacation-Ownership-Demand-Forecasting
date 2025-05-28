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
    p.add_argument('--threshold', type=float, default=0.3,
                   help="Drift score above which to fail")
    args = p.parse_args()

    # 1. Load the two slices
    df_base = pd.read_parquet(args.baseline)
    df_curr = pd.read_parquet(args.current)

    # 2. Build & run an Evidently Report
    report = Report(metrics=[DataDriftPreset()])
    report.run(baseline_data=df_base, current_data=df_curr)

    # 3. Save an HTML report for review
    report.save_html(args.report)

    # 4. Inspect the drift scores and exit non-zero if any exceed threshold
    rd = report.as_dict()
    drift_info = rd['metrics'][0]['result']['metrics']  # DataDriftPreset section
    drifted = []
    for feat in drift_info:
        score = feat['value']['drift_score']
        if score > args.threshold:
            drifted.append((feat['feature_name'], score))

    if drifted:
        print("⚠️ Drift detected on features:")
        for f, s in drifted:
            print(f"   • {f}: {s:.3f} > {args.threshold}")
        exit(1)
    else:
        print("✅ No significant drift detected.")
        exit(0)

if __name__ == "__main__":
    main()
