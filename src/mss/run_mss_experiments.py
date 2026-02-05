#!/usr/bin/env python
"""
Arctic MSS Reconstruction: Slepian and Vanilla SH experiments
Converted from bash script for Windows compatibility
"""

import os
import sys
import subprocess
import numpy as np
import pandas as pd
from glob import glob
from pathlib import Path

def main():
    # Setup paths
    script_dir = Path(__file__).parent.resolve()
    parent_dir = script_dir.parent
    src_dir = parent_dir.parent
    root_dir = src_dir.parent
    
    # Data path (must be set by user or via environment variable)
    data_path = os.environ.get(
        'MSS_DATA_PATH', 
        r"C:\slepian-location-encoding\Experiment_data\Experiment_data"
    )
    
    if not os.path.isdir(data_path):
        print(f"Error: MSS data not found at {data_path}")
        print("Set MSS_DATA_PATH environment variable to the correct path.")
        sys.exit(1)
    
    # Output directories
    results_dir = root_dir / "results" / "mss"
    cache_dir = root_dir / "cache" / "mss"
    figures_dir = results_dir / "figures"
    
    results_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # Training parameters
    archs = ["mlp"]  # Can expand to: ["mlp", "resmlp", "siren", "glu"]
    n_runs = 2
    epochs = 200
    batch_size = 2048
    patience = 30
    num_workers = 2
    label_fracs = "1.0"
    
    # Slepian parameters
    l_global = 10
    lambda_thresh = 0.05
    lat_min = 65.0
    
    print("Arctic MSS Reconstruction Experiments")
    print("=" * 38)
    
    # Slepian experiments
    for arch in archs:
        for L in [40, 80, 120]:
            
            print(f"-> Slepian L={L}, arch={arch})")
            
            cmd = [
                sys.executable,
                str(parent_dir / "train_mss_slepian_masked.py"),
                "--data-path", data_path,
                "--L-global", str(l_global),
                "--L-slepian", str(L),
                "--lat-min", str(lat_min),
                "--lambda-thresh", str(lambda_thresh),
                "--arch", arch,
                "--batch-size", str(batch_size),
                "--epochs", str(epochs),
                "--patience", str(patience),
                "--lr", "1e-3",
                "--hidden-dim", "128",
                "--dropout", "0.1",
                "--num-workers", str(num_workers),
                "--label-fracs", label_fracs,
                "--n-runs", str(n_runs),
                "--seed", "42",
                "--cache-dir", str(cache_dir),
                "--csv-path", str(results_dir / f"slepian_L{L}_{arch}.csv"),
                "--fig-dir", str(figures_dir / f"slepian_L{L}_{arch}"),
            ]
            
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error running command: {e}")
                sys.exit(1)
    
    # Vanilla SH experiments
    for arch in archs:
        for L in [10, 40]:
            print(f"-> Vanilla SH L={L}, arch={arch}")
            
            cmd = [
                sys.executable,
                str(parent_dir / "train_mss_sh_vanilla.py"),
                "--data-path", data_path,
                "--L", str(L),
                "--arch", arch,
                "--batch-size", str(batch_size),
                "--epochs", str(epochs),
                "--patience", str(patience),
                "--lr", "1e-3",
                "--hidden-dim", "128",
                "--dropout", "0.1",
                "--num-workers", str(num_workers),
                "--label-fracs", label_fracs,
                "--n-runs", str(n_runs),
                "--seed", "42",
                "--csv-path", str(results_dir / f"vanilla_sh_L{L}_{arch}.csv"),
                "--fig-dir", str(figures_dir / f"vanilla_sh_L{L}_{arch}"),
            ]
            
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error running command: {e}")
                sys.exit(1)
    
    # Aggregate results
    print("\nAggregating results...")
    csv_files = glob(str(results_dir / "*.csv"))
    csv_files = [f for f in csv_files if "aggregated" not in f]
    
    if csv_files:
        dfs = [pd.read_csv(f) for f in csv_files]
        df = pd.concat(dfs, ignore_index=True)
        output_path = results_dir / "aggregated_results.csv"
        df.to_csv(output_path, index=False)
        print(f"Aggregated {len(csv_files)} files -> aggregated_results.csv")
        
        # Remove individual CSV files
        for f in csv_files:
            os.remove(f)
    
    print(f"\nDone. Results: {results_dir / 'aggregated_results.csv'}")

if __name__ == "__main__":
    main()
