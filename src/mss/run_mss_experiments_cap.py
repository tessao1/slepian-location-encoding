"""Arctic MSS Reconstruction: Slepian cap and Vanilla SH experiments"""

import os
import subprocess
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from glob import glob

# Setup paths
SCRIPT_DIR = Path(__file__).parent.resolve()
PARENT_DIR = SCRIPT_DIR.parent
SRC_DIR = PARENT_DIR.parent
ROOT_DIR = SRC_DIR.parent

# Data path (must be set by user)
DATA_PATH = os.environ.get(
    "MSS_DATA_PATH",
    "C:/repos/slepian-location-encoding/slepian-location-encoding/src/datasets/Experiment_data"
)

if not os.path.isdir(DATA_PATH):
    print(f"Error: MSS data not found at {DATA_PATH}")
    print("Set MSS_DATA_PATH environment variable to the correct path.")
    sys.exit(1)

# Output directories
RESULTS_DIR = ROOT_DIR / "results" / "mss"
CACHE_DIR = ROOT_DIR / "cache" / "mss"
FIGURES_DIR = RESULTS_DIR / "figures"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Training parameters
# ARCHS = ["mlp", "resmlp", "siren", "glu"]
ARCHS = ["mlp"]

N_RUNS = 1
EPOCHS = 200
BATCH_SIZE = 1024
PATIENCE = 30
NUM_WORKERS = 2
LABEL_FRACS = "1.0"

# Slepian parameters
L_GLOBAL = [0, 5, 10, 15]
CAP_RADIUS = 25.0
LAMBDA_THRESH = 0.05

print("Arctic MSS Reconstruction Experiments")
print("=" * 38)

# Slepian experiments
for arch in ARCHS:
    for L in [40]:
        for l in L_GLOBAL:
            # Calculate num_modes (Shannon number)
            theta_rad = CAP_RADIUS * np.pi / 180.0
            shannon = int((L + 1)**2 * (1 - np.cos(theta_rad)) / 2)
            num_modes = min(shannon, (L + 1)**2)
            
            print(f"-> Slepian L={L}, arch={arch} (modes={num_modes})")
            
            cmd = [
                sys.executable,
                str(PARENT_DIR / "mss/train_mss_slepian.py"),
                "--data-path", DATA_PATH,
                "--L-global", str(l),
                "--L-slepian", str(L),
                "--cap-radius", str(CAP_RADIUS),
                "--num-modes", str(num_modes),
                "--lambda-thresh", str(LAMBDA_THRESH),
                "--arch", arch,
                "--batch-size", str(BATCH_SIZE),
                "--epochs", str(EPOCHS),
                "--patience", str(PATIENCE),
                "--lr", "1e-3",
                "--hidden-dim", "128",
                "--dropout", "0.1",
                "--num-workers", str(NUM_WORKERS),
                "--label-fracs", LABEL_FRACS,
                "--n-runs", str(N_RUNS),
                "--seed", "42",
                "--cache-dir", str(CACHE_DIR),
                "--csv-path", str(RESULTS_DIR / f"slepian_cap_L{L}_l{l}_{arch}.csv"),
                "--fig-dir", str(FIGURES_DIR / f"slepian_cap_L{L}_l{l}_{arch}")
            ]
            
            subprocess.run(cmd, check=True)


# Aggregate results
csv_files = glob(str(RESULTS_DIR / "*.csv"))
csv_files = [f for f in csv_files if 'aggregated' not in f]

if csv_files:
    df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
    aggregated_path = RESULTS_DIR / "aggregated_results_hybrid_cap.csv"
    df.to_csv(aggregated_path, index=False)
    print(f"Aggregated {len(csv_files)} files -> aggregated_results_hybrid_cap.csv")
    
    # Remove individual CSV files after aggregation
    for f in csv_files:
        os.remove(f)

print(f"Done. Results: {RESULTS_DIR / 'aggregated_results_hybrid_cap.csv'}")