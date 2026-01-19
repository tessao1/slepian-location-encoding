import subprocess
import sys

# Common parameters
config = {
    'dataset': 'highreslandoceandataset',
    'nn': 'mlp',
    'sampling_method': 'fibonacci',
    'num_samples': 10000,
    'max_epochs': 500,
    'sh_max_degree': 10
}

# Values to loop through
pe_config = {
    'sphericalharmonics': [40],
    # 'slepian': [40, 80, 120],
    'slepianhybrid': [40, 80, 120]
    
}

total_experiments = sum(len(values) for values in pe_config.values())
current_experiment = 0

for pe, legendre_polys_values in pe_config.items():
    for L in legendre_polys_values:
        current_experiment += 1
        print("=" * 60)
        print(f"Experiment {current_experiment}/{total_experiments}")
        print(f"PE: {pe}, Legendre Polys: {L}")
        print("=" * 60)
        
        cmd = [
            sys.executable, 'train.py',
            '--dataset', config['dataset'],
            '--pe', pe,
            '--nn', config['nn'],
            '--legendre-polys', str(L),
            '--sh-max-degree', str(config['sh_max_degree']),
            '--sampling-method', config['sampling_method'],
            '--num-samples', str(config['num_samples']),
            '--max-epochs', str(config['max_epochs']),
            '--matplotlib',
            '--log-wandb',
            '--save-model',
            '--seed', '42'
        ]
        
        result = subprocess.run(cmd)
        
        if result.returncode != 0:
            print(f"Error: Training failed for PE={pe}, legendre-polys={L}")
            sys.exit(1)
        
        print(f"✓ Completed PE={pe}, legendre-polys={L}\n")

print("=" * 60)
print("All experiments completed!")
print(f"Total experiments run: {total_experiments}")