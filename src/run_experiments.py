import subprocess
import sys

# Common parameters
config = {
    'dataset': 'highreslandoceandataset',
    'nn': 'mlp',
    'sampling_method': 'fibonacci',
    'num_samples': 100000,
    'max_epochs': 200,
    'legendre-polys': 30
}

# Values to loop through
pe_list = ['sphericalharmonics', 'slepian', 'slepianhybrid', 'wavelets', 'direct']
sh_max_degrees = [5, 10, 15]

total_experiments = 0
for pe in pe_list:
    if pe == 'slepianhybrid':
        total_experiments += len(sh_max_degrees)
    else:
        total_experiments += 1
current_experiment = 0

for pe in pe_list:
    if pe in ['wavelets', 'direct']:
        current_experiment += 1
        print("=" * 60)
        print(f"Experiment {current_experiment}/{total_experiments}")
        print(f"PE: {pe}")
        print("=" * 60)
        
        cmd = [
            sys.executable, 'train.py',
            '--dataset', config['dataset'],
            '--pe', pe,
            '--nn', config['nn'],
            '--sampling-method', config['sampling_method'],
            '--num-samples', str(config['num_samples']),
            '--max-epochs', str(config['max_epochs']),
            '--save-model',
            '--matplotlib',
            '--log-wandb',
            '--seed', '42'
        ]
        
        result = subprocess.run(cmd)
        
        if result.returncode != 0:
            print(f"Error: Training failed for PE={pe}")
            sys.exit(1)
        
        print(f"Completed PE={pe}\n")
    
    elif pe == 'slepianhybrid':
        for sh_degree in sh_max_degrees:
            current_experiment += 1
            print("=" * 60)
            print(f"Experiment {current_experiment}/{total_experiments}")
            print(f"PE: {pe}, SH Max Degree: {sh_degree}")
            print("=" * 60)
            
            cmd = [
                sys.executable, 'train.py',
                '--dataset', config['dataset'],
                '--pe', pe,
                '--nn', config['nn'],
                '--legendre-polys', str(config['legendre-polys']),
                '--sh-max-degree', str(sh_degree),
                '--sampling-method', config['sampling_method'],
                '--num-samples', str(config['num_samples']),
                '--max-epochs', str(config['max_epochs']),
                '--save-model',
                '--matplotlib',
                '--log-wandb',
                '--seed', '42'
            ]
            
            result = subprocess.run(cmd)
            
            if result.returncode != 0:
                print(f"Error: Training failed for PE={pe}, sh-max-degree={sh_degree}")
                sys.exit(1)
            
            print(f"Completed PE={pe}, sh-max-degree={sh_degree}\n")
    
    else:
        current_experiment += 1
        print("=" * 60)
        print(f"Experiment {current_experiment}/{total_experiments}")
        print(f"PE: {pe}")
        print("=" * 60)
        
        cmd = [
            sys.executable, 'train.py',
            '--dataset', config['dataset'],
            '--pe', pe,
            '--nn', config['nn'],
            '--legendre-polys', str(config['legendre-polys']),
            '--sampling-method', config['sampling_method'],
            '--num-samples', str(config['num_samples']),
            '--max-epochs', str(config['max_epochs']),
            '--save-model',
            '--matplotlib',
            '--log-wandb',
            '--seed', '42'
        ]
        
        result = subprocess.run(cmd)
        
        if result.returncode != 0:
            print(f"Error: Training failed for PE={pe}")
            sys.exit(1)
        
        print(f"Completed PE={pe}, legendre-polys={config['legendre-polys']}\n")

print("=" * 60)
print("All experiments completed!")
print(f"Total experiments run: {total_experiments}")