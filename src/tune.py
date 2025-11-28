import os.path
from locationencoder import LocationEncoder
from data import LandOceanDataModule
import lightning as pl
import optuna
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
import pandas as pd
import yaml
import torch

TUNE_RESULTS_DIR = "results/tune"

torch.set_float32_matmul_precision('high')

import logging
logging.getLogger("lightning").setLevel(logging.ERROR)

def get_hyperparameter(trial: optuna.trial.Trial, legendre_polys, full_dimension=False):

    hparams_pe = {}
    hparams_pe["legendre_polys"] = legendre_polys
    hparams_pe["full_dimension"] = full_dimension
            
    hparams_nn = {}
    hparams_nn["dim_hidden"] = trial.suggest_categorical("dim_hidden", [32, 64, 96])
    hparams_nn["num_layers"] = trial.suggest_categorical("num_layers", [1, 2, 3])

    hparams_opt = {}
    hparams_opt["lr"] = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
    hparams_opt["wd"] = trial.suggest_float("wd", 1e-8, 1e-1, log=True)
    
    hparams = {}
    hparams.update(hparams_pe)
    hparams.update(hparams_nn)
    hparams.update(hparams_opt)
    
    hparams['harmonics_calculation'] = "shtools"
    
    return hparams

def tune(positional_encoding_name, neural_network_name="siren", dataset="landoceandataset",
         legendre_polys=10, full_dimension=False):
    n_trials = 50
    timeout = 4 * 60 * 60 # seconds

    # Check GPU availability and set accelerator
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name()}")
        accelerator = 'gpu'
        devices = 1
    else:
        print("CUDA not available, using CPU")
        accelerator = 'cpu'
        devices = 'auto'

    datamodule = LandOceanDataModule(mode='tune')
    num_classes = 1

    def objective(trial: optuna.trial.Trial) -> float:

        hparams = get_hyperparameter(trial, legendre_polys=legendre_polys, full_dimension=full_dimension)
        hparams["num_classes"] = num_classes

        spatialencoder = LocationEncoder(
                            positional_encoding_name,
                            neural_network_name,
                            hparams=hparams
            )

        max_epochs = 500
       
        trainer = pl.Trainer(
            max_epochs=max_epochs,
            log_every_n_steps=5,
            accelerator= accelerator, 
            devices=devices,
            callbacks=[EarlyStopping(monitor="val_loss", mode="min", patience=30)])
        
        trainer.logger.log_hyperparams(hparams)

        trainer.fit(model=spatialencoder, datamodule=datamodule)

        # Track actual epochs used by early stopping
        actual_epochs = trainer.current_epoch + 1
        stopped_early = trainer.should_stop
        
        trial.set_user_attr("actual_epochs", actual_epochs)
        trial.set_user_attr("stopped_early", stopped_early)
        trial.set_user_attr("max_epochs_limit", max_epochs)

        return trainer.callback_metrics["val_loss"].item()

    full_dim_suffix = "-fulldim" if full_dimension else ""
    study_name = f"{dataset}-{positional_encoding_name}-{neural_network_name}-L{legendre_polys}{full_dim_suffix}"
    os.makedirs(f"{TUNE_RESULTS_DIR}/{dataset}/runs/", exist_ok=True)
    storage_name = f"sqlite:///{TUNE_RESULTS_DIR}/{dataset}/runs/{study_name}.db"

    db_path = f"{TUNE_RESULTS_DIR}/{dataset}/runs/{study_name}.db"
    if os.path.exists(db_path):
        print(f"Removing old study database: {db_path}")
        os.remove(db_path)

    study = optuna.create_study(study_name=study_name, direction="minimize", 
                                storage=storage_name, load_if_exists=False)
                                
    study.optimize(objective, n_trials=n_trials, timeout=timeout)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    study = optuna.load_study(study_name=study_name, storage=storage_name)
    df = study.trials_dataframe()
    df['positional_encoder'] = positional_encoding_name
    df['neural_network'] = neural_network_name
    df['legendre_polys'] = legendre_polys
    df['full_dimension'] = full_dimension

    runsummary = f"{TUNE_RESULTS_DIR}/{dataset}/runs/{positional_encoding_name}-{neural_network_name}-L{legendre_polys}{full_dim_suffix}.csv"
    os.makedirs(os.path.dirname(runsummary), exist_ok=True)

    df.to_csv(runsummary)

def compile_summaries(dataset):
    tune_results_dir_this_datset = os.path.join(TUNE_RESULTS_DIR, dataset)
    runsdir = os.path.join(TUNE_RESULTS_DIR, f"{dataset}/runs")
    if not os.path.exists(runsdir):
            print(f"Runs directory not found: {runsdir}")
            return
    
    csvs = [csv for csv in os.listdir(runsdir) if csv.endswith("csv") and csv != "summary.csv"]
    print(f"Found CSV files: {csvs}")
    if not csvs:
        print("No CSV files found to compile")
        return
    
    summary = []
    hparams = {}
    for csv in csvs:
        df = pd.read_csv(os.path.join(runsdir, csv))
        best_run = df.sort_values(by="value").iloc[0]
        value = best_run.value
        params = {k.replace("params_", ""): v for k, v in best_run.to_dict().items() if "params" in k}
        
        # Extract legendre_polys and full_dimension from the dataframe
        if 'legendre_polys' in best_run.index:
            params['legendre_polys'] = int(best_run['legendre_polys'])
        if 'full_dimension' in best_run.index:
            params['full_dimension'] = bool(best_run['full_dimension'])
        
        # Extract actual epochs used
        actual_epochs = best_run.get('user_attrs_actual_epochs', None)
        if pd.notna(actual_epochs):
            # Round up to nearest 50 for cleaner values
            optimal_epochs = int(((actual_epochs + 49) // 50) * 50)
            params['max_epochs'] = optimal_epochs
        else:
            params['max_epochs'] = 500  # Fallback

        parts = csv.replace(".csv", "").split("-")
        pe = parts[0]
        nn = parts[1]
        L = parts[2]
        full_dimension = f"-{parts[3]}" if len(parts) > 3 else ""
        # Create unique key with L
        key = f"{pe}-{nn}-{L}{full_dimension}"
        hparams[key] = params

        sum = {
            "pe":pe,
            "nn":nn,
            "L":L,
            "full_dimension": full_dimension,
            "value":value,
            "actual_epochs": actual_epochs,
            "optimal_epochs": params['max_epochs']
        }
        sum.update(params)
        summary.append(sum)

    summary = pd.DataFrame(summary).sort_values("value").set_index(["pe","nn","L", "full_dimension"])
    summary.to_csv(os.path.join(tune_results_dir_this_datset, "summary.csv"))

    print("writing " + os.path.join(tune_results_dir_this_datset, "hparams.yaml"))

    hparams_output = {
        dataset: {
            "dataset": {
                "num_classes": 1,
                "num_samples": 10000,
                "batch_size": 512,
                "addcoastline": False,
                "dropout": False
            }
        }
    }
    
    # Add each encoder-nn--L combination
    for key, params in hparams.items():
        hparams_output[dataset][key] = params
    
    with open(os.path.join(tune_results_dir_this_datset, "hparams.yaml"), 'w') as f:
        yaml.dump(hparams_output, f, default_flow_style=False, sort_keys=False)

    value_matrix = pd.pivot_table(summary.value.reset_index(), index=["pe", "L", "full_dimension"], columns="nn", values=["value"])["value"]
    print("writing " + os.path.join(tune_results_dir_this_datset, "values.csv"))
    value_matrix.to_csv(os.path.join(tune_results_dir_this_datset, "values.csv"))


if __name__ == '__main__':
    dataset = "landoceandataset"
    neural_network = "siren"
    L_values = [10,40]

    positional_encoders = ["slepian", "sphericalharmonics", "slepianhybrid"]

    for L in L_values:
        print(f"Tuning for Legendre Polys: {L}")
        for pe in positional_encoders:
            if pe == "slepian":
                print(f"Tuning {pe} with L={L}, full_dimension=False (Shannon)")
                tune(pe, neural_network, dataset=dataset, legendre_polys=L, full_dimension=False)
                
                print(f"Tuning {pe} with L={L}, full_dimension=True (Full)")
                tune(pe, neural_network, dataset=dataset, legendre_polys=L, full_dimension=True)
            else:
                print(f"Tuning {pe} with L={L}")
                tune(pe, neural_network, dataset=dataset, legendre_polys=L, full_dimension=False)
    
    compile_summaries(dataset)
    print("Tuning complete!")
    print(f"Results saved to: {TUNE_RESULTS_DIR}/{dataset}/")
