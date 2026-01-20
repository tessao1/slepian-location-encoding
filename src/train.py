import os
import yaml
import argparse
import json

import lightning as pl
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint, Timer

from data import LandOceanDataModule, HighResLandOceanDataModule
from utils import (
    plot_predictions,
    plot_predictions_at_points,
    plot_longitudinal_accuracy,
    parse_resultsdir,
    count_parameters,
    find_best_checkpoint,
    set_default_if_unset,
)

from locationencoder import LocationImageEncoder, LocationEncoder

from lightning.pytorch.loggers import WandbLogger
import wandb
import torch
import numpy as np
import random

torch.set_float32_matmul_precision('high')

def overwrite_hparams_with_args(hparams, args):
    # overwrites some hparams if specified in arguments
    if  args.legendre_polys is not None:
        hparams["legendre_polys"] = args.legendre_polys
        print(f"using legendre-polys={args.legendre_polys}, as specified in args")
    if "min_radius" in hparams.keys() and args.min_radius is not None:
        hparams["min_radius"] = args.min_radius
        print(f"using min-radius={args.min_radius}, as specified in args")
    if args.harmonics_calculation is not None:
        hparams["harmonics_calculation"] = args.harmonics_calculation
        print(f"using harmonics_calculation={args.harmonics_calculation}, as specified in args")
    if args.max_epochs is not None:
        hparams["max_epochs"] = args.max_epochs
        print(f"using max_epochs={args.max_epochs}, as specified in args")
    hparams["full_dimension"] = args.full_dimension
    if args.full_dimension:
        print(f"using full_dimension={args.full_dimension}, as specified in args")
    if args.num_samples is not None:
        hparams["num_samples"] = args.num_samples
        print(f"using num_samples={args.num_samples}, as specified in args")
    if args.sh_max_degree is not None:
        hparams["sh_max_degree"] = args.sh_max_degree
        print(f"using sh_max_degree={args.sh_max_degree}, as specified in args")
    return hparams


def parse_args():
    parser = argparse.ArgumentParser()

    # Add your arguments here
    parser.add_argument('--dataset', default="landoceandataset", type=str, choices=["checkerboard",
                                                                                    "landoceandataset",
                                                                                    "highreslandoceandataset"
                                                                                    ])
    parser.add_argument('--pe', default=["sphericalharmonics"], type=str, nargs='+', help='positional encoder(s)',
                        choices=["sphericalharmonics", "slepian", "slepianhybrid", "direct"])
    parser.add_argument('--nn', default=["siren"], type=str, nargs='+', help='neural network(s)',
                        choices=["linear", "siren", "fcnet", "mlp"])

    # optional configs
    parser.add_argument('--save-model', action="store_true", help='save model checkpoint to results-dir')
    parser.add_argument('--log-wandb', action="store_true", help='log run to wandb')
    parser.add_argument('--hparams', default="hparams.yaml", type=str, help='hypereparameter yaml')
    parser.add_argument('--results-dir', default="results/train", type=str, help='results directory')
    parser.add_argument('--expname', default=None, type=str,
                        help='experiment name. If specified, saves results in subfolder')
    parser.add_argument('--seed', default=0, type=int, help='global random seed')
    parser.add_argument('--max-epochs', default=None, type=int,
                        help='maximum number of epochs. If unset, uses value in hparams.yaml')
    parser.add_argument('--gpus', default='-1', type=int, nargs='+',
                        help='which gpus to use; if unset uses -1 which we map to auto')
    parser.add_argument('--accelerator', default='auto', type=str,
                        help='lightning accelerator')

    parser.add_argument('-r', '--resume-ckpt-from-results-dir', action="store_true",
                        help="searches through provided results dir and resumes from suitable checkpoint "
                             "that matches pe and nn")
    parser.add_argument('--matplotlib', action="store_true",
                        help="plot maps with matplotlib")
    parser.add_argument('--matplotlib-show', action="store_true",
                        help="shows matplotlib plots (can cause freezing when called remotely)")

    parser.add_argument('--use-expnamehps', default=False, type=bool,
                        help='whether expname is part of the hp file names')

    # checkerboard
    parser.add_argument('--checkerboard-scale', default=1, type=float,
                        help="scales the number of support points for the checkerboard dataset (specificed in hparams.yaml) "
                             "by this factor. This is useful to vary the scale to test different resolutions of encoders")

    # overwrite certain hparams
    parser.add_argument('--legendre-polys', default=None, type=int)
    parser.add_argument('--min-radius', default=None, type=float)
    parser.add_argument('--harmonics-calculation', default="shtools", type=str,
                        choices=["analytic", "closed-form", "discretized", "shtools"],
                        help='calculation of spherical harmonics: ' +
                             'analytic uses pre-computed equations. This is exact, but works only up to degree 50, ' +
                             'closed-form uses one equation but is computationally slower (especially for high degrees)' +
                             'discretized pre-computes harmonics on a grid and interpolates these later' +
                             'shtools uses the pyshtools library to compute spherical harmonics')
    parser.add_argument('--sh-max-degree', default=5, type=int,
                        help='maximum degree for spherical harmonics in hybrid encoding')
    parser.add_argument('--full-dimension', default=False, type=bool,
                        help='whether to use the full embedding dimension based on area for slepian functions')
    parser.add_argument('--num-samples', default=None, type=int,
                        help='number of samples to use for the datasets')
    parser.add_argument('--sampling-method', default='fibonacci', type=str,
                        choices=['fibonacci', 'uniform', 'sphericaluniform'],
                        help='sampling method for generating datasets')
    #High-res landocean dataset visualization
    parser.add_argument('--visualization-regions', default=['Caribbean', 'Indonesia'], 
                        type=str, nargs='+',
                        help='regions to visualize for high-res dataset (used with --matplotlib)')
    parser.add_argument('--visualization-resolution', default=0.03, type=float,
                        help='resolution for high-res visualization grid')

    args = parser.parse_args()
    return args

def fit(args):
    positional_encoding_name = args.pe
    neural_network_name = args.nn
    dataset = args.dataset

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    with open(args.hparams) as f:
        hparams = yaml.safe_load(f)

    dataset_hparams = hparams[dataset]["dataset"]

    hparams = hparams[dataset]
    print(args)

    key = f"{positional_encoding_name}-{neural_network_name}"
        
    if key in hparams:
        print(f"Using hyperparameters for: {key}")
        hparams = hparams[key]
    else:
        available_keys = [k for k in hparams.keys() if k != "dataset"]
        raise KeyError(
            f"Key '{key}' not found in hparams.yaml.\n"
            f"Available keys: {available_keys}"
        )
    
    hparams.update(dataset_hparams)

    hparams = overwrite_hparams_with_args(hparams, args)
    hparams = set_default_if_unset(hparams, "max_radius", 360)

    if args.dataset == "highreslandoceandataset":
        hparams["use_highres_metrics"] = True
        hparams["num_classes"] = 1  # Binary classification
        datamodule = HighResLandOceanDataModule(
            num_samples=hparams["num_samples"], 
            batch_size=hparams["batch_size"], 
            mode='train', 
            sampling_method=args.sampling_method)
    elif args.dataset == "landoceandataset":
        datamodule = LandOceanDataModule(num_samples=hparams["num_samples"], batch_size=hparams["batch_size"], mode='train')
    elif args.dataset == "highreslandoceandataset":
        datamodule = HighResLandOceanDataModule(num_samples=hparams["num_samples"], batch_size=hparams["batch_size"], mode='train', sampling_method=args.sampling_method)


    if args.resume_ckpt_from_results_dir:
        resume_checkpoint = find_best_checkpoint(parse_resultsdir(args),
                                                 f"{positional_encoding_name}-{neural_network_name}-{args.legendre_polys}",
                                                 verbose=True)
        locationencoder = LocationEncoder.load_from_checkpoint(
            resume_checkpoint,
            positional_encoding_name=positional_encoding_name,
            neural_network_name=neural_network_name,
            hparams=hparams
        )
        print(f"Loaded best model from checkpoint: {resume_checkpoint}")
    else:
        resume_checkpoint = None
        locationencoder = LocationEncoder(
            positional_encoding_name,
            neural_network_name,
            hparams=hparams
        )

    timer = Timer()
    callbacks = [timer]
    
    if args.save_model:
        callbacks += [ModelCheckpoint(
            dirpath=parse_resultsdir(args),
            monitor='val_loss',
            filename=f"{positional_encoding_name}-{neural_network_name}-{args.legendre_polys}" + '-{val_loss:.2f}',
            save_last=False
        )]

    if args.log_wandb:
        logger = WandbLogger(project="slepian-location-encoding",
                             name=f"{args.dataset}/{positional_encoding_name}-{neural_network_name}-{args.legendre_polys}")
    else:
        logger = None

    # use GPU if it is available
    accelerator = args.accelerator
    devices = 1
    if args.gpus == -1 or args.gpus == [-1]:
        devices = 'auto'
    else:
        devices = args.gpus

    if accelerator == 'auto':
        if torch.cuda.is_available():
            accelerator = 'gpu'
        else:
            accelerator = 'cpu'

    print(f"Using accelerator: {accelerator}, devices: {devices}")

    trainer = pl.Trainer(
        max_epochs=hparams["max_epochs"],
        log_every_n_steps=5,
        callbacks=callbacks,
        accelerator=accelerator,
        devices=devices,
        logger=logger,
        precision=32)

    if args.resume_ckpt_from_results_dir:
        print("Skipping training - using loaded checkpoint for evaluation only")
    else:
        trainer.fit(model=locationencoder,
                    datamodule=datamodule,
                    ckpt_path=resume_checkpoint
                    )
        
    if args.save_model:
        best_checkpoint = find_best_checkpoint(parse_resultsdir(args),
                                            f"{positional_encoding_name}-{neural_network_name}-{args.legendre_polys}",
                                            verbose=True)
        if best_checkpoint is not None:
            print(f"\nLoading best checkpoint for evaluation: {best_checkpoint}")
            locationencoder = LocationEncoder.load_from_checkpoint(
                best_checkpoint,
                positional_encoding_name=positional_encoding_name,
                neural_network_name=neural_network_name,
                hparams=hparams
            )
        else:
            print("Warning: No checkpoint found, using model from last epoch")
    # Evaluation and visualization
    if dataset == "highreslandoceandataset":
        # High-res dataset evaluation
        testresults = trainer.test(model=locationencoder, datamodule=datamodule)
        
        # PyTorch Lightning returns a list with one dict per test dataloader
        test_uniform = testresults[0] if len(testresults) > 0 else {}
        test_coastline = testresults[1] if len(testresults) > 1 else {}
        test_island = testresults[2] if len(testresults) > 2 else {}

        title = f"{positional_encoding_name:1.8}-{neural_network_name:1.6}-{args.legendre_polys}"
        resultsfile = f"{parse_resultsdir(args)}/{title}.json".replace(" ", "_").replace("%", "")
        os.makedirs(os.path.dirname(resultsfile), exist_ok=True)

        print(f"\nwriting {resultsfile}")
        
        # Extract metrics (they should be in the dicts)
        result = dict(
            # Uniform test set
            test_uniform_loss=test_uniform.get("test_loss_uniform", None),
            test_uniform_accuracy=test_uniform.get("test_accuracy_uniform", None),
            test_uniform_f1=test_uniform.get("test_f1_uniform", None),
            test_uniform_auc=test_uniform.get("test_auc_uniform", None),
            
            # Coastline test set
            test_coastline_loss=test_coastline.get("test_loss_coastline", None),
            test_coastline_accuracy=test_coastline.get("test_accuracy_coastline", None),
            test_coastline_f1=test_coastline.get("test_f1_coastline", None),
            test_coastline_auc=test_coastline.get("test_auc_coastline", None),
            
            # Island test set
            test_island_loss=test_island.get("test_loss_island", None),
            test_island_accuracy=test_island.get("test_accuracy_island", None),
            test_island_f1=test_island.get("test_f1_island", None),
            test_island_auc=test_island.get("test_auc_island", None),
            
            # Model info
            positional_encoder=positional_encoding_name,
            neural_network=neural_network_name,
            legendre_polys=hparams.get('legendre_polys', None),
            sampling_method=args.sampling_method,
            num_samples=hparams.get('num_samples', None),
            num_params=count_parameters(locationencoder),
            test_duration=timer.time_elapsed("test"),
            train_duration=timer.time_elapsed("train"),
            test_samples=len(datamodule.test_uniform_ds),
            train_samples=len(datamodule.train_ds),
            embedding_dim=locationencoder.positional_encoder.embedding_dim,
        )

        # Add cache statistics if available
        if hasattr(locationencoder, 'positional_encoder') and hasattr(locationencoder.positional_encoder, 'get_cache_stats'):
            cache_stats = locationencoder.positional_encoder.get_cache_stats()
            result.update({
                'cache_hits': cache_stats['cache_hits'],
                'cache_misses': cache_stats['cache_misses'],
                'cache_hit_rate': cache_stats['hit_rate'],
                'cache_size': cache_stats['cache_size']
            })

        result.update(hparams)
        
        # Save individual JSON
        with open(resultsfile, "w") as json_file:
            json.dump(result, json_file)

        # Append to consolidated CSV
        csv_file = f"{parse_resultsdir(args)}/final_results.csv"
        import pandas as pd
        
        # Create DataFrame from result
        df_row = pd.DataFrame([result])
        
        # Append to CSV (create if doesn't exist)
        if os.path.exists(csv_file):
            df_existing = pd.read_csv(csv_file)
            df_combined = pd.concat([df_existing, df_row], ignore_index=True)
            df_combined.to_csv(csv_file, index=False)
            print(f"Appended results to {csv_file}")
        else:
            df_row.to_csv(csv_file, index=False)
            print(f"Created new results file: {csv_file}")

        if logger is not None:
            logger.log_metrics({
                "final/num_params": result["num_params"],
                "final/train_duration": result["train_duration"],
                "final/test_duration": result["test_duration"],
                "final/embedding_dim": result["embedding_dim"],
                "final/test_uniform_accuracy": result["test_uniform_accuracy"],
                "final/test_coastline_accuracy": result["test_coastline_accuracy"],
                "final/test_island_accuracy": result["test_island_accuracy"],
                "final/test_uniform_f1": result["test_uniform_f1"],
                "final/test_coastline_f1": result["test_coastline_f1"],
                "final/test_island_f1": result["test_island_f1"],
                "final/test_uniform_auc": result["test_uniform_auc"],
                "final/test_coastline_auc": result["test_coastline_auc"],
                "final/test_island_auc": result["test_island_auc"],
            })

        # Print results table
        print(f"\n{title} Results:")
        print(f"{'Test Set':<12} {'Accuracy':>10} {'F1':>10} {'AUC':>10} {'Loss':>10}")
        print("-"*54)
        if result['test_uniform_accuracy'] is not None:
            print(f"{'Uniform':<12} {result['test_uniform_accuracy']:>10.4f} {result['test_uniform_f1']:>10.4f} {result['test_uniform_auc']:>10.4f} {result['test_uniform_loss']:>10.4f}")
        if result['test_coastline_accuracy'] is not None:
            print(f"{'Coastline':<12} {result['test_coastline_accuracy']:>10.4f} {result['test_coastline_f1']:>10.4f} {result['test_coastline_auc']:>10.4f} {result['test_coastline_loss']:>10.4f}")
        if result['test_island_accuracy'] is not None:
            print(f"{'Island':<12} {result['test_island_accuracy']:>10.4f} {result['test_island_f1']:>10.4f} {result['test_island_auc']:>10.4f} {result['test_island_loss']:>10.4f}")

        # High-res visualization (only if --matplotlib is used)
        if args.matplotlib or args.matplotlib_show:
            import matplotlib.pyplot as plt
            from visualization import plot_region_comparison, load_land_geometry
            
            show = args.matplotlib_show
            land_union = load_land_geometry()
            
            base_path = f"{parse_resultsdir(args)}/{title}".replace(" ", "_").replace("%", "")
            os.makedirs(os.path.dirname(base_path), exist_ok=True)
            
            # Get device from model
            device = next(locationencoder.parameters()).device
            
            # Plot each region
            for region_name in args.visualization_regions:
                savepath = f"{base_path}_{region_name.lower()}.png"
                print(f"\nGenerating visualization for {region_name}...")
                
                fig, metrics = plot_region_comparison(
                    locationencoder,
                    region_name=region_name,
                    land_union=land_union,
                    resolution=args.visualization_resolution,
                    device=device,
                    save_path=savepath,
                    show=show
                )
                
                # Close the figure to prevent blocking and free memory
                plt.close(fig)
                
                if logger is not None:
                    logger.experiment.log({
                        f"plots/{region_name.lower()}": wandb.Image(savepath, caption=f"{title} - {region_name}")
                    })
                
                print(f"  {region_name} - Accuracy: {metrics['accuracy']:.4f}, Errors: {metrics['errors']}/{metrics['total']}")

    elif "landoceandataset" in dataset or dataset == "checkerboard":
        # Evaluation on test set
        testresults = trainer.test(model=locationencoder, datamodule=datamodule)
        testloss = testresults[0]["test_loss"]
        testaccuracy = testresults[0]["test_accuracy"]
        testiou = testresults[0]["test_IoU"]
        testlocalaccuracy = testresults[0].get("test_local_accuracy", None)

        title = f"{positional_encoding_name:1.8}-{neural_network_name:1.6}"
        resultsfile = f"{parse_resultsdir(args)}/{title}.json".replace(" ", "_").replace("%", "")
        os.makedirs(os.path.dirname(resultsfile), exist_ok=True)

        print(f"writing {resultsfile}")
        result = dict(
            iou=testiou,
            accuracy=testaccuracy,
            testloss=testloss,
            num_params=count_parameters(locationencoder),
            mean_dist=datamodule.mean_dist if hasattr(datamodule, "mean_dist") else None,
            test_duration=timer.time_elapsed("test"),
            train_duration=timer.time_elapsed("train"),
            test_samples=len(datamodule.test_dataloader().dataset),
            train_samples=len(datamodule.train_dataloader().dataset),
            embedding_dim=locationencoder.positional_encoder.embedding_dim
        )
        if testlocalaccuracy is not None:
            result["local_accuracy"] = testlocalaccuracy

        if logger is not None:
            logger.log_metrics({
                "final/num_params": result["num_params"],
                "final/train_duration": result["train_duration"],
                "final/test_duration": result["test_duration"],
                "final/embedding_dim": result["embedding_dim"]
            })
                # Add cache statistics to results if available
        if hasattr(locationencoder, 'positional_encoder') and hasattr(locationencoder.positional_encoder, 'get_cache_stats'):
            cache_stats = locationencoder.positional_encoder.get_cache_stats()
            result.update({
                'cache_hits': cache_stats['cache_hits'],
                'cache_misses': cache_stats['cache_misses'],
                'cache_hit_rate': cache_stats['hit_rate'],
                'cache_size': cache_stats['cache_size']
            })

        result.update(hparams)
        with open(resultsfile, "w") as json_file:
            json.dump(result, json_file)

        
        if args.matplotlib or args.matplotlib_show:
            show = args.matplotlib_show
            title = f"{positional_encoding_name:1.8}-{neural_network_name:1.6} loss {testloss:.3f} acc {testaccuracy * 100:.2f} IoU {testiou * 100:.2f}"

            base_path = f"{parse_resultsdir(args)}/{title}".replace(" ", "_").replace("%", "")
            os.makedirs(os.path.dirname(base_path), exist_ok=True)
            
            savepath_map = base_path + ".png"
            savepath_globe = base_path + "_globe.png"
            
            plot_predictions(locationencoder, title=title, show=show, savepath=savepath_map, save_globe=True)

            if logger is not None:
                logger.experiment.log({
                    "plots/prediction_map": wandb.Image(savepath_map, caption=f"{title} - Map View"),
                    "plots/prediction_globe": wandb.Image(savepath_globe, caption=f"{title} - Globe View")
                })
 
    return locationencoder


if __name__ == '__main__':
    args = parse_args()

    positional_encoders = args.pe
    neural_networks = args.nn

    for pe in positional_encoders:
        for nn in neural_networks:
            args.nn = nn
            args.pe = pe
            fit(args)

