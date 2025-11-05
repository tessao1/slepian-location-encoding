import os
import yaml
import argparse
import json

import lightning as pl
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint, Timer

from data import LandOceanDataModule
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
import torch
import numpy as np
import random



def overwrite_hparams_with_args(hparams, args):
    # overwrites some hparams if specified in arguments
    if "legendre_polys" in hparams.keys() and args.legendre_polys is not None:
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
    return hparams


def parse_args():
    parser = argparse.ArgumentParser()

    # Add your arguments here
    parser.add_argument('--dataset', default="landoceandataset", type=str, choices=["checkerboard",
                                                                                    "landoceandataset"
                                                                                    ])
    parser.add_argument('--pe', default=["sphericalharmonics"], type=str, nargs='+', help='positional encoder(s)',
                        choices=["sphericalharmonics", "slepian"])
    parser.add_argument('--nn', default=["siren"], type=str, nargs='+', help='neural network(s)',
                        choices=["linear", "siren", "fcnet"])

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
    parser.add_argument('--harmonics-calculation', default="analytic", type=str,
                        choices=["analytic", "closed-form", "discretized", "shtools"],
                        help='calculation of spherical harmonics: ' +
                             'analytic uses pre-computed equations. This is exact, but works only up to degree 50, ' +
                             'closed-form uses one equation but is computationally slower (especially for high degrees)' +
                             'discretized pre-computes harmonics on a grid and interpolates these later' +
                             'shtools uses the pyshtools library to compute spherical harmonics')
    parser.add_argument('--full-dimension', default=False, type=bool,
                        help='whether to use the full embedding dimension based on area for slepian functions')

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
    if args.use_expnamehps:
        if 'seed' in args.expname:
            appender_in_yaml = args.expname.split('_seed')[0]
        else:
            appender_in_yaml = args.expname
        hparams = hparams[f"{positional_encoding_name}-{neural_network_name}-{appender_in_yaml}"]
    else:
        hparams = hparams[f"{positional_encoding_name}-{neural_network_name}"]
    hparams.update(dataset_hparams)

    hparams = overwrite_hparams_with_args(hparams, args)
    hparams = set_default_if_unset(hparams, "max_radius", 360)

    if args.dataset == "landoceandataset":
        datamodule = LandOceanDataModule(batch_size=hparams["batch_size"])


    if args.resume_ckpt_from_results_dir:
        resume_checkpoint = find_best_checkpoint(parse_resultsdir(args),
                                                 f"{positional_encoding_name}-{neural_network_name}",
                                                 verbose=True)
    else:
        resume_checkpoint = None

    locationencoder = LocationEncoder(
        positional_encoding_name,
        neural_network_name,
        hparams=hparams
    )

    timer = Timer()
    callbacks = [
        EarlyStopping(monitor="val_loss", mode="min", patience=hparams["patience"]),
        timer
    ]
    if args.save_model:
        callbacks += [ModelCheckpoint(
            dirpath=parse_resultsdir(args),
            monitor='val_loss',
            filename=f"{positional_encoding_name}-{neural_network_name}" + '-{val_loss:.2f}',
            save_last=False
        )]

    if args.log_wandb:
        logger = WandbLogger(project="slepian-location-encoding",
                             name=f"{args.dataset}/{positional_encoding_name}-{neural_network_name}")
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

    # if torch.cuda.is_available():
    #     accelerator = 'gpu'


    # print(f"using gpus: {devices}")

    trainer = pl.Trainer(
        max_epochs=hparams["max_epochs"],
        log_every_n_steps=5,
        callbacks=callbacks,
        accelerator=accelerator,
        devices=devices,
        logger=logger,
        precision=32)

    trainer.fit(model=locationencoder,
                datamodule=datamodule,
                ckpt_path=resume_checkpoint
                )

    if "landoceandataset" in dataset or dataset == "checkerboard":
        # Evaluation on test set
        testresults = trainer.test(model=locationencoder, datamodule=datamodule)
        testloss = testresults[0]["test_loss"]
        testaccuracy = testresults[0]["test_accuracy"]
        testiou = testresults[0]["test_IoU"]

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
            # plotting of world map
            title = f"{positional_encoding_name:1.8}-{neural_network_name:1.6} loss {testloss:.3f} acc {testaccuracy * 100:.2f} IoU {testiou * 100:.2f}"

            savepath = f"{parse_resultsdir(args)}/{title}.png".replace(" ", "_").replace("%", "")
            os.makedirs(os.path.dirname(savepath), exist_ok=True)

            plot_predictions(locationencoder, title=title, show=show, savepath=savepath)
            # plot_predictions(locationencoder, title=title, show=show, savepath=savepath.replace('.pdf', '.png'))

 
    return locationencoder


if __name__ == '__main__':
    args = parse_args()

    positional_encoders = args.pe
    neural_networks = args.nn

    for pe in positional_encoders:
        for nn in neural_networks:
            # overwrite lists with single argument
            args.nn = nn
            args.pe = pe
            fit(args)

