# -*- coding: utf-8 -*-
r"""
Lightning Trainer Setup
=======================
   Setup logic for the lightning trainer.
"""
import os
from argparse import Namespace
from datetime import datetime

import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import TensorBoardLogger

from utils import Config


class TrainerConfig(Config):
    """
    The TrainerConfig class is used to define default hyper-parameters that
    are used to initialize our Lightning Trainer. These parameters are then overwritted
    with the values defined in the YAML file.

    -------------------- General Parameters -------------------------

    :param seed: Training seed.

    :param deterministic: If true enables cudnn.deterministic. Might make your system
        slower, but ensures reproducibility.

    :param verbose: verbosity mode.

    :param overfit_batches: Uses this much data of the training set. If nonzero, will use
        the same training set for validation and testing. If the training dataloaders
        have shuffle=True, Lightning will automatically disable it.

    -------------------- Model Checkpoint & Early Stopping -------------------------

    :param early_stopping: If true enables EarlyStopping.

    :param save_top_k: If save_top_k == k, the best k models according to the metric
        monitored will be saved.

    :param monitor: Metric to be monitored.

    :param save_weights_only: Saves only the weights of the model.

    :param metric_mode: One of {min, max}. In min mode, training will stop when the
        metric monitored has stopped decreasing; in max mode it will stop when the
        metric monitored has stopped increasing.

    :param min_delta: Minimum change in the monitored metric to qualify as an improvement.

    :param patience: Number of epochs with no improvement after which training will be stopped.

    :param accumulate_grad_batches: Gradient accumulation steps.
    """

    seed: int = 3
    deterministic: bool = True
    verbose: bool = False
    overfit_batches: float = 0.0

    # Model Checkpoint & Early Stopping
    early_stopping: bool = True
    save_top_k: int = 1
    monitor: str = "macro-f1"
    save_weights_only: bool = False
    metric_mode: str = "max"
    min_delta: float = 0.0
    patience: int = 1
    accumulate_grad_batches: int = 1

    dataset: str = "ekman"
    experiment_name: str = "baseline"


    def __init__(self, initial_data: dict) -> None:
        trainer_attr = pl.Trainer.default_attributes()
        for key in trainer_attr:
            setattr(self, key, trainer_attr[key])

        for key in initial_data:
            if hasattr(self, key):
                setattr(self, key, initial_data[key])


def build_trainer(hparams: Namespace) -> pl.Trainer:
    """
    :param hparams: Namespace

    Returns:
        - pytorch_lightning Trainer
    """
    # Early Stopping Callback
    early_stop_callback = EarlyStopping(
        monitor=hparams.monitor,
        min_delta=hparams.min_delta,
        patience=hparams.patience,
        verbose=hparams.verbose,
        mode=hparams.metric_mode,
    )

    # TestTube Logger Callback
    tb_logger = TensorBoardLogger(
        save_dir="experiments/",
        version=hparams.dataset  + "_" +str(hparams.experiment_name) + "_" + datetime.now().strftime("%d-%m-%Y--%H-%M-%S"),
        name="",
    )

    # Model Checkpoint Callback
    ckpt_path = os.path.join(
        "experiments/",
        tb_logger.version,
        "checkpoints",
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_path,
        save_top_k=hparams.save_top_k,
        verbose=hparams.verbose,
        monitor=hparams.monitor,
        save_weights_only=hparams.save_weights_only,
        #period=0,  # Always allow saving checkpoint even within the same epoch
        mode=hparams.metric_mode,
    )

    trainer = pl.Trainer(
        logger=tb_logger,
        callbacks=[LearningRateMonitor(), early_stop_callback, checkpoint_callback],
        gradient_clip_val=hparams.gradient_clip_val,
        gpus=hparams.gpus,
        log_gpu_memory="all",
        deterministic=hparams.deterministic,
        overfit_batches=hparams.overfit_batches,
        #accumulate_grad_batches=hparams.accumulate_grad_batches,
        max_epochs=hparams.max_epochs,
        min_epochs=hparams.min_epochs,
        limit_train_batches=hparams.limit_train_batches,
        limit_val_batches=hparams.limit_val_batches,
        val_check_interval=hparams.val_check_interval,
        log_every_n_steps=hparams.log_every_n_steps,
        #distributed_backend="dp",
        precision=hparams.precision,
        weights_summary="top",
        profiler=hparams.profiler,
        num_sanity_val_steps=5,
    )
    return trainer
