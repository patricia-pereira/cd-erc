# -*- coding: utf-8 -*-
r"""
Command Line Interface
=======================
   Commands:
   - train: for Training a new model.
   - interact: Model interactive mode where we can "talk" with a trained model.
   - test: Tests the model ability to rank candidate answers and generate text.
"""
import json
import click
import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning import seed_everything

from model.data_module import DataModule
from model.emotion_transformer import EmotionTransformer
from trainer import TrainerConfig, build_trainer



@click.group()
def cli():
    pass


@cli.command(name="train")
@click.option(
    "--config",
    "-f",
    type=click.Path(exists=True),
    required=True,
    help="Path to the configure YAML file",
)
def train(config: str) -> None:
    yaml_file = yaml.load(open(config).read(), Loader=yaml.FullLoader)
    # Build Trainer
    train_configs = TrainerConfig(yaml_file)
    seed_everything(train_configs.seed)
    trainer = build_trainer(train_configs.namespace())

    # Build Model
    model_config = EmotionTransformer.ModelConfig(yaml_file)
    model = EmotionTransformer(model_config.namespace())
    data = DataModule(model.config, model.tokenizer)
    trainer.fit(model, data)


@cli.command(name="interact")
@click.option(
    "--experiment",
    type=click.Path(exists=True),
    required=True,
    help="Path to the experiment folder containing the checkpoint we want to interact with.",
)
def interact(experiment: str) -> None:
    """Interactive mode command where we can have a conversation with a trained model
    that impersonates a Vegan that likes cooking and radical activities such as sky-diving.
    """
    model = EmotionTransformer.from_experiment(experiment)
    while 1:
        print("Please write a sentence or quit to exit the interactive shell:")
        # Get input sentence
        input_sentence = input("> ")
        if input_sentence == "q" or input_sentence == "quit":
            break
        prediction = model.predict(samples=[input_sentence])
        print(json.dumps(prediction[0], indent=3))


@cli.command(name="test")
@click.option(
    "--experiment",
    type=click.Path(exists=True),
    required=True,
    help="Path to the experiment folder containing the checkpoint we want to interact with.",
)
def test(
    experiment: str,
) -> None:
    """Testing function where a trained model is tested in its ability to rank candidate
    answers and produce replies.
    """
    model = EmotionTransformer.from_experiment(experiment)
    data = DataModule(model.config, model.tokenizer)
    data.prepare_data()

    # Build a very simple trainer
    trainer = pl.Trainer(
        gpus=1 if torch.cuda.is_available() else 0,
        deterministic=True,
        logger=False
    )

    trainer.test(model, test_dataloaders=data.test_dataloader())

if __name__ == "__main__":
    cli()
