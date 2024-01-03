# -*- coding: utf-8 -*-
r""" 
EmotionTransformer Model
==================
    Hugging-face Transformer Model implementing the PyTorch Lightning interface that
    can be used to train an Emotion Classifier.
"""
import multiprocessing
import os
from argparse import Namespace
from typing import Any, Dict, List, Tuple
import click
import pytorch_lightning as pl
import torch
import torch.nn as nn
import yaml

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from torch.utils.data import DataLoader, TensorDataset
from torchnlp.utils import lengths_to_mask
from transformers import AdamW, AutoModel

from model.data_module import DataModule
from model.tokenizer import Tokenizer
from utils import Config

DAILYDIALOG = ["no emotion", "anger", "disgust", "fear", "happiness", "sadness", "surprise"]

EMOWOZ = ["neutral","fearful-sad-disappointed", "dissatisfied-disliking","apologetic","abusive","excited-happy-anticipating","satisfied-liking"]


class EmotionTransformer(pl.LightningModule):
    """Hugging-face Transformer Model implementing the PyTorch Lightning interface that
    can be used to train an Emotion Classifier.

    :param hparams: ArgumentParser containing the hyperparameters.
    """

    class ModelConfig(Config):
        """The ModelConfig class is used to define Model settings.

        ------------------ Architecture --------------------- 
        :param pretrained_model: Pretrained Transformer model to be used.
        
        ----------------- Tranfer Learning --------------------- 
        :param nr_frozen_epochs: number of epochs where the `encoder` model is frozen.
        :param encoder_learning_rate: Learning rate to be used to fine-tune parameters from the `encoder`.
        :param learning_rate: Learning Rate used during training.
        :param layerwise_decay: Learning rate decay for to be applied to the encoder layers.

        ----------------------- Data --------------------- 
        :param dataset_path: Path to a json file containing our data.
        :param labels: Label set (options: `ekman`, `goemotions`)
        :param batch_size: Batch Size used during training.
        """

        pretrained_model: str = "roberta-base"

        # Optimizer
        nr_frozen_epochs: int = 1
        encoder_learning_rate: float = 1.0e-5
        learning_rate: float = 5.0e-5
        layerwise_decay: float = 0.95

        # Data configs
        dataset_path: str = ""
        dataset: str = "dailydialog"
        labels: str = "dailydialog"

        # Training details
        batch_size: int = 4

        context: bool = True
        context_turns: int = 3

    def __init__(self, hparams: Namespace):
        super().__init__()
        self.config = hparams
        self.save_hyperparameters(self.config)
        print(self.config)
        self.transformer = AutoModel.from_pretrained(self.config.pretrained_model)
        self.tokenizer = Tokenizer(self.config.pretrained_model, self.config.context)
       
        self.encoder_features = self.transformer.config.hidden_size

        # Resize embeddings to include the added tokens
        self.transformer.resize_token_embeddings(self.tokenizer.vocab_size)

        self.num_layers = self.transformer.config.num_hidden_layers + 1

        if self.config.labels == "dailydialog":
            self.label_encoder = {DAILYDIALOG[i]: i for i in range(len(DAILYDIALOG))}
        elif self.config.labels == "emowoz":
            self.label_encoder = {EMOWOZ[i]: i for i in range(len(EMOWOZ))}
        else:
            raise Exception("unrecognized label set: {}".format(self.config.labels))

        # Classification head
        self.classification_head = nn.Linear(
            self.encoder_features, len(self.label_encoder)
        )

        self.softmax=nn.Softmax(dim=1)

        self.loss = nn.BCEWithLogitsLoss()

        if self.config.nr_frozen_epochs > 0:
            self.freeze_encoder()
        else:
            self._frozen = False
        self.nr_frozen_epochs = self.config.nr_frozen_epochs

    def unfreeze_encoder(self) -> None:
        """ un-freezes the encoder layer. """
        if self._frozen:
            click.secho("-- Encoder model fine-tuning", fg="yellow")
            for param in self.transformer.parameters():
                param.requires_grad = True
            self._frozen = False

    def freeze_encoder(self) -> None:
        """ freezes the encoder layer. """
        for param in self.transformer.parameters():
            param.requires_grad = False
        self._frozen = True

    def layerwise_lr(self, lr: float, decay: float) -> list:
        """ Separates layer parameters and sets the corresponding learning rate to each layer.

        :param lr: Initial Learning rate.
        :param decay: Decay value.

        :return: List with grouped model parameters with layer-wise decaying learning rate
        """
        opt_parameters = [
            {
                "params": self.transformer.embeddings.parameters(),
                "lr": lr * decay ** (self.num_layers),
            }
        ]
        opt_parameters += [
            {
                "params": self.transformer.encoder.layer[l].parameters(),
                "lr": lr * decay ** (self.num_layers - 1 - l),
            }
            for l in range(self.num_layers - 1)
        ]
        return opt_parameters
    
    # Pytorch Lightning Method
    def configure_optimizers(self):
        layer_parameters = self.layerwise_lr(
            self.config.encoder_learning_rate, self.config.layerwise_decay
        )
        head_parameters = [
            {
                "params": self.classification_head.parameters(),
                "lr": self.config.learning_rate,
            }
        ]

        optimizer = AdamW(
            layer_parameters + head_parameters,
            lr=self.config.learning_rate,
            correct_bias=True,
        )
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return [optimizer], [scheduler]

    def forward(
        self,
        input_ids: torch.Tensor,
        input_lengths: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        # Reduce unnecessary padding.
        input_ids = input_ids[:, : input_lengths.max()]
        mask = lengths_to_mask(input_lengths, device=input_ids.device)

        

        # Run model.
        output = self.transformer(
            input_ids=input_ids,
            attention_mask=mask,
            output_hidden_states=True,
        )

        if len(output) == 3:
            last_hidden_state = output['last_hidden_state']
            pooler_output = output['pooler_output']
            word_embeddings = output['hidden_states']
        elif len(output) == 2:
            last_hidden_state = output['last_hidden_state']
            word_embeddings = output['hidden_states']
        else:
            raise Exception(f"Can't unpack values {self.haprams.pretrained_model}")

        # Pooling Layer
        sentemb = last_hidden_state[:, 0, :] 

        return self.classification_head(sentemb)

    # Pytorch Lightning Method
    def training_step(
        self, batch: Tuple[torch.Tensor], batch_nb: int, *args, **kwargs
    ) -> Dict[str, torch.Tensor]:
        input_ids, input_lengths, labels= batch
        logits = self.forward(input_ids, input_lengths)
        
        loss_value = self.loss(logits, labels) 
        
        if (
            self.nr_frozen_epochs < 1.0
            and self.nr_frozen_epochs > 0.0
            and batch_nb > self.epoch_total_steps * self.nr_frozen_epochs
        ):
            self.unfreeze_encoder()
            self._frozen = False

        # can also return just a scalar instead of a dict (return loss_val)
        return {"loss": loss_value, "log": {"train_loss": loss_value}}

    # Pytorch Lightning Method
    def validation_step(
        self, batch: Tuple[torch.Tensor], batch_nb: int, *args, **kwargs
    ) -> Dict[str, torch.Tensor]:
        input_ids, input_lengths, labels= batch
        logits = self.forward(input_ids, input_lengths)
       
        loss_value = self.loss(logits, labels)
        predictions = torch.argmax(logits, dim=1)

        return {"val_loss":  loss_value, "predictions": predictions, "labels": labels}

    # Pytorch Lightning Method
    def validation_epoch_end(
        self, outputs: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
   
        predictions = torch.cat([o["predictions"] for o in outputs], dim=0)
        labels = torch.cat([o["labels"] for o in outputs], dim=0)
    
        y_hat_predictions = predictions.cpu().numpy()
        y_labels = torch.argmax(labels, dim=1).cpu().numpy()
       
        # We will log the macro and micro-averaged metrics:
        metrics = {
            "macro-precision": torch.tensor(
                precision_score(y_labels, y_hat_predictions, average='macro', zero_division=0)),
            "macro-recall": torch.tensor(
                recall_score(y_labels, y_hat_predictions, average='macro', zero_division=0)),
            "macro-f1": torch.tensor(f1_score(y_labels, y_hat_predictions, average='macro', zero_division=0)),
            "micro-precision": torch.tensor(
                precision_score(y_labels, y_hat_predictions, average='micro', zero_division=0)),
            "micro-recall": torch.tensor(
                recall_score(y_labels, y_hat_predictions, average='micro', zero_division=0)),
            "micro-f1": torch.tensor(f1_score(y_labels, y_hat_predictions, labels=[1,2,3,4,5,6], average='micro', zero_division=0)),
        }
        self.log("macro-precision", metrics["macro-precision"].to(self.transformer.device), prog_bar=True)
        self.log("macro-recall", metrics["macro-recall"].to(self.transformer.device), prog_bar=True)
        self.log("macro-f1", metrics["macro-f1"].to(self.transformer.device), prog_bar=True)
        self.log("micro-f1", metrics["micro-f1"].to(self.transformer.device), prog_bar=True)

        return {
            "progress_bar": metrics,
            "log": metrics,
        }

    # Pytorch Lightning Method
    def test_step(
        self, batch: Tuple[torch.Tensor], batch_nb: int, *args, **kwargs
    ) -> Dict[str, torch.Tensor]:
        """ Same as validation_step. """
        return self.validation_step(batch, batch_nb)

    # Pytorch Lightning Method
    def test_epoch_end(
        self, outputs: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, float]:
        """ Similar to the validation_step_end but computes precision, recall, f1 for each label."""
       
        predictions = torch.cat([o["predictions"] for o in outputs], dim=0)
        labels = torch.cat([o["labels"] for o in outputs], dim=0)
   
        y_hat_predictions = predictions.cpu().numpy()
        y_labels = torch.argmax(labels, dim=1).cpu().numpy()
        # We will log the macro and micro-averaged metrics:
    

        metrics = {
            "macro-precision": torch.tensor(
                precision_score(y_labels, y_hat_predictions, average='macro', zero_division=0)),
            "macro-recall": torch.tensor(
                recall_score(y_labels, y_hat_predictions, average='macro', zero_division=0)),
            "macro-f1": torch.tensor(f1_score(y_labels, y_hat_predictions, average='macro', zero_division=0)),
            "micro-precision": torch.tensor(
                precision_score(y_labels, y_hat_predictions, average='micro', zero_division=0)),
            "micro-recall": torch.tensor(
                recall_score(y_labels, y_hat_predictions, average='micro', zero_division=0)),
            "micro-f1": torch.tensor(f1_score(y_labels, y_hat_predictions, labels=[1,2,3,4,5,6], average='micro', zero_division=0)),
            "weighted-f1": torch.tensor(f1_score(y_labels, y_hat_predictions, labels=[1,2,3,4,5,6], average='weighted', zero_division=0)),
            "accuracy": torch.tensor(accuracy_score(y_labels, y_hat_predictions)),
        }
        
        # metrics per class
        for label, i in self.label_encoder.items():
            metrics[label+"-precision"] = precision_score(y_labels, y_hat_predictions, average=None, zero_division=0)[i]
            metrics[label+"-recall"] = recall_score(y_labels, y_hat_predictions, average=None, zero_division=0)[i]
            metrics[label+"-f1"] = f1_score(y_labels, y_hat_predictions, average=None, zero_division=0)[i]

        self.log('metrics', metrics)
     
        return {
            "progress_bar": metrics,
            "log": metrics,
        }

    # Pytorch Lightning Method
    def on_epoch_end(self):
        """ Pytorch lightning hook """
        if self.current_epoch + 1 >= self.nr_frozen_epochs:
            self.unfreeze_encoder()

    @classmethod
    def from_experiment(cls, experiment_folder: str):
        """Function that loads the model from an experiment folder.

        :param experiment_folder: Path to the experiment folder.

        :return: Pretrained model.
        """
        hparams_file = experiment_folder + "hparams.yaml"
        hparams = yaml.load(open(hparams_file).read(), Loader=yaml.FullLoader)

        checkpoints = [
            file for file in os.listdir(experiment_folder +"checkpoints/") if file.endswith(".ckpt")
        ]
        print(checkpoints)
        checkpoint_path = experiment_folder +"checkpoints/"+ checkpoints[-1]
        model = cls.load_from_checkpoint(
            checkpoint_path, hparams=Namespace(**hparams), strict=True
        )
        # Make sure model is in prediction mode
        model.eval()
        model.freeze()
        return model

    def predict(self, samples: List[str]) -> Dict[str, Any]:
        """ Predict function.

        :param samples: list with the texts we want to classify.

        :return: List with classified texts.
        """
        if self.training:
            self.eval()
       
        output = [{"text": sample} for sample in samples]
        # Create inputs
        input_ids = [self.tokenizer.encode(s) for s in samples]
        input_lengths = [len(ids) for ids in input_ids]
        samples = {"input_ids": input_ids, "input_lengths": input_lengths}
        # Pad inputs
        samples = DataModule.pad_dataset(samples)
        dataloader = DataLoader(
            TensorDataset(
                torch.tensor(samples["input_ids"]),
                torch.tensor(samples["input_lengths"]),
            ),
            batch_size=self.config.batch_size,
            num_workers=multiprocessing.cpu_count(),
        )

        i = 0
        with torch.no_grad():
            for input_ids, input_lengths in dataloader:
                logits = self.forward(input_ids, input_lengths)
                
                # Turn logits into probabilities
                probs = torch.sigmoid(logits)
                for j in range(probs.shape[0]):
                    label_probs = {}
                    for label, k in self.label_encoder.items():
                        label_probs[label] = probs[j][k].item()
                    output[i]["emotions"] = label_probs
                    i += 1
        print(output)
        return output

   