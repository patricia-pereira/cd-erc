# -*- coding: utf-8 -*-
r""" 
DataModule
==========
    The DataModule encapsulates all the steps needed to process data:
    - Download / tokenize
    - Save to disk.
    - Apply transforms (tokenize, pad, batch creation, etc…).
    - Load inside Dataset.
    - Wrap inside a DataLoader.
"""
import hashlib
import multiprocessing
import os
from argparse import Namespace
from collections import defaultdict
from typing import Dict, List


import click
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm



from model.tokenizer import Tokenizer
from model.utils import load_dailydialog, load_emowoz

PADDED_INPUTS = ["input_ids"]
MODEL_INPUTS = ["input_ids", "input_lengths", "labels"]


class DataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule.

    :param hparams: Namespace with data specific arguments.
    :param tokenizer: Model Tokenizer.

    """

    def __init__(self, hparams: Namespace, tokenizer: Tokenizer):
        super().__init__()
        self.config = hparams
        self.tokenizer = tokenizer

    @classmethod
    def build_input(
        self,
        tokenizer: Tokenizer,
        sentence: List[int],
        label_encoder: Dict[str, int] = None,
        labels: List[float] = None,
        prepare_labels: bool = True,
    ) -> Dict[str, List[int]]:
        if not prepare_labels:
            return {"input_ids": sentence, "input_lengths": len(sentence)}

        label_encoding = [0] * len(label_encoder)
        for l in labels:
            label_encoding[l] = 1
        
        
        output = {
            "input_ids": sentence,
            "input_lengths": [len(sentence)],
            "labels": label_encoding
        }

        return(output)

    @classmethod
    def build_input_context(
            self,
            tokenizer: Tokenizer,
            sentences: List[int],
            label_encoder: Dict[str, int],
            labels: List[float],
           
            
    ) -> Dict[str, List[int]]:
       
       

        label_encoding = [0] * len(label_encoder)
        for l in labels:
            label_encoding[l] = 1         

        input_ids = [tokenizer.bos_index]
        
        for s in sentences:
            input_ids.extend(s)
            input_ids.extend([tokenizer.eos_index])
            
           # if the input is larger than 512 (bert's max input length), trim.
        if len(input_ids) > 512:
            input_ids = input_ids[:511].extend([tokenizer.eos_index])

        output = {
            "input_ids": input_ids,
            "input_lengths": len(input_ids),
            "labels": label_encoding,
        }


        return output

    def _tokenize(self, data: List[Dict[str, str]]):
        for i in tqdm(range(len(data))):
           
            data[i]["text"] = self.tokenizer.encode(str(data[i]["text"]))
            
            data[i]["label"] = [int(data[i]["label"])]
    
        return data

    def _get_dataset(
        self,
        dataset_path: str,
        data_folder: str = "data/",
    ):
        """Loads an Emotion Dataset.

        :param dataset_path: Path to a folder containing the training csv, the development csv's
             and the corresponding labels.
        :param data_folder: Folder used to store data.

        :return: Returns a dictionary with the training and validation data.
        """
        if not os.path.isdir(dataset_path):
            click.secho(f"{dataset_path} not found!", fg="red")

        dataset_hash = (
                int(hashlib.sha256(dataset_path.encode("utf-8")).hexdigest(), 16) % 10 ** 8
        )

        # To avoid using cache for different models
        # split(/) for google/electra-base-discriminator
        pretrained_model = (
            self.config.pretrained_model.split("/")[1]
            if "/" in self.config.pretrained_model
            else self.config.pretrained_model
        )
        dataset_cache = data_folder + ".dataset_" + str(dataset_hash) + pretrained_model

        if os.path.isfile(dataset_cache):
            click.secho(f"Loading tokenized dataset from cache: {dataset_cache}.")
            return torch.load(dataset_cache)

        dataset_path += "" if dataset_path.endswith("/") else "/"

        with open(dataset_path + "labels.txt", "r") as fp:
            labels = [line.strip() for line in fp.readlines()]
            label_encoder = {labels[i]: i for i in range(len(labels))}
         
        if self.config.dataset == "emowoz":
            train, valid, test = load_emowoz(dataset_path)

        elif self.config.dataset == "dailydialog":
            train = load_dailydialog(dataset_path + "dialogues_train.txt", dataset_path + "dialogues_emotion_train.txt")
            valid = load_dailydialog(dataset_path + "dialogues_validation.txt",
                                     dataset_path + "dialogues_emotion_validation.txt")
            test = load_dailydialog(dataset_path + "dialogues_test.txt", dataset_path + "dialogues_emotion_test.txt")

        dataset = {
            "train": train,
            "valid": valid,
            "test": test
        }

        dataset["label_encoder"] = label_encoder
       
        # Tokenize
        dataset["train"] = self._tokenize(dataset["train"])
        dataset["valid"] = self._tokenize(dataset["valid"])
        dataset["test"] = self._tokenize(dataset["test"])
            
        #torch.save(dataset, dataset_cache)
      
        return dataset

    @classmethod
    def pad_dataset(
        cls, dataset: dict, padding: int = 0, padded_inputs: List[str] = PADDED_INPUTS
    ):
        """
        Pad the dataset.
        NOTE: This could be optimized by defining a Dataset class and
        padding at the batch level, but this is simpler.

        :param dataset: Dictionary with sequences to pad.
        :param padding: padding index.
        :param padded_inputs:
        """
        max_l = max(len(x) for x in dataset["input_ids"])
        for name in padded_inputs:
            dataset[name] = [x + [padding] * (max_l - len(x)) for x in dataset[name]]
        return dataset

    def prepare_data(self):
        """
        Lightning DataModule function that will be used to load/download data,
        build inputs with padding and to store everything as TensorDatasets.
        """
        data = self._get_dataset(self.config.dataset_path)
        label_encoder = data["label_encoder"]
        del data["label_encoder"]

        click.secho("Building inputs and labels.", fg="yellow")
        
      
        datasets = {
            "train": defaultdict(list),
            "valid": defaultdict(list),
            "test": defaultdict(list),
        }

       

        if self.config.context:
            for dataset_name, dataset in data.items():
                limit = len(dataset) - 1
                for i, sample in tqdm(enumerate(dataset)):
                    if i >= limit: break

                    # create samples input
                    samples = []
                    samples.append(sample["text"])
                   

                    if i != 0:
                        for turn in range(self.config.context_turns):
                            if sample["dialog_id"] == dataset[i - (turn + 1)]["dialog_id"]:
                                samples.append(dataset[i - (turn + 1)]["text"])

                    flag=0
                    if sample["label"]!=[-1]:    
                        flag=1

                        instance = self.build_input_context(
                            self.tokenizer,
                            samples,
                            label_encoder,
                            sample["label"],
                    )

                    if flag==1:
                        for input_name, input_array in instance.items():
                            datasets[dataset_name][input_name].append(input_array)

        else:
            for dataset_name, dataset in data.items():
                for sample in dataset:
                    flag=0
                    if sample['label']!=[-1]:
                        flag=1

                        instance = self.build_input(
                            self.tokenizer, sample["text"], label_encoder, sample["label"]
                        )
                    if flag==1:
                        for input_name, input_array in instance.items():
                            datasets[dataset_name][input_name].append(input_array)

        click.secho("Padding inputs and building tensors.", fg="yellow")
        tensor_datasets = {"train": [], "valid": [], "test": []}
        for dataset_name, dataset in datasets.items():

            dataset = self.pad_dataset(dataset, padding=self.tokenizer.pad_index)

            for input_name in MODEL_INPUTS:
                if input_name == "labels":
                    tensor = torch.tensor(dataset[input_name], dtype=torch.float32)
                
                else:
                    tensor = torch.tensor(dataset[input_name])
              
                tensor_datasets[dataset_name].append(tensor)
     
        self.train_dataset = TensorDataset(*tensor_datasets["train"])
        self.valid_dataset = TensorDataset(*tensor_datasets["valid"])
        self.test_dataset = TensorDataset(*tensor_datasets["test"])

        click.secho(
           "Train dataset (Batch, Candidates, Seq length): {}".format(
               self.train_dataset.tensors[0].shape
           ),
           fg="yellow",
        )
        click.secho(
           "Valid dataset (Batch, Candidates, Seq length): {}".format(
               self.valid_dataset.tensors[0].shape
           ),
           fg="yellow",
        )
        click.secho(
            "Test dataset (Batch, Candidates, Seq length): {}".format(
                self.test_dataset.tensors[0].shape
            ),
            fg="yellow",
        )

    def train_dataloader(self) -> DataLoader:
        """ Function that loads the train set. """
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=multiprocessing.cpu_count(),
        )

    def val_dataloader(self) -> DataLoader:
        """ Function that loads the validation set. """
        return DataLoader(
            self.valid_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=multiprocessing.cpu_count(),
        )

    def test_dataloader(self) -> DataLoader:
        """ Function that loads the validation set. """
        return DataLoader(
            self.test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=multiprocessing.cpu_count(),
        )
