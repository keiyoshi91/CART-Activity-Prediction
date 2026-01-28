import logging
import os
from typing import Tuple, Union

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from lib.finetune.model.esm2 import FinetunedEsmModel, PretrainedEsmModel
from lib.predict.model.embedding import get_outputs
from lib.utils.jsonl import read_jsonl

logger = logging.getLogger(__name__)


def get_dataloaders(
    base_dir: str,
    kfold_num: int,
    plm: Union[PretrainedEsmModel, FinetunedEsmModel],
    batch_size: int = 8,
) -> Tuple[list, list]:
    train_dataloaders = []
    valid_dataloaders = []

    file_dir = os.path.join(base_dir, f"k{kfold_num}")
    if not os.path.exists(file_dir):
        logger.error("action=get_dataloaders status=directory does not exsit")
        raise

    train_data = read_jsonl(os.path.join(file_dir, "train.jsonl"))
    train_data = pd.DataFrame(train_data)
    valid_data = read_jsonl(os.path.join(file_dir, "valid.jsonl"))
    valid_data = pd.DataFrame(valid_data)

    train_dataset = TensorDataset(
        torch.from_numpy(get_outputs(train_data.sequence.tolist(), plm)[-1]),
        torch.from_numpy(train_data.property.values.astype(np.float32)),
    )
    valid_dataset = TensorDataset(
        torch.from_numpy(get_outputs(valid_data.sequence.tolist(), plm)[-1]),
        torch.from_numpy(valid_data.property.values.astype(np.float32)),
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    train_dataloaders.append(train_dataloader)
    valid_dataloaders.append(valid_dataloader)

    return train_dataloaders, valid_dataloaders


def get_dataloaders_batch(
    base_dir: str,
    kfold_num: int,
    plm: Union[PretrainedEsmModel, FinetunedEsmModel],
    batch_size: int = 8,
):
    file_dir = os.path.join(base_dir, f"k{kfold_num}")
    if not os.path.exists(file_dir):
        logger.error("action=get_dataloaders status=directory does not exsit")
        raise

    # Load data
    train_data = read_jsonl(os.path.join(file_dir, "train.jsonl"))
    train_data = pd.DataFrame(train_data)
    valid_data = read_jsonl(os.path.join(file_dir, "valid.jsonl"))
    valid_data = pd.DataFrame(valid_data)

    # Train dataloaders
    train_dataloaders = []
    for _, batch in train_data.groupby("plate"):
        train_dataset = TensorDataset(
            torch.from_numpy(get_outputs(batch.sequence.tolist(), plm)[-1]),
            torch.from_numpy(batch.property.values.astype(np.float32)),
        )
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
        )
        train_dataloaders.append(train_dataloader)

    # Valid dataloader
    valid_dataset = TensorDataset(
        torch.from_numpy(get_outputs(valid_data.sequence.tolist(), plm)[-1]),
        torch.from_numpy(valid_data.property.values.astype(np.float32)),
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    return train_dataloaders, valid_dataloader
