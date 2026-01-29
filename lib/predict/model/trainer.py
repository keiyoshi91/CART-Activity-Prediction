import csv
import os
import random
from typing import Any, Union

import numpy as np
import torch
from scipy import stats
from tqdm import tqdm

from lib.predict.model.cnn import CnnModel
from lib.predict.model.score import (precision_top5_and_top10,
                                     recall_top5_and_top10)
from lib.utils.jsonl import dump_jsonl


class Trainer:
    def __init__(
        self,
        k: int,
        device: int,
        n_steps: int,
        train_dls: list,
        valid_dl,
        model: Union[CnnModel, Any],
        optimizer: torch.optim.Optimizer,
        loss_fn: torch.nn.MSELoss,
        loss_save_dir: str,
        model_save_dir: str,
        estimate_save_dir: str,
        logger,
    ) -> None:
        self.k = k
        self.device = device
        self.n_steps = n_steps
        self.train_dls = train_dls
        self.valid_dl = valid_dl
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.loss_save_dir = loss_save_dir
        self.model_save_dir = model_save_dir
        self.estimate_save_dir = estimate_save_dir
        self.logger = logger

    def train(self):
        self.logger.info("action=train status=run")

        current_rho = None
        results = []
        progress_bar = tqdm(range(self.n_steps))

        for epoch in range(self.n_steps):
            random.shuffle(self.train_dls)

            # Train
            train_loss = []
            self.model.train()

            for train_dl in self.train_dls:
                for batch in train_dl:
                    inputs, targets = batch
                    inputs = inputs.to(self.device)
                    targets = targets.unsqueeze(1).to(self.device)
                    outputs = self.model(inputs)
                    loss = self.loss_fn(targets, outputs)
                    train_loss.append(loss.item())

                    # Update params
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

            # Validation
            valid_loss = []
            estimates = None
            self.model.eval()

            for batch in self.valid_dl:
                inputs, targets = batch
                inputs = inputs.to(self.device)
                targets = targets.unsqueeze(1).to(self.device)
                outputs = self.model(inputs)
                loss = self.loss_fn(targets, outputs)
                valid_loss.append(loss.item())

                outputs = outputs.detach().cpu().numpy()
                targets = targets.detach().cpu().numpy()
                vals = np.concatenate([outputs, targets], axis=1)
                if estimates is None:
                    estimates = vals
                else:
                    estimates = np.concatenate([estimates, vals], axis=0)

            # Logger
            mean_train_loss = np.mean(train_loss)
            mean_valid_loss = np.mean(valid_loss)
            r = stats.pearsonr(estimates[:, 0], estimates[:, 1]).statistic
            rho = stats.spearmanr(estimates[:, 0], estimates[:, 1]).statistic
            precision_top5, precision_top10 = precision_top5_and_top10(
                estimates[:, 0], estimates[:, 1]
            )
            recall_top5, recall_top10 = recall_top5_and_top10(
                estimates[:, 0], estimates[:, 1]
            )

            result = {
                "epoch": epoch,
                "mean_train_loss": mean_train_loss,
                "mean_valid_loss": mean_valid_loss,
                "pearsonr": r,
                "spearmanr": rho,
                "precision_top5": precision_top5,
                "precision_top10": precision_top10,
                "recall_top5": recall_top5,
                "recall_top10": recall_top10,
            }
            results.append(result)
            self.logger.info(result)

            if current_rho is None:
                current_rho = rho
            else:
                if current_rho < rho:
                    # Save model
                    torch.save(
                        {
                            "epoch": epoch,
                            "state_dict": self.model.state_dict(),
                        },
                        os.path.join(self.model_save_dir, f"best-model_k{self.k}.pth"),
                    )
                    # Save estimates
                    with open(
                        os.path.join(
                            self.estimate_save_dir, f"best-estimate_k{self.k}.csv"
                        ),
                        "w",
                    ) as f:
                        dict_writer = csv.DictWriter(
                            f, fieldnames=["estimate", "actual"]
                        )
                        dict_writer.writeheader()
                        for row in estimates:
                            dict_writer.writerow({"estimate": row[0], "actual": row[1]})
                    current_rho = rho
            progress_bar.update(1)

        dump_jsonl(
            results,
            os.path.join(self.loss_save_dir, f"train-valid_loss_k{self.k}.jsonl"),
        )
        self.logger.info("action=train status=end")
