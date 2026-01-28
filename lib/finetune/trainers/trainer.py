import logging
import os

import numpy as np
import torch
from tqdm import tqdm

from lib.finetune.trainers.score import compute_perplexity
from lib.utils import settings

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(
        self,
        seed,
        model,
        tokenizer,
        device,
        rank,
        train_dataloader,
        valid_dataloader,
        optimizer,
        lr_scheduler,
        is_clipping,
        clip_coef,
        n_steps,
        log_dir,
        model_save_dir,
        model_save_interval,
        writer,
    ):
        self.seed = seed
        self.model = model.to(device)
        self.tokenzier = tokenizer
        self.device = device
        self.rank = rank
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.is_clipping = is_clipping
        self.clip_coef = clip_coef
        self.n_steps = int(n_steps)
        self.log_dir = log_dir
        self.model_save_dir = model_save_dir
        self.model_save_interval = model_save_interval
        self.writer = writer

        # Logger
        handler = logging.FileHandler(os.path.join(self.log_dir, settings.LOG_NAME))
        handler.setFormatter(logging.Formatter(settings.FORMATTER))
        logger.addHandler(handler)

    def train(self):
        logger.info("action=train status=run")
        training_steps = self.n_steps * len(self.train_dataloader)
        progress_bar = tqdm(range(training_steps))

        current_loss = None

        for epoch in range(1, self.n_steps + 1):
            # Training steps
            self.model.train()
            train_loss = []

            for batch in self.train_dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                loss = outputs.loss
                train_loss.append(loss.item())

                self.optimizer.zero_grad()
                loss.backward()
                if self.is_clipping:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.clip_coef
                    )
                self.optimizer.step()
                progress_bar.update(1)
            self.lr_scheduler.step()

            # Validation steps
            self.model.eval()
            valid_loss = []
            batch_perplexity = []

            for batch in self.valid_dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                with torch.no_grad():
                    outputs = self.model(**batch)
                    loss = outputs.loss
                    valid_loss.append(loss.item())
                perplexity = compute_perplexity(batch, outputs)
                batch_perplexity.append(perplexity)

            mean_train_loss = np.mean(train_loss)
            mean_valid_loss = np.mean(valid_loss)
            mean_perplexity = np.mean(batch_perplexity)

            # Logger
            self.writer.add_scalar("loss/train", mean_train_loss, epoch)
            self.writer.add_scalar("loss/valid", mean_valid_loss, epoch)
            self.writer.add_scalar("score/perplexity", mean_perplexity, epoch)
            logger.info(
                {
                    "epoch": epoch,
                    "train_loss": mean_train_loss,
                    "valid_loss": mean_valid_loss,
                    "perplexity": mean_perplexity,
                }
            )

            # Save model
            if epoch % self.model_save_interval == 0:
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                    },
                    os.path.join(self.model_save_dir, f"epoch{epoch}.pth"),
                )

            if current_loss is None:
                current_loss = mean_valid_loss
                continue

            if mean_valid_loss < current_loss:
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                    },
                    os.path.join(self.model_save_dir, f"best.pth"),
                )
                current_loss = mean_valid_loss

        logger.info("action=train status=end")
