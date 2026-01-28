import argparse
import json
import logging
import os
import pathlib
import sys

wd = pathlib.Path(__file__).parent.parent.parent.resolve()
sys.path.append(str(wd))

import peft
import torch
import torch.utils
import transformers
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from transformers import (DataCollatorForLanguageModeling, EsmForMaskedLM,
                          EsmTokenizer)

from lib.finetune.dataset.jsonl_dataset import JsonlDataset
from lib.finetune.trainers.distributed import init_distributed
from lib.finetune.trainers.trainer import Trainer
from lib.utils import settings
from lib.utils.jsonl import read_jsonl
from lib.utils.utils import set_random_seed

logging.basicConfig(level=logging.INFO, format=settings.FORMATTER, stream=sys.stdout)
logger = logging.getLogger(__name__)


def main(args):
    try:
        os.makedirs(os.path.join(args.log_dir), exist_ok=True)
        os.makedirs(os.path.join(args.model_save_dir), exist_ok=True)
    except Exception as e:
        logger.error("action=main error=%s", e)
        print(f"error={e}")

    # Logger
    handler = logging.FileHandler(os.path.join(args.log_dir, settings.LOG_NAME))
    handler.setFormatter(logging.Formatter(settings.FORMATTER))
    logger.addHandler(handler)
    logger.info(json.dumps({"args": vars(args)}, indent=4, separators=(",", ": ")))

    # Seup
    rank, local_rank, world_size = init_distributed(master_port=args.master_port)
    device = local_rank

    # Seed
    set_random_seed(args.seed)

    # Tokenizer and Model
    tokenizer = EsmTokenizer.from_pretrained(args.model_name)
    model = EsmForMaskedLM.from_pretrained(args.model_name)

    # Peft
    if args.peft_enable:
        peft_config = peft.LoraConfig(
            task_type="CAUSAL_LM",  # FEATURE_EXTRACTION
            r=8,
            lora_alpha=32,
            target_modules=["query", "value"],
            lora_dropout=0.1,
        )
        model = peft.get_peft_model(model, peft_config)

    # Distributed
    model = (
        DDP(model.to(device), device_ids=[local_rank], find_unused_parameters=True)
        if str_to_bool(args.distributed_enable)
        else model
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=0.1,
    )
    lr_scheduler = transformers.get_scheduler(
        name=args.scheduler_name,
        optimizer=optimizer,
        num_training_steps=args.n_steps,
        num_warmup_steps=int(args.n_steps * args.n_warm_steps_rate),
    )

    # Dataloader
    train_data_path = f"data/generated_sequences/database/{args.db_name}/valid.jsonl"
    valid_data_path = f"data/generated_sequences/database/{args.db_name}/valid.jsonl"
    train_data = read_jsonl(train_data_path)
    valid_data = read_jsonl(valid_data_path)

    max_aa_length = max(
        max([row["aa_length"] for row in train_data]),
        max([row["aa_length"] for row in valid_data]),
    )
    train_dataset = JsonlDataset(
        data=train_data, tokenizer=tokenizer, max_aa_length=max_aa_length
    )
    valid_dataset = JsonlDataset(
        data=valid_data, tokenizer=tokenizer, max_aa_length=max_aa_length
    )

    train_sampler = (
        DistributedSampler(train_dataset, shuffle=True)
        if args.distributed_enable
        else None
    )
    valid_sampler = (
        DistributedSampler(valid_dataset, shuffle=True)
        if args.distributed_enable
        else None
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm_probability=args.mlm_prob
    )
    train_dataloader = DataLoader(
        train_dataset,
        collate_fn=data_collator,
        batch_size=args.batch_size,
        sampler=train_sampler,
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        collate_fn=data_collator,
        batch_size=args.batch_size,
        sampler=valid_sampler,
    )

    # Logger
    writer = SummaryWriter(log_dir=args.log_dir)
    writer.add_hparams(
        {
            "model_name": f"facebook/{args.model_name}",
            "lr": args.lr,
            "scheduler_name": args.scheduler_name,
            "n_steps": args.n_steps,
            "n_warm_steps_rate": args.n_warm_steps_rate,
            "peft_enable": args.peft_enable,
            "mlm_prob": args.mlm_prob,
            "batch_size": args.batch_size,
            "is_clipping": args.is_clipping,
            "clip_coef": args.clip_coef,
        },
        {},
    )

    # Trainer
    model_save_dir = f"results/finetuning/{args.model_name}_{args.db_name}_batch{args.batch_size}_{args.scheduler_name}_warmup{args.n_warm_steps_rate}_lr{args.lr}/models"
    log_dir = f"results/finetuning/{args.model_name}_{args.db_name}_batch{args.batch_size}_{args.scheduler_name}_warmup{args.n_warm_steps_rate}_lr{args.lr}/logs"
    trainer = Trainer(
        seed=args.seed,
        model=model,
        tokenizer=tokenizer,
        device=device,
        rank=rank,
        train_dataloader=train_dataloader,
        valid_dataloader=valid_dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        is_clipping=args.is_clipping,
        clip_coef=args.clip_coef,
        n_steps=args.n_steps,
        model_save_interval=args.model_save_interval,
        model_save_dir=model_save_dir,
        log_dir=log_dir,
        writer=writer,
    )
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--db_name", type=str, choices=["high", "low"])
    parser.add_argument(
        "--model_name",
        type=str,
        choices=[
            "esm2_t6_8M_UR50D",
            "esm2_t12_35M_UR50D",
            "esm2_t30_150M_UR50",
            "esm2_t33_650M_UR50D",
        ],
    )
    parser.add_argument("--lr", default=5e-6, type=float)
    parser.add_argument("--scheduler_name", default="constant", type=str)
    parser.add_argument("--n_steps", default=50, type=float)
    parser.add_argument("--n_warm_steps_rate", default=0.0, type=float)
    parser.add_argument("--peft_enable", action="store_true")
    parser.add_argument("--distributed_enable", action="store_false")
    parser.add_argument("--master_port", default="12345", type=str)
    parser.add_argument("--mlm_prob", default=0.15, type=float)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--is_clipping", action="store_true")
    parser.add_argument("--clip_coef", default=1.0, type=float)
    parser.add_argument("--model_save_interval", default=5, type=float)
    args = parser.parse_args()
    main(args)
