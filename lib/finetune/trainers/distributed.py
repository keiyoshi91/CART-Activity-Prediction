import os

import torch
import torch.distributed as dist


def get_rank() -> int:
    return int(os.getenv("LOCAL", "0"))


def get_local_rank() -> int:
    return int(os.getenv("LOCAL_RANK", "0"))


def get_world_size() -> int:
    return int(os.getenv("WORLD_SIZE", "1"))


def set_master_adder(maseter_addr, master_port) -> None:
    os.environ["MASTER_ADDR"] = maseter_addr
    os.environ["MASTER_PORT"] = master_port


def init_distributed(maseter_addr="localhost", master_port="12345") -> tuple:
    rank = get_rank()
    local_rank = get_local_rank()
    world_size = get_world_size()
    set_master_adder(maseter_addr, master_port)
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    return rank, local_rank, world_size
