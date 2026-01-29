import logging
import os
from typing import List

from sklearn.model_selection import train_test_split

from lib.utils.jsonl import dump_jsonl

logger = logging.getLogger(__name__)


def get_max_min_aalength(data: List[dict]) -> tuple:
    try:
        len_list = [val["aa_length"] for val in data]
        return max(len_list), min(len_list)
    except Exception as e:
        logger.error(e)
        raise


def create_train_valid_dataset(
    jsonl: List[dict],
    save_dir: str,
    save_tag: str = "",
    valid_data_size: float = 0.2,
    seed=0,
) -> None:
    train_data, valid_data = train_test_split(
        jsonl, test_size=valid_data_size, shuffle=True, random_state=seed
    )
    dump_jsonl(train_data, file_path=os.path.join(save_dir, f"train{save_tag}.jsonl"))
    dump_jsonl(valid_data, file_path=os.path.join(save_dir, f"valid{save_tag}.jsonl"))
