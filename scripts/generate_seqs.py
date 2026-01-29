import argparse
import json
import logging
import os
import pathlib
import sys

wd = pathlib.Path(__file__).parent.parent.parent.resolve()
sys.path.append(str(wd))

from lib.finetune.dataset.homolog_combine import (
    HighDiversityHomologCombinedSequences,
    LowDiversityHomologCombinedSequences)
from lib.finetune.dataset.utils import create_train_valid_dataset
from lib.utils import constants, settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main(args):
    try:
        os.makedirs(os.path.join(args.log_dir, args.db_name_1), exist_ok=True)
        os.makedirs(os.path.join(args.log_dir, args.db_name_2), exist_ok=True)
    except Exception as e:
        print(f"error={e}")

    fh = logging.FileHandler(os.path.join(args.log_dir, args.log_name))
    fh.setFormatter(logging.Formatter(settings.FORMATTER))
    logger.addHandler(fh)
    logger.info("args: %s", json.dumps(vars(args), indent=4, separators=(",", ": ")))

    # input data info
    homolog_info = [
        {
            "input_fasta_path": args.fasta_file_path_1,
            "template_seq": constants.HINGE_28 + constants.TM_28 + constants.COSTIM_28,
            "save_json": args.save_json,
        },
        {
            "input_fasta_path": args.fasta_file_path_2,
            "template_seq": constants.ACTIVATION_3Z,
            "save_json": args.save_json,
        },
    ]

    # High diversity sequences
    high_div = HighDiversityHomologCombinedSequences(homolog_info)
    high_div_seqs = high_div.generate_combined_seqs(args.gen_num)
    create_train_valid_dataset(
        jsonl=high_div_seqs,
        save_dir=os.path.join(wd, args.save_dir, args.db_name_1),
        valid_data_size=args.valid_data_size,
        seed=args.seed,
    )

    # Low diversity sequences
    low_div = LowDiversityHomologCombinedSequences(homolog_info)
    low_div_seqs = low_div.generate_combined_seqs()
    create_train_valid_dataset(
        jsonl=low_div_seqs,
        save_dir=os.path.join(wd, args.save_dir, args.db_name_2),
        valid_data_size=args.valid_data_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--db_name_1", default="high", type=str)
    parser.add_argument("--db_name_2", default="low", type=str)
    parser.add_argument("--log_dir", default="data/generated_sequences", type=str)
    parser.add_argument("--log_name", default="args.log", type=str)
    parser.add_argument("--save_dir", default="data/generated_sequences", type=str)
    parser.add_argument(
        "--fasta_file_path_1",
        default="data/generated_sequences/phmmer/refprot_CD28_114-220.fa",
        type=str,
    )
    parser.add_argument(
        "--fasta_file_path_2",
        default="data/generated_sequences/phmmer/refprot_CD3z_52-164.fa",
        type=str,
    )
    parser.add_argument("--save_json", action="store_true")
    parser.add_argument("--gen_num", default=5500, type=int)
    parser.add_argument("--valid_data_size", default=0.2, type=float)
    args = parser.parse_args()
    main(args)
