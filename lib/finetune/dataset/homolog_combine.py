import copy
import itertools
import logging
import random
from typing import List

import pandas as pd

from lib.finetune.dataset.clustering import ClusteringHomologSequence

logger = logging.getLogger(__name__)


class HighDiversityHomologCombinedSequences:
    def __init__(self, homolog_info: List[dict]) -> None:
        self.homolog_info = homolog_info
        self.seq_df_list = self.create_seq_df_list()

    def create_seq_df_list(self) -> List[pd.DataFrame]:
        seq_df_list = []
        for input in self.homolog_info:
            cls_hs = ClusteringHomologSequence(**input)
            seq_df = cls_hs.create_homolog_seq_df()
            seq_df_list.append(seq_df)
        return seq_df_list

    def generate_combined_seqs(self, gen_num: int) -> List[dict]:
        current_num = 0
        generated_seqs = set()
        outputs = []
        while current_num < gen_num:
            seq = self.pick_and_combine_seqs()
            generated_seqs.add(seq)
            if len(generated_seqs) == current_num:
                continue
            else:
                outputs.append(
                    {
                        "id": "hc" + f"{current_num+1}".zfill(5),
                        "aa_length": len(seq),
                        "sequence": seq,
                    }
                )
                current_num += 1
        return outputs

    def pick_and_combine_seqs(self) -> str:
        selected_seqs = [self.random_select_seq(df) for df in self.seq_df_list]
        combined_seq = "".join(selected_seqs)
        return combined_seq

    @staticmethod
    def random_select_seq(df: pd.DataFrame) -> str:
        cls_num = random.choice(df.cluster.unique())
        seqs = df[df.cluster == cls_num].sequence.tolist()
        seq = random.choice(seqs)
        return seq


class LowDiversityHomologCombinedSequences:
    def __init__(self, homolog_info: List[dict], n_cluster: int = 1) -> None:
        self.homolog_info = homolog_info
        self.n_cluster = n_cluster
        self.seq_df_list = self.get_target_seq_df_list()

    def get_target_seq_df_list(self) -> List[pd.DataFrame]:
        target_seq_df_list = []
        for input in self.homolog_info:
            cls_hs = ClusteringHomologSequence(**input)
            n_cluster = copy.copy(self.n_cluster)
            while True:
                seq_df = cls_hs.create_homolog_seq_df(n_cluster)
                target_cls_num = seq_df[
                    seq_df.sequence == cls_hs.template_seq
                ].cluster.item()
                target_seq_df = seq_df[seq_df.cluster == target_cls_num]
                if len(target_seq_df) <= int(len(seq_df) * 0.25):
                    logger.info(
                        "action=get_target_seq_df_list input=%s n_cluster=%s",
                        input,
                        n_cluster,
                    )
                    target_seq_df_list.append(target_seq_df)
                    break
                else:
                    n_cluster += 1
        return target_seq_df_list

    def generate_combined_seqs(self) -> List[dict]:
        outputs = []
        seqs = [target_df.sequence.tolist() for target_df in self.seq_df_list]
        combs = itertools.product(*seqs)
        for i, comb in enumerate(combs, start=1):
            sequence = "".join(comb)
            aa_length = len(sequence)
            outputs.append(
                {
                    "id": "hc" + f"{i+1}".zfill(5),
                    "aa_length": aa_length,
                    "sequence": sequence,
                }
            )
        return outputs
