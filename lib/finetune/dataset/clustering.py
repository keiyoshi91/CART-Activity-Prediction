import json
import os
from collections import Counter, defaultdict
from typing import List

import pandas as pd
from sklearn.cluster import KMeans

from lib.predict.model.embedding import one_hot_vectors
from lib.utils import constants


class ClusteringHomologSequence:
    def __init__(
        self,
        input_fasta_path: str,
        template_seq: str,
        thr: float = 1.25,
        reduction_rate: float = 0.05,
        save_json: bool = False,
        seed: int = 0,
    ) -> None:
        self.input_fasta_path = input_fasta_path
        self.template_seq = template_seq
        self.thr = thr
        self.reduction_rate = reduction_rate
        self.save_json = save_json
        self.seed = seed
        self.aa_list = constants.AA_ALPHABET20

    def create_homolog_seq_df(self, n_cluster=None):
        seqs = self.get_unique_seq_from_fasta()
        seqs = self.remove_shorter_longer_seq(seqs)
        seqs = self.remove_non20aa_seq(seqs)
        if n_cluster is None:
            n_cluster = int(len(seqs) * self.reduction_rate)
        df = self.clustering_seqs(seqs, n_cluster)
        df = self.remove_minor_cluster(df)
        return df

    def get_unique_seq_from_fasta(self) -> List[str]:
        seqs = defaultdict(str)
        with open(self.input_fasta_path, "r") as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                if line.startswith(">"):
                    id = line.split("/")[0].replace(">", "").split("_")[0]
                    if seqs.get(id):
                        seqs[id] = ""
                    continue
                else:
                    if not seqs.get(id):
                        seqs[id] = line.replace("\n", "")
                    else:
                        seqs[id] += line.replace("\n", "")
            if self.save_json:
                json_output_dir = os.path.dirname(self.input_fasta_path)
                save_name = os.path.basename(self.input_fasta_path).replace(".fa", "")
                with open(os.path.join(json_output_dir, f"{save_name}.json"), "w") as f:
                    f.writelines(json.dumps(seqs, indent=4, separators=(",", ": ")))
        return list(set(seqs.values()))

    def remove_shorter_longer_seq(self, seqs: List[str]) -> List[str]:
        max_len = len(self.template_seq) * self.thr
        min_len = len(self.template_seq) * (2 - self.thr)
        seqs = [seq for seq in seqs if len(seq) >= min_len and len(seq) <= max_len]
        return seqs

    def remove_non20aa_seq(self, seqs: List[str]):
        outputs = []
        for seq in seqs:
            for aa in seq:
                if aa not in self.aa_list:
                    break
            else:
                outputs.append(seq)
                continue
        return outputs

    def clustering_seqs(self, seqs: List[str], n_clusters: int) -> pd.DataFrame:
        embs = one_hot_vectors(seqs)
        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=self.seed).fit(
            embs
        )
        df = pd.DataFrame({"sequence": seqs, "cluster": list(kmeans.labels_)})
        return df.sort_values("cluster")

    def remove_minor_cluster(self, df):
        counts = dict(Counter(df.cluster.values))
        thr_num = int(len(df) * (self.reduction_rate / 2))
        cluster_num = [k for k, v in counts.items() if v >= thr_num]
        df = df[df.cluster.apply(lambda x: x in cluster_num)]
        return df
