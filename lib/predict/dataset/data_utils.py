from typing import List

import pandas as pd


def load_dataframe(file_path_list: List[str]) -> pd.DataFrame:
    loaded_df = []
    for i, file_path in enumerate(file_path_list):
        df = pd.read_csv(file_path)
        df = df[df.id.apply(lambda x: x.startswith("bm"))]
        df.insert(0, "exp", i)
        loaded_df.append(df)
    return pd.concat(loaded_df)


def create_jsonl_dataset(df: pd.DataFrame) -> List[dict]:
    dataset = []
    for i, (_, data) in enumerate(df.groupby(["exp", "plate"])):
        for id, rows in data.groupby("id"):
            dataset.append(
                {
                    "plate": i,
                    "id": id,
                    "sequence": rows.aa_seq.unique().item(),
                    "delta_killing": rows.delta_killing.mean(),
                }
            )
    return dataset
