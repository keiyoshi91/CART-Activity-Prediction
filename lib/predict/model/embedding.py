from typing import List

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from lib.utils import constants


def get_outputs(seq_list: List[str], plm, batch_size: int = 32) -> list:
    max_length = max([len(seq) for seq in seq_list]) + 2
    tokenizer = plm.tokenizer
    model = plm.masked_LM
    num_layers = plm.config.num_hidden_layers + 1

    encodings = tokenizer.batch_encode_plus(
        seq_list,
        return_tensors="pt",
        max_length=max_length,
        truncation=True,
        padding="max_length",
    )
    encodings_list = [
        encodings["input_ids"],
        encodings["attention_mask"],
    ]

    dataset = TensorDataset(*encodings_list)
    dataloader = DataLoader(
        dataset,
        shuffle=False,
        batch_size=batch_size,
        num_workers=2,
    )

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(DEVICE)
    model.eval()
    reps = []

    for data in dataloader:
        with torch.no_grad():
            input_ids, attention_mask = data
            input_ids = input_ids.to(DEVICE)
            attention_mask = attention_mask.to(DEVICE)
            outputs = model(input_ids, attention_mask)
        reps.append([hs.detach().cpu().numpy() for hs in outputs.hidden_states])

    outputs = []
    for i in range(num_layers):
        outputs.append(np.concatenate([rep[i] for rep in reps], axis=0))
    return outputs


def get_cls_reps(outputs: dict, layer_num: int = -1) -> np.ndarray:
    return outputs[layer_num][:, 0, :]


def get_mean_reps(outputs: dict, layer_num: int = -1) -> np.ndarray:
    return outputs[layer_num].mean(axis=1)


def get_max_reps(outputs: dict, layer_num: int = -1) -> np.ndarray:
    return outputs[layer_num].max(axis=1)


def attention_map(outputs: torch.Tensor) -> torch.Tensor:
    attention_map = torch.mean(outputs.attentions[-1], dim=1)
    return attention_map.detach().cpu().numpy()


def one_hot_vectors(seqs: List[str]) -> np.ndarray:
    one_hot_vecs = []
    vec_size = len(constants.AA_ALPHABET20)
    seq_len = max([len(seq) for seq in seqs])

    for seq in seqs:
        one_hot_vec = []
        for i in range(seq_len):
            one_hot = np.zeros([1, vec_size])
            if i < len(seq):
                idx = constants.AA_ALPHABET20.index(seq[i])
                one_hot[0, idx] = 1
            one_hot_vec.append(one_hot)
        one_hot_vecs.append(np.concatenate(one_hot_vec, axis=1))

    return np.concatenate(one_hot_vecs, axis=0)
