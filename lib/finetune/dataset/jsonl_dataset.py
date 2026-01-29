from typing import Any, Dict, List

from torch.utils.data import Dataset
from transformers.tokenization_utils import PreTrainedTokenizer


class JsonlDataset(Dataset):
    def __init__(
        self,
        data: List[Dict],
        tokenizer: PreTrainedTokenizer,
        max_aa_length: int,
    ) -> None:
        self.data = data
        self.tokenizer = tokenizer
        self.max_aa_length = max_aa_length

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx) -> dict[str, Any]:
        seq = self.data[idx]["sequence"]
        encoded = self.tokenizer.encode_plus(
            seq,
            max_length=self.max_aa_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        return dict(
            input_ids=input_ids.squeeze(0),
            attention_mask=attention_mask.squeeze(0),
        )
