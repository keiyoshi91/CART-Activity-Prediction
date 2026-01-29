import numpy as np
import torch
import torch.nn.functional as F
from transformers.modeling_outputs import MaskedLMOutput


def compute_perplexity(
    batch_inputs: dict, batch_outputs: MaskedLMOutput, mask_token_id: int = 32
) -> float:
    input_ids = batch_inputs["input_ids"]
    labels = batch_inputs["labels"]
    logits = batch_outputs.logits

    true_ids = torch.where(input_ids == mask_token_id, labels, input_ids)
    masks = (input_ids == mask_token_id).long()  # -> (batch_size, aa_length)

    log_prob = F.log_softmax(logits, dim=-1)  #  -> (batch_size, aa_length, aa_ids)
    log_prob = torch.gather(log_prob, dim=-1, index=true_ids.unsqueeze(-1)).squeeze(
        -1
    )  # -> (batch_size, aa_length)

    mean_val = ((log_prob * masks).sum() / masks.sum()).item()
    perplexity = np.exp(-mean_val)

    return perplexity
