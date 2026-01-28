from copy import deepcopy

import numpy as np
import torch

from lib.finetune.model.esm2 import PretrainedEsmModel


def compute_pseudo_loglikelihood(aa_seq: str, plm: PretrainedEsmModel) -> float:
    whole_ll = compute_whole_loglikelihood(aa_seq, plm)
    ppl = np.mean(whole_ll)
    return ppl


def compute_whole_loglikelihood(aa_seq: str, plm: PretrainedEsmModel) -> list:
    aa_seq = "M" + aa_seq
    vocab_size = len(plm.tokenizer.get_vocab())
    encoding = plm.tokenizer.encode(aa_seq, return_tensors="pt")
    labels = deepcopy(encoding)

    ppl_list = []
    start_posi = 2  # skip cls token and head 'Methionie'
    end_posi = encoding.shape[1] - 1  # skip eos token
    for mask_posi in range(start_posi, end_posi):
        input_ids = deepcopy(encoding)
        input_ids[:, mask_posi] = plm.tokenizer.mask_token_id
        true_label = labels[:, mask_posi].item()
        with torch.no_grad():
            input_ids = input_ids.to("cuda")
            labels = labels.to("cuda")
            model = plm.masked_LM.to("cuda")
            output = model(input_ids, labels=labels)
            scores = output.logits.view(-1, vocab_size)
        score = scores[mask_posi].to("cpu")
        prob = torch.nn.Softmax(dim=0)(score)
        prob = prob.detach().numpy()
        ppl_list.append(np.log(prob[true_label]))
    return ppl_list
