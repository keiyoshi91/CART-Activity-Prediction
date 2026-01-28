import numpy as np


def get_mutation_position(mutant_seq: str, original_seq: str) -> str:
    aa_list = []
    if len(original_seq) == len(mutant_seq):
        for i, (aa1, aa2) in enumerate(zip(original_seq, mutant_seq)):
            if aa1 != aa2:
                aa_list.append(f"{aa1}{i}{aa2}")
    else:
        raise NotImplementedError
    return ":".join(aa_list)


def get_mut_vectors(seqs: list, wt_seq: str) -> np.ndarray:
    vectors = []
    for seq in seqs:
        annot = np.empty([1, len(seq)], dtype="<U1")
        vector = np.zeros([1, len(seq)], dtype=np.float16)
        for i, (aa1, aa2) in enumerate(zip(seq, wt_seq)):
            if aa1 != aa2:
                annot[0, i] = aa1
                vector[0, i] = 1
            else:
                annot[0, i] = ""
        vectors.append(vector)
    return np.concatenate(vectors)
