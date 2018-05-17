import numpy as np
import torch
import torch.nn.functional as f

def normalize(inputs, p=2, dim=1):
    """
    Performs `L_p` normalization of inputs over specified dimension.

    Args:
        input: input tensor of any shape
        p (float): the exponent value in the norm formulation. Default: 2
        dim (int): the dimension to reduce. Default: 1 (rows)
    """
    return f.normalize(inputs, p=p, dim=dim)

def make_constructed_utts(utt_tokens):
    """
    Converts utterances into correct form for composition
    Args:
        utt_tokens: List of utterances (utterance = list of tokens)

    Returns: 
        utts:
            List of utterances for individual tokens and compositions
        compose_idx:
            List of indices in utts to be composed
        [["green", "-ish"]] => 
        [["#start#", "green", "#end#"],
         ["#start#", "-ish", "#end#"],
         ["#start#", "green", "-ish", "#end#"]], 
        [(0,2)]

    """
    START = "#start#"
    END = "#end#"
    utts = []
    compose_idx = []
    for utt in utt_tokens:
        compose_idx.append((len(utts), len(utts)+len(utt))) # [start, end)
        for tok in utt:
            utts.append([START, tok, END])
        utts.append([START] + utt + [END])
    return utts, compose_idx

def meaning_pointwise_product(meanings):
    """
    Computes composition of meanings using pointwise product
    Args:
        meanings:
            (Utterance count) x (Color count) 
                tensor of meaning values
    Returns:
        (Color count)
            tensor of meaning values
    """
    result = torch.ones(*meanings[0].size())
    for u in range(0, meanings.size(0)): # For each utterance
        result *= meanings[u]

    return result

def meaning_pointwise_difference(composed_meaning, word_meanings):
    results = []
    for u in range(word_meanings.size(0)):
        results.append(torch.max(composed_meaning - word_meanings[u], torch.zeros(1)))

    return results

# def normalize_and_recenter(meanings):
#     comp_mean = []
#     comp_std = []
#     for u in range(1, meanings.size(0)):
#         m = meanings[u].numpy()
    
#         # shape (ncolors)
#         utt_mean = np.mean(numpy_image)
#         utt_std = np.std(numpy_image)
        
#         comp_mean.append(utt_mean)
#         comp_std.append(utt_std)

#     comp_mean = np.array(comp_mean).mean()
#     comp_std = np.array(comp_std).mean()
#     return f.normalize(meanings, dim=0)

# def recenter(inputs, dim=1):
#     """
#     Mean-centers the data, then recenters at 0.5

#     Args:
#         input: input tensor of any shape
#         dim (int): the dimension to reduce. Default: 1 (rows)
#     """
