import numpy as np

def make_sua_datum_token_frequency_fn(D):
    data_set = D.get_data()
    token_counts = dict()
    for d in data_set:
        utts = d.get("utterances")
        for utt in utts:
            strs = utt["nlp"]["clean_strs"]["strs"]
            for s in strs:
                if s not in token_counts:
                    token_counts[s] = 0.0
                token_counts[s] += 1.0
            
    for token in token_counts.keys():
        token_counts[token] = np.log(token_counts[token])

    def sua_datum_utt_frequency_fn(datum):
        score = 0.0
        token_count = 0.0
        utterances = datum.get("utterances")
        for utt in utterances:
            strs = utt["nlp"]["clean_strs"]["strs"]
            for s in strs:
                score += token_counts[s]
            token_count += len(strs)
        score /= token_count
        return -score

    return sua_datum_utt_frequency_fn

def make_sua_datum_utt_length_fn(D):
    def sua_datum_utt_length_fn(datum):
        l = 0
        utterances = datum.get("utterances")
        for u in utterances:
            l += len(u["nlp"]["clean_strs"]["strs"])
        return l
    return sua_datum_utt_length_fn
