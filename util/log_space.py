import numpy as np

log = np.log
exp = np.exp


def log_add(*logits):
    """
    Assumes logits are all negative
    """
    z = min(*logits)
    return z + log(sum(exp(l - z) for l in logits))
