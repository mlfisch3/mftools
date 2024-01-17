# Functions for calculating entropy-related quantities for arrays
# default behavior assumes input arrays are images
#     (i.e., unsigned integers in range [0,255] or floats in range [0,1])

import numpy as np
from bit_utils import bit_concat

def entropy(array, bins=255, lo=0, hi=255):

    if array.dtype.name[:5] == 'float':
        array = (array * 255).astype(np.uint8)
    
    counts = np.histogram(array,bins=bins, range=(lo,hi))[0]
    counts = counts[counts>0]
    N = counts.sum()
    return (np.log2(N) - (np.dot(counts, np.log2(counts)) / N)).astype(np.float32)
    

def xentropy(p, q, bins=255, lo=0, hi=255):
    if p.dtype.name[:5] == 'float':
        p = (p * 255).astype(np.uint8)

    if q.dtype.name[:5] == 'float':
        q = (q * 255).astype(np.uint8)
    
    counts_p = np.histogram(p,bins=bins, range=(lo,hi))[0] + 1e-5
    counts_q = np.histogram(q,bins=bins, range=(lo,hi))[0] + 1e-5
    N = counts_q.sum()
    return (np.log2(N) - (np.dot(counts_p, np.log2(counts_q)) / N)).astype(np.float32)


def KL(P,Q, bins=255, lo=0, hi=255):
    return xentropy(P,Q, bins=bins, lo=lo, hi=hi) - entropy(P, bins=bins, lo=lo, hi=hi)


def joint_entropy(p,q, bins=256*256, lo=0, hi=256*256):
    pq = bit_concat(p,q)
    pq_counts = np.histogram(pq, bins=bins, range=(lo,hi))[0]
    pq_counts = pq_counts[pq_counts>0]
    N = pq_counts.sum()
    return (np.log2(N) - (np.dot(pq_counts, np.log2(pq_counts)) / N)).astype(np.float32)


def mutual_information(p,q):
    return entropy(p) + entropy(q) - joint_entropy(p,q)


def conditional_entropy(p,q):
    return joint_entropy(p,q) - entropy(q)
    #return entropy(p) - mutual_information(p,q)


def variation_of_information(p,q):
    return 2 * joint_entropy(p,q) - entropy(p) - entropy(q)
    #return joint_entropy(p,q) - mutual_information(p,q)


def normalized_variation_of_information(p,q):
    joint = joint_entropy(p,q) + 1e-5
    mutual = mutual_information(p,q)
    return (1. - mutual/joint)