import scipy.stats
import numpy as np

"""def matching_pairs(segments):
    master_segments = segments['master_chronology']
    sample_segments = segments['sample']
    pairs = {}
    if len(master_segments) > len(sample_segments):
        for i in range(len(master_segments)):
            for j in range(len(sample_segments)):
                pairs[-i] = (master_segments[i], sample_segments[j])
    else:
        for i in range(len(sample_segments)):
            for j in range(len(master_segments)):
                pairs[-i] = (master_segments[i], sample_segments[j])
    return pairs
"""


def matching_pairs(segments):
    """Statistical method: Returns a list of all
    possible combinations between the sample and the master chronology"""
    master_segments = segments['master_chronology']
    sample_segments = segments['sample']

    pairs = [(x, y) for x in master_segments for y in sample_segments]
    return pairs


# TODO: Research the relative magnitudes of t values. Find a way to test this.
def t_values(pairs):
    t_vals = {}
    for i in range(len(pairs)):
        (seg1, seg2) = pairs[i]
        if len(seg1) == len(seg2) == 10:
            r = scipy.stats.pearsonr(seg1, seg2)[0]
            n = min(len(seg1), len(seg2))
            t = (r - np.sqrt(n - 2)) / np.sqrt(1 - (r * r))
            t_vals[t] = (seg1, seg2)
    return t_vals
