import scipy.stats
import numpy as np


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


def sig_t_val(t_vals):
    sig = list(t_vals.keys())
    std = np.std(sig)
    mean = np.mean(sig)
    out_up = mean + 3 * std
    out_down = mean - 3 * std
    outlying = []
    for x in range(len(sig)):
        if sig[x] >= out_up or sig[x] <= out_down:
            outlying.append(sig[x])
    outlying.sort(reverse=True)
    return outlying


def top_pairs(outlying, t_vals, index):
    """Take the list of outlying t values, a dictionary of pairs between a sample and the
    master chronology and an index for which chronologies should be specified"""
    (master_seg, sample_seg) = t_vals[outlying[index]]
    # Would need to check both lists are sequential for the entire segment using assertion and samples
    return master_seg, sample_seg


def adding_padding(df, samples, master_seg, sample_seg, index):
    indices = np.where(df["master_chronology"] == master_seg[0])
    start = indices[0] - samples[1].index(sample_seg[0]) - 1  # Not sure how the indexing works here
    beginning = df.index[0]
    ending = df.index[-1]
    padding = start[0] - 1
    padding_end = len(df.index) - (padding + len(samples[1]))
    chronology = []
    for i in range(padding):
        chronology.append(None)
        i += 1
    for j in range(len(samples[1])):
        chronology.append(samples[1][j])
        j += 1
    if padding_end > 0:
        for k in range(padding_end):
            chronology.append(None)
            k += 1
    else:
        deletion = int(len(chronology) - (ending - beginning)) - 1
        print("Deletion", deletion)
        for x in range(deletion):
            chronology.pop()
            x += 1

    df['Aligned_' + index] = chronology
