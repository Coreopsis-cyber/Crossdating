import logging

import scipy.stats
import numpy as np


def matching_pairs(segments):
    """Statistical method: Returns a list of all
    possible combinations between the sample and the master chronology"""
    master_segments = segments[0]
    sample_segments = segments[1]
    pairs = [(x, y) for x in master_segments for y in sample_segments]
    assert len(pairs) / len(sample_segments) == int
    return pairs, len(sample_segments)


# TODO: Research the relative magnitudes of t values. Find a way to test this.
def t_values(pairs, size):
    t_vals = {}
    for i in range(len(pairs)):
        (seg1, seg2) = pairs[i]
        if len(seg1) == len(seg2) == size:
            r = np.corrcoef(seg1, seg2)[1, 0]
            n = min(len(seg1), len(seg2))
            t = (r * (np.sqrt(n - 2)) / (np.sqrt(1 - (r * r))))
            t_vals[str(i) + " " + str(t)] = (seg1, seg2)
        else:
            logging.ERROR("T-value failed pair " + seg1, seg2)
    return t_vals


def sig_t_val(t_vals, stride):
    sig = list(t_vals.keys())
    std = np.std(sig)
    mean = np.mean(sig)
    out_up = mean + 3 * std
    out_down = mean - 3 * std
    outlying = []
    for x in range(len(sig)):
        if sig[x] >= out_up or sig[x] <= out_down:
            outlying.append(sig[x])
    top_contenders = []
    stri = stride+1
    x = list(map(str, outlying))
    z = "-".join(x)
    contender = []
    for y in range(len(outlying)):
        ind = sig.index(outlying[y])
        if z.find(str(sig[ind + stri])) != -1:
            if z.find(str(sig[ind + stri + stri])) != -1:
                if z.find(str(sig[ind + stri + stri + stri])) != -1:
                    if z.find(str(sig[ind + stri + stri + stri + stri])) != -1:
                        if z.find(str(sig[ind + stri + stri + stri + stri + stri])) != -1:
                            if z.find(str(sig[ind + stri + stri + stri + stri + stri + stri])) != -1:
                                if z.find(str(sig[ind + stri + stri + stri + stri + stri + stri + stri])) != -1:
                                    if z.find(str(sig[ind + stri + stri + stri + stri + stri + stri + stri + stri])) != -1:
                                        contender.append(outlying[y])
                                        y += 1
                                    else:
                                        y += 1
                                else:
                                    y += 1
                            else:
                                y += 1
                        else:
                            y += 1

                    else:
                        y += 1
                else:
                    y += 1
            else:
                y += 1
        else:
            y += 1
            for i in range(3):
                max_contender = (max(contender, key=abs))
                contender.remove(max_contender)
                max_contender = str(str(sig.index(max_contender)) + ' ' + str(max_contender))
                top_contenders.append(max_contender)
                i += 1
    return top_contenders


def top_pairs(top_contenders, t_vals, index):
    """Take the list of outlying t values, a dictionary of pairs between a sample and the
    master chronology and an index for which chronologies should be specified"""
    (master_seg, sample_seg) = t_vals[top_contenders[index]]
    # Would need to check both lists are sequential for the entire segment using assertion and samples
    return master_seg, sample_seg


def adding_padding(df, samples, master_seg, sample_seg, index):
    indices = np.where(df["master_chronology"] == master_seg[0])
    start = indices[0]-samples[1].index(sample_seg[0])  # Not sure how the indexing works here
    beginning = df.index[0]
    ending = df.index[-1]
    padding = start[0]
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
