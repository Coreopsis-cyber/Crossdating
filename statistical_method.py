from collections import Counter
import numpy as np
import pandas as pd


def matching_pairs(segments):
    """Statistical method: Returns a list of all
    possible combinations between the sample and the master chronology"""
    master_segments = segments[0]
    sample_segments = segments[1]
    pairs = [(x, y) for x in master_segments for y in sample_segments]
    assert len(pairs) % len(sample_segments) == 0
    return pairs, len(sample_segments)


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
            continue
    return t_vals


def sig_t_val(t_vals, standard_div_no, stride, consecutive):
    """Method finds outlying values of t values and returns a list for which there is a consecutive number of t
    values which are outlying."""
    sig = list(t_vals.keys())
    for i in range(len(sig)):
        sig[i] = float(sig[i].split()[-1])
    std = np.std(sig)
    mean = np.mean(sig)
    out_up = mean + standard_div_no * std
    out_down = mean - standard_div_no * std
    outlying = []
    for x in range(len(sig)):
        if sig[x] >= out_up or sig[x] <= out_down:
            outlying.append(sig[x])
    top_contenders = []
    step = stride + 1
    x = list(map(str, outlying))
    z: str = "-".join(x)
    contender = outlying.copy()
    for y in range(len(outlying)):
        try:
            ind = sig.index(outlying[y])
            s = ind + step
            for i in range(consecutive):
                if z.find(str(sig[s])) == -1:
                    contender.remove(outlying[y])
                    continue
                else:
                    s += step
        except:
            continue
    if len(contender) == 0:
        print(f"Method returned no outliers with {consecutive} consecutive outliers for any start positions. This may "
              "mean there are errors in the sample or this sample is not a match for the inputted master chronology."
              " Adjusting previous constraints such as segment length and the standard deviation number may help"
              " identify the issue.")
    else:
        for i in range(len(contender)):
            top_contenders.append(str(sig.index(contender[i])) + ' ' + str(contender[i]))
    return top_contenders


def top_pairs(df, top_contenders, t_vals, samples, startdate):
    """Take the list of outlying t values, a dictionary of pairs between a sample and the
    master chronology and an index for which chronologies should be specified"""
    if len(top_contenders) == 0:
        print("The sample cannot be aligned because there are no possible start dates. Please repeat the method with "
              "different parameters.")
        return None
    else:
        start_year = []
        for i in range(len(top_contenders)):
            (master_seg, sample_seg) = t_vals[top_contenders[i]]
            start_year.append(df.first_valid_index() + samples[0].index(master_seg[0]) - samples[1].index(sample_seg[0]))
        start_year_dic = Counter(start_year)
        number = start_year_dic.most_common(startdate)
        top_start_years = []
        for i in range(len(number)):
            top_start_years.append(start_year_dic.most_common(startdate)[i][0])

        return top_start_years, start_year_dic


def adding_padding(df, samples, start_year, ind, output):
    """Adds the correct amount of padding for a given start year and adds the resulting list into the original
    dataframe. """
    start = start_year[ind]
    beginning = df.index[0]
    ending = df.index[-1]
    padding = start - beginning
    padding_end = len(df.index) - (padding + len(samples[1]))
    chronology = []
    for i in range(padding):
        chronology.append(None)
    for j in range(len(samples[1])):
        chronology.append(samples[1][j])
    if padding_end > 0:
        for k in range(padding_end):
            chronology.append(None)
    else:
        deletion = int(len(chronology) - (ending - beginning)) - 1
        for x in range(deletion):
            chronology.pop()
    output['Aligned_' + str(ind)] = chronology
    return output

