

def matching_pairs(segments):
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


i