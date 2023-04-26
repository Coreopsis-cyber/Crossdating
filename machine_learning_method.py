import numpy as np


def setting_up_training_data(training_data):
    """Creates the input and output values for the training data."""
    i = 1
    total_input = []
    total_output = []
    for data in training_data:
        input, output = training_data_for_first_mlp(data)
        total_input += input
        total_output += output
        i += 1

    return total_input, total_output


def correct_training_pairs(training_data):
    """ Creates the output for the 3rd MLP:
    Creates a list of outputs where a value of 1 is given to a sample and master chronologies that align in the
    correct position and a value of 0 is given to a sample and master chronology not correctly aligned """
    i = 1
    correct_pairs_training = []
    for data in training_data:
        input = correct_pairs(data)
        correct_pairs_training.append(input)
        i += 1

    return correct_pairs_training


def training_data_for_first_mlp(df):
    """Creates the input and output data for the first mlp"""
    samples = []
    samples1 = []
    answer = []
    segments = {}
    i = 0
    for col in df.columns:
        col_list = df[col].tolist()
        cleanedList = [x for x in col_list if x == x]
        samples.append(cleanedList)

    size = 100
    step = 50
    length = len(samples)
    segments[0] = ([samples[0][i: i + size] for i in range(0, len(samples[0]), step)])

    short0 = []
    for j in range(len(segments[0])):
        if len(segments[0][j]) != 100:
            short0.append(segments[0][j])
        else:
            continue

    size = 10
    step = 1
    length = len(samples)
    segments[1] = ([samples[1][i: i + size] for i in range(0, len(samples[1]), step)])

    short1 = []

    for j in range(len(segments[1])):
        if len(segments[1][j]) != 10:

            short1.append(segments[1][j])

        else:
            continue

    segments0 = [x for x in segments[0] if x not in short0]
    segments1 = [x for x in segments[1] if x not in short1]

    segments[0] = segments0
    segments[1] = segments1

    assert len(segments[0][-1]) == 100
    assert len(segments[1][-1]) == 10
    segments0 = segments[0]
    segments1 = segments[1]
    pairs = [(x + y) for x in segments0 for y in segments1]

    while i < (df.shape[0] - 100):
        new = df.iloc[i:i + 100]
        for col in new.columns:
            col_list = new[col].tolist()
            cleanedList = [x for x in col_list if x == x]
            samples1.append(cleanedList)
        i += 50

    assert len(samples1[-2]) == 100

    i = 1
    while i < len(samples1):

        if len(samples1[i]) < 10:
            for j in range(len(segments[1])):
                answer.append(0)
        elif len(samples1[i]) == 10:
            if find_element_in_list(samples1[i], segments[1]) != None:
                index = find_element_in_list(samples1[i], segments[1])
                for j in range(len(segments[1]) - index - 1):
                    answer.append(0)
                answer.append(1)
                for j in range(len(segments[1]) - index):
                    answer.append(0)
            elif find_element_in_list(samples[i], segments[1]) is None:
                for j in range(len(segments[1])):
                    answer.append(0)
        elif len(samples1[i]) > 10:
            size = 10
            step = 1
            length = len(samples1[i])
            seg={}
            seg[0] = ([samples1[i][j: j + size] for j in range(0, len(samples1[i]), step)])
            short2 = []
            for j in range(len(seg[0])):
                if len(seg[0][j]) != 10:

                    short2.append(seg[0][j])
                else:
                    continue
            final = [x for x in seg[0] if x not in short2]

            for j in range(len(segments[1])):
                if find_element_in_list(segments[1][j], final) is not None:

                    answer.append(1)
                else:

                    answer.append(0)
        i += 2

    return pairs, answer


def correct_pairs(df):
    """Creates a list of sample and master chronologies that align in the correct position"""
    pair = []
    correct = []
    samples = []
    i = 0
    for col in df.columns:
        col_list = df[col].tolist()
        cleanedList = [x for x in col_list if x == x]
        samples.append(cleanedList)
    for i in range(np.where(df.iloc[:, 1] == samples[1][0])[0][0],
                   (np.where(df.iloc[:, 1] == samples[1][-1])[0][0] - 8)):
        new = df.iloc[i:i + 10]

        for col in new.columns:
            col_list = new[col].tolist()

            pair.append(col_list)
        correct.append(pair[0] + pair[1])
        pair = []
    no_repeat_correct = [i for n, i in enumerate(correct) if i not in correct[:n]]

    return no_repeat_correct


def find_element_in_list(element, list_element):
    """Helper method that checks if an element is in a list"""
    try:
        index_element = list_element.index(element)
        return index_element
    except ValueError:
        return None


def create_pairs(dataframe, section):
    """Creates segments of master and segment pairs from a dataframe"""
    samples = []
    segments = {}
    segment = {}
    size = 10
    step = 1
    i = 0
    for df in dataframe:
        col_list = df.iloc[:, 1].tolist()
        cleanedList = [x for x in col_list if x == x]
        samples.append(cleanedList)

    for j in range(len(section)):
        key = int(j)
        segments[key] = ([section[j][i: i + size] for i in range(0, len(section[j]), step)])

    for k in range(len(samples)):
        key = int(len(section) + k)
        segments[key] = ([samples[k][i: i + size] for i in range(0, len(samples[k]), step)])

    short = []

    for i in range(len(segments)):
        for j in range(len(segments[i])):
            if len(segments[i][j]) != 10:
                short.append(segments[i][j])
            else:
                continue

    master = []
    for i in range(len(section)):
        master.append([x for x in segments[i] if x not in short])
    sample = []
    for i in range(len(samples)):
        sample.append([x for x in segments[i + len(section)] if x not in short])

    segment[0] = [item for sublist in master for item in sublist]
    segment[1] = [item for sublist in sample for item in sublist]

    segments0 = segment[0]
    segments1 = segment[1]
    pairs = [(x + y) for x in segments0 for y in segments1]
    assert len(pairs[-1]) == 20

    return pairs


def training_data_for_second_mlp(pairs, correct_pairs):
    """Creates the training data for the second mlp:
    Creates the output for the second mlp giving segments that overlap by at least 5 years then it is given a value of 1"""
    binary = []
    first_pairs = []
    for j in range(len(correct_pairs)):
        first_pairs.append(correct_pairs[j][:5])
    correct_pairs = [item for sublist in correct_pairs for item in sublist]
    i = 0
    while i < len(pairs):

        if pairs[i] in correct_pairs and pairs[i] in first_pairs:
            for t in range(len(first_pairs)):
                if pairs[i] in first_pairs[t]:
                    for j in range(0, first_pairs[t].index(pairs[i])):
                        binary.pop()
                    for h in range(0, first_pairs[t].index(pairs[i])):
                        binary.append(1)
                    for f in range(6):
                        binary.append(1)
                    i += 6
        elif pairs[i] in correct_pairs:
            for z in range(0, 5):
                binary.pop()
            for f in range(0, 11):
                binary.append(1)
            i += 6
        else:
            binary.append(0)
            i += 1
    return binary


def testing_data_for_second_mlp(pairs, correct_pairs):
    """This method is only used when debugging the method"""
    binary = []
    found_pairs = []
    i = 0
    while i < len(pairs):
        if pairs[i] in correct_pairs:
            if correct_pairs.index(pairs[i]) < 5:
                for j in range(0, correct_pairs.index(pairs[i])):
                    binary.pop()
                for h in range(0, correct_pairs.index(pairs[i])):
                    binary.append(1)
                for f in range(6):
                    binary.append(1)

                i += 6
            else:
                for z in range(0, 5):
                    binary.pop()
                if i < len(pairs) - 6:
                    for f in range(0, 11):
                        binary.append(1)
                else:
                    for f in range(5 + len(pairs) - i):
                        binary.append(1)
                i += 6
        else:
            binary.append(0)

            i += 1
    return binary


def testing_data_for_third_mlp(pairs, correct_pairs):
    """This method is only used when debugging the method"""
    binary = []
    for i in range(len(pairs)):
        if pairs[i] in correct_pairs:
            binary.append(1)
        else:
            binary.append(0)
    return binary


def training_data_for_third_mlp(pairs, correct_pairs):
    """Creates a list of outputs for the training data"""
    binary = []
    correct_pairs = [item for sublist in correct_pairs for item in sublist]
    for i in range(len(pairs)):
        if pairs[i] in correct_pairs:
            binary.append(1)
        else:
            binary.append(0)
    return binary


def adding_padding(df, samples, start_year, ind, output):
    """Adds the correct amount of padding for a given start year and adds the resulting list into the original
    dataframe. """
    start = start_year.most_common(ind)[0][0]
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
    output['Aligned_' + str(0)] = chronology
    return output
