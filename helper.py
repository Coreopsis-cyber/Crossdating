import pandas as pd
import numpy as np
# TODO: Add appropriate commenting, checking for edge cases and logging


def read_csv_to_dataframe(filename):
    try:
        df = pd.read_csv(filename, index_col=0)
        return df

    except FileNotFoundError:
        print("File not found. Please check the file is in the current working directory and has been spelt correctly")

    except pd.errors.EmptyDataError:
        print("No data.")

    except pd.errors.ParserError:
        print("Parse error.")


# TODO: Find a way to combine the master chronology and the sample dataframe
# TODO: Find a way of specifying what is the sample and what is part of the master chronology
def master_chronology(df):
    """Creating a master chronology from 20 samples. """
    df_master = df.iloc[:, [i for i in range(df.shape[1]) if i <= 19]].copy()

    df_master['master_chronology'] = df_master.mean(axis=1)

    return df_master


def convert_dataframe_to_array(df):
    samples = []

    for col in df.columns:

        col_list = df[col].to_numpy()
        samples.append(col_list[~np.isnan(col_list)])

    return samples


def split_with_overlap(array, len_chunk, len_sep=1):
    """Returns a matrix of all full overlapping chunks of the input `array`, with a chunk
    length of `len_chunk` and a separation length of `len_sep`. Begins with the first full
    chunk in the array.  """

    n_arrays = int(np.ceil((array.size - len_chunk + 1) / len_sep))

    array_matrix = np.tile(array, n_arrays).reshape(n_arrays, -1)

    columns = np.array(((len_sep * np.arange(0, n_arrays)).reshape(n_arrays, -1) + np.tile(
        np.arange(0, len_chunk), n_arrays).reshape(n_arrays, -1)), dtype=np.intp)

    rows = np.array((np.arange(n_arrays).reshape(n_arrays, -1) + np.tile(
        np.zeros(len_chunk), n_arrays).reshape(n_arrays, -1)), dtype=np.intp)

    return array_matrix[rows, columns]


def divide_array_into_segments(samples):
    segments = {}
    length = len(samples)

    for i in range(length):
        g = split_with_overlap(samples[i], 10, 7)
        i += 1
        segments[i] = g

    return segments


def convert_dataframe_to_list(df):
    samples = []

    for col in df.columns:

        col_list = df[col].tolist()
        cleaned_list = [x for x in col_list if x == x]
        samples.append(cleaned_list)

    return samples


def divide_list_to_segments(samples, size, step):
    """Divides a list of lists into a dictionary of lists of smaller lists given
       a size of segments and the step between segments. Recommended size is 10 and
       step is 7."""
    segments = {}

    length = len(samples)
    for j in range(length):
        segments[j] = ([samples[j][i: i + size] for i in range(0, len(samples[j]), step)])
        j += 1

    return segments
