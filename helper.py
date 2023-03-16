from pandas import read_csv, errors
from wotan import flatten
import numpy as np
import logging
import statsmodels.api as sm
from patsy import dmatrices

# TODO: Add appropriate commenting, checking for edge cases and logging


def read_csv_to_dataframe(filename):
    logging.info("Collection data from selected file is being attempted. ")
    try:
        df = read_csv(filename, index_col=0)
        logging.info("Collection data from selected file is successful.")
        return df

    except FileNotFoundError:
        print("File not found. Please check the file is in the current working directory and has been spelt correctly")
        logging.error("File not found, attempt is unsuccessful.")
    except errors.EmptyDataError:
        print("No data.")
        logging.error("File contained no data, attempt is unsuccessful.")

    except errors.ParserError:
        print("Parse error, attempt is unsuccessful.")
        logging.error("Parse error.")


# TODO: Find a way to combine the master chronology and the sample dataframe
# TODO: Find a way of specifying what is the sample and what is part of the master chronology
# TODO: Integrate this with other methods so that errors can be caught
def master_chronology(df):
    """Creating a master chronology by averaging the rows of 20 samples. """
    logging.info("Creation of a master chronology is being attempted.")
    df_master = df.iloc[:, [i for i in range(df.shape[1]) if i <= 19]].copy()

    df_master['master_chronology'] = df_master.mean(axis=1)
    logging.info("Creation of a master chronology is successful.")
    return df_master


def convert_dataframe_to_array(df):
    samples = []

    for col in df.columns:

        col_list = df[col].to_numpy()
        samples.append(col_list[~np.isnan(col_list)])

    return samples


def convert_dataframe_to_list(df):
    samples = []

    for col in df.columns:

        col_list = df[col].tolist()
        cleaned_list = [x for x in col_list if x == x]
        samples.append(cleaned_list)

    return samples

def create_master_chronology(df, samples):
    master_df = df[:19]
    flatten_lc, trend_lc = flatten(time, flux, window_length=0.5, return_trend=True)


def divide_list_to_segments(samples, size, step):
    """Divides a list of lists into a dictionary of lists of smaller lists given
       a size of segments and the step between segments. Recommended size is 10 and
       step is 7."""
    segments = {}
    length = len(samples)
    for j in range(length):
        segments[j] = ([samples[j][i: i + size] for i in range(0, len(samples[j]), step)])
        j += 1
    # print(segments[0])
    # print(segments[1])

    short = []

    for i in range(len(segments)):
        for j in range(len(segments[i])):
            if len(segments[i][j]) != 10:
                # print(segments[i][j])
                short.append(segments[i][j])
                j += 1
            else:
                continue
        i += 1

    # print(short)
    segments0 = [x for x in segments[0] if x not in short]
    segments1 = [x for x in segments[1] if x not in short]

    # print(segments0)

    segments[0] = segments0
    segments[1] = segments1
    segments[0][-1]
    assert len(segments[0][-1]) == 10
    assert len(segments[1][-1]) == 10
    return segments, len(segments[1])
