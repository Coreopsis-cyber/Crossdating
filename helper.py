from pandas import read_csv, errors
import logging


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


def rename_dataframe(df):

    assert len(df.columns) == 2, "Dataframe does not contain 2 columns please reenter a new csv in the correct form."
    column = list(df.columns)
    df = df.rename({column[0]: "master_chronology", column[1]: "sample"}, axis=1)
    return df


def convert_dataframe_to_list(df):
    samples = []

    for col in df.columns:

        col_list = df[col].tolist()
        cleaned_list = [x for x in col_list if x == x]
        samples.append(cleaned_list)

    return samples


def divide_list_to_segments(samples, size):
    """Divides a list of lists into a dictionary of lists of smaller lists given
       a size of segments and the step between segments. Recommended size is 10 and
       step is 7."""
    segments = {}
    length = len(samples)
    for j in range(length):
        segments[j] = ([samples[j][i: i + size] for i in range(0, len(samples[j]), 1)])
        j += 1
    short = []
    for i in range(len(segments)):
        for j in range(len(segments[i])):
            if len(segments[i][j]) != size:
                short.append(segments[i][j])
                j += 1
            else:
                continue
        i += 1
    segments[0] = [x for x in segments[0] if x not in short]
    segments[1] = [x for x in segments[1] if x not in short]
    assert len(segments[0][-1]) == size
    assert len(segments[1][-1]) == size
    return segments
