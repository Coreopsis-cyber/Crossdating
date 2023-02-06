import seaborn as sns


def comparing_result_and_actual(df):
    sns.set(rc={'figure.figsize': (20, 13)})
    graph = sns.lineplot(data=df[["sample", 'Aligned']])
    return graph


def master_and_sample(df):
    sns.set(rc={'figure.figsize': (20, 13)})
    graph = sns.lineplot(data=df[["master_chronology", 'Aligned']])
    return graph
