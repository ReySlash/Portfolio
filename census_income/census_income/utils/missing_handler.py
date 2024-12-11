import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def number_missing(df) -> int:
    """ Return the number of missing values in the DataFrame. """
    return df.isna().sum().sum()

def number_completes(df) -> int:
    """ Return the number of complete values in the DataFrame. """
    return df.size - number_missing(df)

def missing_variable_summary(df) -> pd.DataFrame:
    """
    Return a summary of the number of missing values in each column
    of the DataFrame.
    """
    return df.isnull().pipe(
        lambda df_1: (
            df_1.sum()
            .reset_index(name="n_missing")
            .rename(columns={"index": "variable"})
            .assign(
                n_cases=len(df_1),
                missing_percentage=lambda df_2: df_2.n_missing / df_2.n_cases * 100,
            )
        )
    )
def missing_summary(df, missing_types=[np.nan,"NA","N/A","n/a","N / A","n / a","/","-","*"," ",None]):
        """
        Return a summary of the number of missing values in each column
        of the DataFrame.
        """
        df_missing = pd.DataFrame()
        for i in missing_types:
            if pd.isna(i):
                obj = df.isna().sum().to_frame(str(i))
            else:
                obj = df.apply(lambda x: x == i).sum().to_frame(i)
            df_missing = pd.concat([df_missing, obj], axis=1)
        return df_missing
    
def missing_variable_plot(df):
    df = missing_variable_summary(df).sort_values("n_missing")

    plot_range = range(1, len(df.index) + 1)

    plt.hlines(y=plot_range, xmin=0, xmax=df.n_missing, color="black")

    plt.plot(df.n_missing, plot_range, "o", color="black")

    plt.yticks(plot_range, df.variable)

    plt.grid(axis="y")

    plt.xlabel("Number missing")
    plt.ylabel("Variable")