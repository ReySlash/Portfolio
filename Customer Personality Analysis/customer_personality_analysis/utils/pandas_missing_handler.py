import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import missingno
import janitor
import upsetplot
import itertools
from statsmodels.graphics.mosaicplot import mosaic

try:
    del pd.DataFrame.missing
except AttributeError:
    pass

@pd.api.extensions.register_dataframe_accessor("missing")
class MissingHandler:
    def __init__(self, pandas_obj):
        if isinstance(pandas_obj, pd.DataFrame):
            self._df = pandas_obj
        else:
            raise ValueError("The given object is not a Pandas DataFrame.")

    def number_missing(self) -> int:
        """ Return the number of missing values in the DataFrame. """
        return self._df.isna().sum().sum()

    def number_completes(self) -> int:
        """ Return the number of complete values in the DataFrame. """
        return self._df.size - self.number_missing()

    def missing_variable_summary(self) -> pd.DataFrame:
        """
        Return a summary of the number of missing values in each column
        of the DataFrame.
        """
        return self._df.isnull().pipe(
            lambda df_1: (
                df_1.sum()
                .reset_index(name="n_missing")
                .rename(columns={"index": "variable"})
                .assign(
                    n_cases=len(df_1),
                    pct_missing=lambda df_2: df_2.n_missing / df_2.n_cases * 100,
                )
            )
        )

    def summary(self, missing_types=[np.nan,"NA","N/A","n/a","N / A","n / a","/","-","*"," ",None]):
        """
        Return a summary of the number of missing values in each column
        of the DataFrame.
        """
        df_missing = pd.DataFrame()
        for i in missing_types:
            if pd.isna(i):
                obj = self._df.isna().sum().to_frame(str(i))
            else:
                obj = self._df.apply(lambda x: x == i).sum().to_frame(i)
            df_missing = pd.concat([df_missing, obj], axis=1)
        return df_missing

    def missing_case_summary(self) -> pd.DataFrame:
        """
        Return a summary of the number of missing values in each row
        of the DataFrame.
        """
        return self._df.assign(
            case=lambda df: df.index,
            n_missing=lambda df: df.apply(
                axis="columns", func=lambda row: row.isna().sum()
            ),
            pct_missing=lambda df: df["n_missing"] / df.shape[1] * 100,
        )[["case", "n_missing", "pct_missing"]]
    
    def missing_variable_run(self, variable) -> pd.DataFrame:
        rle_list = self._df[variable].pipe(
            lambda s: [[len(list(g)), k] for k, g in itertools.groupby(s.isnull())]
        )
    
        return pd.DataFrame(data=rle_list, columns=["run_length", "is_na"]).replace(
            {False: "complete", True: "missing"}
        )
        
    def missing_variable_span(self, variable: str, span_every: int) -> pd.DataFrame:
        return (
            self._df.assign(
                span_counter=lambda df: (
                    np.repeat(a=range(df.shape[0]), repeats=span_every)[: df.shape[0]]
                )
            )
            .groupby("span_counter")
            .aggregate(
                n_in_span=(variable, "size"),
                n_missing=(variable, lambda s: s.isnull().sum()),
            )
            .assign(
                n_complete=lambda df: df.n_in_span - df.n_missing,
                pct_missing=lambda df: df.n_missing / df.n_in_span * 100,
                pct_complete=lambda df: 100 - df.pct_missing,
            )
            .drop(columns=["n_in_span"])
            .reset_index()
        )
    
    def sort_variables_by_missingness(self, ascending=False):

        return self._df.pipe(
            lambda df: (df[df.isna().sum().sort_values(ascending=ascending).index])
        )
        
    def create_shadow_matrix(
        self,
        true_string: str = "Missing",
        false_string: str = "Not Missing",
        only_missing: bool = False,
        suffix: str = "_NA",
    ) -> pd.DataFrame:
        return (
            self._df.isna()
            .pipe(lambda df: df[df.columns[df.any()]] if only_missing else df)
            .replace({False: false_string, True: true_string})
            .add_suffix(suffix)
        )
    
    def bind_shadow_matrix(
        self,
        true_string: str = "Missing",
        false_string: str = "Not Missing",
        only_missing: bool = False,
        suffix: str = "_NA",
    ) -> pd.DataFrame:
        return pd.concat(
            objs=[
                self._df,
                self._df.missing.create_shadow_matrix(
                    true_string=true_string,
                    false_string=false_string,
                    only_missing=only_missing,
                    suffix=suffix,
                ),
            ],
            axis="columns",
        )
        
    def missing_scan_count(self, search) -> pd.DataFrame:
        return (
            self._df.apply(axis="rows", func=lambda column: column.isin(search))
            .sum()
            .reset_index()
            .rename(columns={"index": "variable", 0: "n"})
            .assign(original_type=self._df.dtypes.reset_index()[0])
        )

     # Plotting functions ---

    def missing_variable_plot(self):
        df = self._df.missing.missing_variable_summary().sort_values("n_missing")

        plot_range = range(1, len(df.index) + 1)

        plt.hlines(y=plot_range, xmin=0, xmax=df.n_missing, color="black")

        plt.plot(df.n_missing, plot_range, "o", color="black")

        plt.yticks(plot_range, df.variable)

        plt.grid(axis="y")

        plt.xlabel("Number missing")
        plt.ylabel("Variable")
    
    def missing_case_plot(self):

        df = self._df.missing.missing_case_summary()

        sns.displot(data=df, x="n_missing", binwidth=1, color="black")

        plt.grid(axis="x")
        plt.xlabel("Number of missings in case")
        plt.ylabel("Number of cases")
        
    def missing_variable_span_plot(
        self, variable: str, span_every: int, rot: int = 0, figsize=None
    ):

        (
            self._df.missing.missing_variable_span(
                variable=variable, span_every=span_every
            ).plot.bar(
                x="span_counter",
                y=["pct_missing", "pct_complete"],
                stacked=True,
                width=1,
                color=["black", "lightgray"],
                rot=rot,
                figsize=figsize,
            )
        )
        
    def missing_upsetplot(self, variables: list[str] = None, **kwargs):

        if variables is None:
            variables = self._df.columns.tolist()

        return (
                self._df.isna()
                .value_counts(variables)
                .pipe(lambda df: upsetplot.plot(df, **kwargs))
            )
    def scatter_imputation_plot(
        self, x, y, imputation_suffix="_imp", show_marginal=False, **kwargs
    ):

        x_imputed = f"{ x }{ imputation_suffix }"
        y_imputed = f"{ y }{ imputation_suffix }"

        plot_func = sns.scatterplot if not show_marginal else sns.jointplot

        return (
            self._df[[x, y, x_imputed, y_imputed]]
            .assign(is_imputed=lambda df: df[x_imputed] | df[y_imputed])
            .pipe(lambda df: (plot_func(data=df, x=x, y=y, hue="is_imputed", **kwargs)))
        )
        
    def missing_mosaic_plot(
        self,
        target_var: str,
        x_categorical_var: str,
        y_categorical_var: str,
        ax = None
    ):
        return (
            self._obj
            .assign(
                **{target_var: lambda df: df.weight.isna().replace([True, False], ["NA", "!NA"])}
            )
            .groupby(
                [x_categorical_var, y_categorical_var, target_var],
                dropna=False,
                as_index=True,
            )
            .size()
            .pipe(
                lambda df: mosaic(
                    data=df,
                    properties=lambda key: {"color": "r" if "NA" in key else "gray"},
                    ax=ax,
                    horizontal=True,
                    axes_label=True,
                    title="",
                    labelizer=lambda key: "",
                )
            )
        )