import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

try:
    del pd.DataFrame.custom
except AttributeError:
    pass

@pd.api.extensions.register_dataframe_accessor("custom")
class CustomAccessor:
    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    def distribution_analysis(self, feature, kde=False):
        plt.figure(figsize=(10, 6))
        plt.subplot(2, 1, 1)
        plt.title(f"{feature.name} distribution")
        sns.histplot(
            data=self._obj,
            x=feature,
            kde=kde
        )
        plt.axvline(feature.mean(), color='red')
        plt.axvline(feature.median(), color='orange')
        plt.axvline(feature.mean() + feature.std(), color='green')
        plt.axvline(feature.mean() - feature.std(), color='green')
        plt.subplot(2, 1, 2)
        sns.boxplot(
            data=self._obj,
            x=feature
        )
        return plt.show()

    def frequency_table(self, column):
        data = self._obj[column].value_counts().reset_index()
        f_table = pd.DataFrame(data=data)
        f_table.columns = [column, 'count']
        f_table['FI'] = f_table['count'].cumsum()
        f_table['hi'] = f_table['count'] / f_table['count'].sum()
        return f_table

