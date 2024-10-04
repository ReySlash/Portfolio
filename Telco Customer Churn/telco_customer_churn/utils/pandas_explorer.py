import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Attempt to delete the custom accessor if it already exists
try:
    del pd.DataFrame.explorer
except AttributeError:
    pass

@pd.api.extensions.register_dataframe_accessor("explorer")
class explorer(pd.DataFrame):
    """Accessor for enhanced data exploration methods on pandas DataFrames.

    This class adds methods to visualize and analyze numerical and categorical features
    in a DataFrame, including distribution plots, count plots, scatter plots, and 
    percentage tables.
    """

    def numerical_dist(self, feature):
        """Visualize the distribution of a numerical feature using histogram and boxplot.

        Args:
            feature (str): The name of the numerical feature to analyze.

        Returns:
            tuple: A plot of the distribution and a Series with value counts of the feature.
        """
        mean = self[feature].mean()
        median = self[feature].median()
        std = self[feature].std()

        # Create a histogram of the feature
        plt.subplot(2, 1, 1)
        sns.histplot(
            data=self,
            x=feature
        )

        # Draw lines for mean, median, and standard deviations
        plt.axvline(mean, color="red", label='Mean')
        plt.axvline(median, color="orange", label='Median')
        plt.axvline(mean + std, color="green", linestyle='--', label='Mean + 1 Std')
        plt.axvline(mean - std, color="green", linestyle='--', label='Mean - 1 Std')
        plt.legend()

        # Create a boxplot of the feature
        plt.subplot(2, 1, 2)
        sns.boxplot(data=self, x=feature)
        plt.tight_layout()

        return plt.show(), self[feature].value_counts()

    def categorical_dist(self, feature, xtickrotation=0):
        """Visualize the distribution of a categorical feature using a bar plot.

        Args:
            feature (str): The name of the categorical feature to analyze.
            xtickrotation (int): The rotation angle for the x-axis tick labels.

        Returns:
            tuple: A plot of the distribution and a Series with value counts of the feature.
        """
        dist = self[feature].value_counts()
        sns.barplot(x=dist.index, y=dist.values)
        plt.xticks(rotation=xtickrotation)
        return plt.show(), dist
    
    def countplot_hue(self, feature, hue, xtickrotation=0):
        """Create a count plot for a categorical feature, with an additional hue dimension.

        Args:
            feature (str): The name of the feature for the x-axis.
            hue (str): The name of the feature for the hue.
            xtickrotation (int): The rotation angle for the x-axis tick labels.

        Returns:
            None: Displays the count plot.
        """
        sns.countplot(data=self, x=feature, hue=hue)
        bars = plt.gca().patches
        plt.xticks(rotation=xtickrotation)

        # Annotate bars with their heights
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, height + 30, int(height), ha='center')

        return plt.show()
    
    def scatter_corr(self, var1, var2, *args):
        """Create a scatter plot to visualize the correlation between two variables.

        Args:
            var1 (str): The name of the first variable (x-axis).
            var2 (str): The name of the second variable (y-axis).
            args (tuple): Additional arguments for hue.

        Returns:
            None: Displays the scatter plot and prints the Pearson correlation index.
        """
        sns.scatterplot(data=self, x=var1, y=var2, hue=args[0])
        print(f'The Pearson Correlation Index Is: {round(self[var1].corr(self[var2]), 3)}')
        return plt.show()
    
    def percentage_table(self, feature, hue):
        """Generate a percentage table showing the proportion of a hue within each category of a feature.

        Args:
            feature (str): The name of the feature for grouping.
            hue (str): The name of the hue variable.

        Returns:
            DataFrame: A DataFrame containing the counts and percentages for each combination of feature and hue.
        """
        # Count the occurrences of each combination of feature and hue
        df1 = self[[feature, hue]].value_counts().reset_index(name='Count')
        
        # Count the total occurrences of each feature category
        total_counts = self[feature].value_counts().reset_index()
        total_counts.columns = [feature, 'Total']

        # Merge the counts and totals DataFrames
        df1 = pd.merge(df1, total_counts, on=feature)

        # Calculate the percentage
        df1['%'] = round(df1['Count'] / df1['Total'] * 100, 2)

        # Ensure 'Total' and 'Count' columns are integers
        df1[['Total', 'Count']] = df1[['Total', 'Count']].astype(int)
        df1 = df1.reindex(columns=[feature, 'Total', hue, 'Count', '%'])
        
        return df1
