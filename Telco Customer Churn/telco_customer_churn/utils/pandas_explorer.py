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
        sns.histplot(data=self, x=feature)

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

    def dist_pieplot(self, feature):
        """Visualize the distribution of a categorical feature using a pie chart.

        Args:
            feature (str): The name of the categorical feature to analyze.

        Returns:
            tuple: A plot of the distribution and a Series with value counts of the feature.
        """
        
        plt.title(f'{feature} distribution')
        # Calculate the frequencies of each category
        values = self[feature].value_counts().sort_index()

        # Set Colors
        colors = sns.color_palette("tab10", n_colors=len(self)) 
        # Create a pie chart
        wedges, texts, autotexts = plt.pie(
            values,
            labels=values.index,
            autopct='%1.2f%%',
            startangle=90,
            pctdistance=0.85,
            colors=colors
        )
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        plt.tight_layout()


    def categorical_dist(self, feature, xtickrotation=0, ytickrotation=0, orient=None):
        """Visualize the distribution of a categorical feature using a bar plot.

        This function generates a bar plot showing the frequency of each category within 
        a specified categorical feature. If the number of categories exceeds 10 or if 
        `orient='h'` is specified, a horizontal bar plot is generated; otherwise, a vertical 
        bar plot is created. Percentage labels are added next to each bar, indicating the 
        proportion of each category relative to the total.

        Args:
            feature (str): The name of the categorical feature to analyze.
            xtickrotation (int): The rotation angle for x-axis tick labels 
                                (applicable only for vertical bar plots).
            ytickrotation (int): The rotation angle for y-axis tick labels 
                                (applicable only for horizontal bar plots).
            orient (str, optional): Orientation of the bar plot; use 'h' for a horizontal plot. 
                                    Defaults to None, which creates a vertical plot if the 
                                    number of categories is 10 or fewer.

        Returns:
            tuple: A plot of the distribution and a Series with the value counts of the feature.
        """
        dist = self[feature].value_counts()

        # Calculate the percentage of the total
        percentages = (dist / dist.sum()) * 100

        # Create a DataFrame to facilitate the plot
        percentages_df = percentages.reset_index()
        percentages_df.columns = ['Category', 'Percentage']

        if self[feature].value_counts().count() > 10 or orient == 'h':
            # Generate the horizontal bar plot
            bar_plot = sns.barplot(y=dist.index, x=dist.values, orient='h')

            # Add percentage labels to the side of each bar
            for index, row in percentages_df.iterrows():
                width = bar_plot.patches[index].get_width()
                bar_plot.text(width + 2, index, f'{row.Percentage:.2f}%', color='black', va='center')

            plt.yticks(rotation=ytickrotation)
            plt.xlabel('Count')
            plt.ylabel(feature)
        else:
            # Generate the bar plot       
            bar_plot = sns.barplot(x=dist.index, y=dist.values)

            # Add percentage labels above each bar
            for index, row in percentages_df.iterrows():
                height = bar_plot.patches[index].get_height()
                bar_plot.text(index, height + 2, f'{row.Percentage:.2f}%', color='black', ha='center')

            plt.xticks(rotation=xtickrotation)
            plt.xlabel('Count')
            plt.ylabel(feature)

    def countplot_hue(self, feature, hue, xtickrotation=0, orient=None):
        """Create a count plot for a categorical feature, with an additional hue dimension.

        Args:
            feature (str): The name of the feature for the x-axis.
            hue (str): The name of the feature for the hue.
            xtickrotation (int): The rotation angle for the x-axis tick labels.

        Returns:
            None: Displays the count plot.
        """
        plt.title(f'{feature} distribution by {hue}')
        if orient is None:
            sns.countplot(data=self, x=feature, hue=hue)
            bars = plt.gca().patches
            plt.xticks(rotation=xtickrotation)

            # Annotate bars with their heights
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width() / 2, height + 30, int(height), ha='center')
        else:
            sns.countplot(data=self, y=feature, hue=hue, orient='h')
            bars = plt.gca().patches

            # Annotate bars with their width
            for bar in bars:
                width = bar.get_width()
                plt.text(width + 0.5, bar.get_y() + bar.get_height() / 2, int(width), va='center')
        plt.tight_layout()

    def scatter_corr(self, var1, var2, hue=None):
        """Create a scatter plot to visualize the correlation between two variables.

        Args:
            var1 (str): The name of the first variable (x-axis).
            var2 (str): The name of the second variable (y-axis).
            args (tuple): Additional arguments for hue.

        Returns:
            None: Displays the scatter plot and prints the Pearson correlation index.
        """
        sns.scatterplot(data=self, x=var1, y=var2, hue=hue)

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

    def feature_describe(self, feature):
        """Generate descriptive statistics for a feature.

        Args:
            feature (str): The name of the feature to describe.

        Returns:
            DataFrame: A DataFrame containing descriptive statistics for the feature.
        """
        index = ['count', 'mean', 'median', 'std', 'max', 'min', 'IQR', 'Q1', 'Q2', 'Q3']
        values = [
            len(self[feature]),
            self[feature].mean(),
            self[feature].median(),
            self[feature].std(),
            self[feature].max(),
            self[feature].min(),
            self[feature].max() - self[feature].min(),
            self[feature].quantile(0.25),
            self[feature].quantile(0.5),
            self[feature].quantile(0.75)
        ]
        statistics = pd.DataFrame(values, index=index, columns=['values'])

        return statistics

    def frequency_table(self, feature):
        """Generate a frequency table for a given feature in a DataFrame.

        Args:
            feature (str): The name of the feature to analyze.

        Returns:
            pd.DataFrame: A DataFrame containing the frequency and percentage distribution of the feature.
        """
        dist = self[feature].value_counts()
        percentage = dist / dist.sum() * 100
        frequency_table = pd.DataFrame({'Frequency': dist, 'Percentage': percentage})

        return frequency_table

