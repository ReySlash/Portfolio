import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import (
    QuantileTransformer,
    PowerTransformer
)


def apply_transformation(data, column, transformation='Quantile-Transform', 
                                  n_quantiles=1000, output_distribution='normal'):
    """
    Applies the specified transformation to a column in a dataset and plots the distribution 
    before and after the transformation, including histograms and boxplots.

    Parameters:
        data (pd.DataFrame): The dataset containing the original data.
        column (str): The column to be transformed.
        transformation (str): The transformation to apply ('Quantile-Transform', 'Box-cox', 'Yeo-Johnson').
        n_quantiles (int): Number of quantiles for QuantileTransformer (default 1000).
        output_distribution (str): Desired output distribution for QuantileTransformer ('uniform' or 'normal').

    Returns:
        np.array: The transformed column as a numpy array.
    """
    # Initial validation
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in the dataset.")
    
    # Preserve original data
    original_data = data[column].dropna().values.reshape(-1, 1)

    # Apply the specified transformation
    if transformation == 'Quantile-Transform':
        transformer = QuantileTransformer(n_quantiles=n_quantiles, output_distribution=output_distribution, random_state=42)
        transformed_data = transformer.fit_transform(original_data)

    elif transformation == 'Box-cox':
        # Box-Cox requires positive values, so check if the column has any non-positive values
        if (original_data <= 0).any():
            raise ValueError("Box-Cox transformation requires strictly positive values.")
        transformer = PowerTransformer(method='box-cox')
        transformed_data = transformer.fit_transform(original_data)

    elif transformation == 'Yeo-Johnson':
        transformer = PowerTransformer(method='yeo-johnson')
        transformed_data = transformer.fit_transform(original_data)

    else:
        raise ValueError("Invalid transformation type. Choose from 'Quantile-Transform', 'Box-cox', or 'Yeo-Johnson'.")

    # Create a DataFrame for easy plotting
    df_plot = pd.DataFrame({
        'Original': original_data.flatten(),
        'Transformed': transformed_data.flatten()
    })
    
    # Create a figure with 2 rows and 2 columns
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
    
    # Histogram of original data
    sns.histplot(df_plot['Original'], bins=30, ax=axes[0, 0])
    axes[0, 0].set_title('Original Distribution')
    axes[0, 0].set_xlabel(column)
    
    # Histogram of transformed data
    sns.histplot(df_plot['Transformed'], bins=30, ax=axes[0, 1], color='salmon')
    axes[0, 1].set_title('Transformed Distribution')
    axes[0, 1].set_xlabel(f"{column} (Transformed)")
    
    # Boxplot of original data
    sns.boxplot(x=df_plot['Original'], ax=axes[1, 0])
    axes[1, 0].set_title('Original Boxplot')
    axes[1, 0].set_xlabel(column)
    
    # Boxplot of transformed data
    sns.boxplot(x=df_plot['Transformed'], ax=axes[1, 1], color='salmon')
    axes[1, 1].set_title('Transformed Boxplot')
    axes[1, 1].set_xlabel(f"{column} (Transformed)")
    
    # Adjust layout to avoid overlapping
    plt.tight_layout()
    plt.show()
    
    # Return the transformed data as a numpy array
    return transformed_data.flatten()