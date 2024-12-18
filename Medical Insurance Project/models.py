import sys
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from utils import Utils

class Models:

    def __init__(self):
        # Dictionary containing model names and their respective instances
        self.regressors = {
            'linear_regression': LinearRegression(),        # Linear Regression model
            'ridge_regression': Ridge(random_state=42),     # Ridge Regression model
            'lasso_regression': Lasso(random_state=42),     # Lasso Regression model
            'random_forest': RandomForestRegressor(random_state=42),  # Random Forest Regressor
            'lightgbm': LGBMRegressor(random_state=42)      # LightGBM Regressor
        }

        # Dictionary containing hyperparameter grids for each model
        self.params = {
            'linear_regression': {
                'pca__n_components': [None],   # Optional PCA component number
            },
            'ridge_regression': {
                'pca__n_components': [None],
                'model__alpha': [0.1,0.15,0.2,0.25,0.3],  # Regularization strength
            },
            'lasso_regression': {
                'pca__n_components': [None],
                'model__alpha': [0.001,0.005,0.01,0.015],  # Regularization strength
            },
            'random_forest': {
                'pca__n_components': [None],
                'model__n_estimators': list(range(50,351,50)),  # Number of trees
                'model__max_depth': [None,5,8,10,12],           # Max depth of trees
                'model__min_samples_split': [2,4,6,8,10],       # Minimum number of samples to split
                'model__min_samples_leaf': [2,5,8,10,12],       # Minimum number of samples per leaf
            },
            'lightgbm': {
                'pca__n_components': [None],
                'model__verbosity': [-1],                   # Warnings silenced
                'model__num_leaves': [13,15,17,18],         # Number of leaves
                'model__max_depth': [-1, 3,5,8,10],         # Max depth of trees
                'model__learning_rate': [0.13,0.15,0.18],   # Learning rate
                'model__n_estimators': [15,25,28,30],       # Number of boosting rounds
            }
        }

    def grid_training(self, X, y):
        """
        Perform hyperparameter tuning and model evaluation using GridSearchCV.
        Separates training and test data, handles evaluation, and identifies the best model.
        """
        best_score = float('-inf')  # Initialize the best score for comparison
        best_model = None  # To store the best model
        best_params = None  # To store the best hyperparameters

        # Split the data into training and testing sets (70% train, 30% test)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Loop through each model
        for name, regressor in self.regressors.items():
            print('\n',f'#'*32, f'Evaluating: {name}','#'*32,'\n')
            
            # Create a pipeline combining PCA and the model
            pipeline = Pipeline([
                ('pca', PCA()),        # PCA for dimensionality reduction
                ('model', regressor)   # Regression model
            ])

            # Define your KFold cross-validation setup (3 folds)
            kf = KFold(n_splits=3, shuffle=True, random_state=42)

            # Ensure the model has associated parameters in the dictionary
            if name not in self.params:
                print(f"Warning: No hyperparameters found for {name}. Skipping...")
                continue
                
            # Initialize GridSearchCV for hyperparameter tuning
            grid_search = GridSearchCV(
                pipeline,
                self.params[name],
                cv=kf,                           # Use K-Fold cross-validation
                n_jobs=-1,                       # Utilize all available processors
                return_train_score=True,         # Return training scores
                error_score='raise',             # Raise errors if encountered
                scoring='neg_mean_squared_error' # Using MSE as the evaluation metric
            )

            try:
                # Fit the model using GridSearchCV
                grid_search.fit(X_train, y_train)
            except Exception as e:
                print(f"Error while training {name}: {e}")
                continue

            # Display the best parameters and cross-validation score
            print("Best parameters:", grid_search.best_params_)
            print("Best cross-validation score (MSE):", grid_search.best_score_)

            # Evaluate the best model on the training and test set
            train_y_pred = grid_search.best_estimator_.predict(X_train)
            train_mse = mean_squared_error(y_train, train_y_pred)
            train_rmse = np.sqrt(train_mse)
            print("Train MSE:", train_mse)
            print("Train RMSE:", train_rmse)
            
            test_y_pred = grid_search.best_estimator_.predict(X_test)
            test_mse = mean_squared_error(y_test, test_y_pred)
            test_rmse = np.sqrt(test_mse)
            print("Test MSE:", test_mse)
            print("Test RMSE:", test_rmse)
            
            # R2 Score
            r2_train = r2_score(y_train, train_y_pred)
            r2_test = r2_score(y_test, test_y_pred)
            print("Train R2:", r2_train)
            print("Test R2:", r2_test)

            # Saving the best params for each model
            csv_name = f"Best_{name}_model"
            pd.DataFrame([grid_search.best_params_]).to_csv(f'models/{csv_name}.csv')
            
            # Update the best model if the current one performs better
            if grid_search.best_score_ > best_score:
                best_score = grid_search.best_score_
                best_model = grid_search.best_estimator_
                best_params = grid_search.best_params_

        # Final output of the best model's parameters and score
        print("-"*32, "\nBest model identified:","-"*32)
        print(f"Parameters: {best_params}")
        print(f"Score (MSE): {best_score}")
        print('-'*64)
        
        # Export the best model
        utils = Utils()
        utils.model_export(best_model, best_score)
        print("\nModel successfully exported.")
