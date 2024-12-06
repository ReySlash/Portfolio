import sys

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, cross_validate

from lightgbm import LGBMClassifier

from utils import Utils
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, roc_auc_score, make_scorer, f1_score, balanced_accuracy_score
from imblearn.under_sampling import RandomUnderSampler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA



class Models:

    def __init__(self):
        # Dictionary containing model names and their respective instances
        self.clas = {
            'logistic_regression': LogisticRegression(random_state=42),  # Logistic Regression model
            'random_forest': RandomForestClassifier(random_state=42),    # Random Forest model
            'lightgbm': LGBMClassifier(random_state=42)                  # LightGBM model
        }

        # Dictionary containing hyperparameter grids for each model
        self.params = {
            'logistic_regression': {
                'pca__n_components': [None],                      # Number of PCA components (optional)            
                # 'model__penalty': ['l2', 'l1', None],                           # Type of regularization (l2 for Ridge regularization, l1 for Lasso)
                # 'model__C': [0.1, 1, 10],                                       # Inverse of regularization strength; smaller values indicate more regularization
                'model__solver': ['liblinear', 'lbfgs', 'newton-cg', 'saga'],   # Optimization algorithm
                # 'model__max_iter': [100, 200, 300],                             # Maximum number of iterations for optimization methods
                'model__class_weight': [None, 'balanced'],                      # Adjust weights for imbalanced classes
                # 'model__fit_intercept': [True, False],                          # Whether to include the intercept term
                # 'model__intercept_scaling': [1, 10],                            # Scaling of the intercept term for certain solvers
                # 'model__warm_start': [True, False]                              # Reuse the solution of the previous call to fit and add more estimators
                'model__max_iter' : [100000]
            },
            'random_forest': {
                'pca__n_components': [None],
                'model__criterion': ['gini', 'entropy'],        # Split criterion
                'model__n_estimators': list(range(20,200,20)),  # Number of trees in the forest
                'model__max_depth': list(range(2,8,2)),  # Maximum depth of the trees
                'model__min_samples_split': [2, 5, 10],  # Minimum number of samples required to split an internal node
                'model__min_samples_leaf': [5, 6, 7],  # Minimum number of samples required to be at a leaf node
                'model__max_features': ['sqrt', 'log2', None],  # Number of features to consider when looking for the best split
                'model__bootstrap': [True],  # Whether bootstrap samples are used when building trees
                # 'model__oob_score': [True, False],  # Whether to use out-of-bag samples to estimate the error
                'model__class_weight': [None, 'balanced'],  # Handle imbalanced classes
                'model__n_jobs': [-1],  # Number of processors to use (use -1 for all)
            },
            # Define the grid of hyperparameters for an SVM model
            'lightgbm': {
                'pca__n_components': [None],
                'model__verbosity': [-1],
                'model__num_leaves': [13, 15, 35],          # Number of leaves controls the model's complexity.
                'model__max_depth': [-1, 5, 7, 9],          # -1 allows unlimited depth.
                'model__learning_rate': [0.05, 0.07, 0.1],  # Learning rate adjusts the step size during training. Lower values improve generalization, while higher values converge faster.
                'model__n_estimators': [50 ,75, 130],       # Number of boosting iterations. Higher values allow deeper learning at the cost of computation time.
                'model__class_weight': [None, 'balanced']   # Handle imbalanced classes
            }
        }

    def grid_training(self, X, y, balanced = False):
        """
        Perform hyperparameter tuning and model evaluation using GridSearchCV.
        Separates training and test data, handles evaluation, and identifies the best model.
        """
        best_score = float('-inf')  # Initialize the best score for comparison
        best_model = None  # To store the best model
        best_params = None  # To store the best hyperparameters

        # Split the data into training and testing sets (70% train, 30% test)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

        
        if balanced:
            # Applying undersampling
            undersampler = RandomUnderSampler(sampling_strategy='auto', random_state=42)
            X_train, y_train = undersampler.fit_resample(X_train, y_train)
                 
        # Loop through each model
        for name, clas in self.clas.items():
            print('\n',f'#'*32, f'Evaluating: {name}','#'*32,'\n')
            
            # Create a pipeline combining PCA and the model
            pipeline = Pipeline([
                ('pca', PCA()),          # PCA for dimensionality reduction
                ('model', clas)          # Model (e.g., logistic regression, random forest)
            ])

            # StratifiedKFold Cross Validation setup (5 folds)
            skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

            # Ensure the model has associated parameters in the dictionary
            if name not in self.params:
                print(f"Warning: No hyperparameters found for {name}. Skipping...")
                continue
            
            # Setting f1-score as main score metric
                    
            # Initialize GridSearchCV for hyperparameter tuning
            grid_search = GridSearchCV(
                pipeline,
                self.params[name],
                cv=skfold,                       # Use K-Fold cross-validation
                n_jobs=-1,                      # Utilize all available processors
                return_train_score=True,        # Return training scores
                error_score='raise',            # Raise errors if encountered
                scoring='balanced_accuracy'    
            )

            try:
                # Fit the model using GridSearchCV
                grid_search.fit(X_train, y_train)
            except Exception as e:
                print(f"Error while training {name}: {e}")
                continue

            # Display the best parameters and cross-validation score
            print("Best parameters:", grid_search.best_params_)
            print("Best cross-validation score:", grid_search.best_score_)

            # Evaluate the best model on the training and test set
            train_y_pred = grid_search.best_estimator_.predict(X_train)
            train_accuracy = accuracy_score(y_train, train_y_pred)
            train_balanced_accuracy = balanced_accuracy_score(y_train, train_y_pred)
            print("Train Accuracy:", train_accuracy)
            print("Train Balanced-Accuracy:", train_balanced_accuracy)
            
            test_y_pred = grid_search.best_estimator_.predict(X_test)
            test_accuracy = accuracy_score(y_test, test_y_pred)
            test_balanced_accuracy = balanced_accuracy_score(y_test, test_y_pred)
            print("Test Accuracy:", test_accuracy)
            print("Test Balanced-Accuracy:", test_balanced_accuracy)
            
            c_matrix = confusion_matrix(y_test, test_y_pred)
            print(f"Confusion Matrix:\n{pd.DataFrame(c_matrix)}")
            
            # Classification Report
            print("\nClassification Report:\n", classification_report(y_test, test_y_pred))

            # If the model supports predict_proba, calculate AUC
            if hasattr(grid_search.best_estimator_.named_steps['model'], 'predict_proba'):
                y_pred_proba = grid_search.best_estimator_.predict_proba(X_test)[:, 1]
                auc = roc_auc_score(y_test, y_pred_proba)
                print(f"AUC: {auc}")
            
            
            # Saving the best params for each class
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
        print(f"Score: {best_score}")
        print('-'*64)
        
        # Export the best model
        utils = Utils()
        utils.model_export(best_model, best_score)
        print("\nModel successfully exported.")

