import pandas as pd
import joblib
import numpy as np

class Utils:

    # Load dataset from a CSV file
    def load_from_csv(self, path):
        return pd.read_csv(path)

    # Placeholder for loading data from MySQL (not implemented)
    def load_from_mysql(self):
        pass
    
    # Prepare features (X) and target (y) from the dataset
    def features_target(self, dataset, drop_cols, y):
        X = dataset.drop(drop_cols, axis=1)
        y = dataset[y].values.ravel()
        return X, y

    # Export trained model to a file
    def model_export(self, clf, score):
        print("Exporting model")
        print(f"Type of clf: {type(clf)}")
        joblib.dump(clf, 'telco_customer_churn/models/best_model.pkl')
