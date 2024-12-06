import sys
from models import Models
from utils import Utils

if __name__ == "__main__":

    models = Models()
    utils = Utils()
    
    # Save original stdout to restore later
    original_stdout = sys.stdout
    print('Starting Training...')
    
    try:
        # Open file to save output
        file = open('models/models_evaluation.txt', 'w')
        sys.stdout = file
        
        # Load data and prepare features and target
        data = utils.load_from_csv('census_income/data/selected_features.csv')
        X, y = utils.features_target(
            dataset=data, 
            drop_cols=['income_ >50K'],
            y='income_ >50K'
        )

        # Train models
        models.grid_training(X, y)

        # Close file and restore stdout
        file.close()
        sys.stdout = original_stdout
        
        print("Training has finished successfully")
    except Exception as e:
        sys.stdout = original_stdout
        print(e)
