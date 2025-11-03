from src.training.sl_training.data_loader import DataLoader
from src.training.sl_training.train_pipeline import TrainingPipeline
from src.training.sl_training.model_definitions import ModelFactory
import os

CONFIG = {
    "data_path": "E:/saas-idea-validator/data/processed/vectorised_dataset.csv",
    "vector_col": "vector",
    "target_col": "label_numeric",
    "model_save_dir": "E:/saas-idea-validator/models",
    "report_dir": "reports"
}

def main():
    # Load your data
    data_loader = DataLoader(
        CONFIG["data_path"], 
        vector_col=CONFIG["vector_col"], 
        target_col=CONFIG["target_col"]
    )

    # Initialize training pipeline
    pipeline = TrainingPipeline(
        data_loader, 
        report_dir=CONFIG["report_dir"], 
        model_save_dir=CONFIG["model_save_dir"]
    )

    # Get all models & hyperparameter grids
    models_dict, param_grids = ModelFactory.get_models_and_grids()

    # Run training, CV, logging, and auto-save models
    results = pipeline.run(models_dict=models_dict, param_grids=param_grids, cv_folds=10)

    # Print all saved model paths
    print("\n[SAVED MODELS]")
    for model_name in models_dict.keys():
        model_file = os.path.join(CONFIG["model_save_dir"], f"{model_name}_model.joblib")
        if os.path.exists(model_file):
            print(f"{model_name}: {model_file}")
        else:
            print(f"{model_name}: NOT FOUND!")

if __name__ == "__main__":
    main()
