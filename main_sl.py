from src.training.sl_training.data_loader import DataLoader
from src.training.sl_training.train_pipeline import TrainingPipeline
from src.training.sl_training.model_definitions import ModelFactory

CONFIG = {
    "data_path": "E:/saas-idea-validator/data/processed/vectorised_dataset.csv",
    "vector_col": "vector",
    "target_col": "label_numeric"
}

def main():
    data_loader = DataLoader(CONFIG["data_path"], vector_col=CONFIG["vector_col"], target_col=CONFIG["target_col"])
    pipeline = TrainingPipeline(data_loader)

    # Get models and optional hyperparameter grids
    models_dict, param_grids = ModelFactory.get_models_and_grids()

    pipeline.run(models_dict=models_dict, param_grids=param_grids, cv_folds=10)

if __name__ == "__main__":
    main()
