import optuna
from configs.config import Config
from main import process_dataset
from src.utils.common import set_seed
import os
import json
def objective(trial,filename='best_trial_16s_1.json'):
    # Create a new config and modify using trial suggestions
    config = Config()
    set_seed(config.SEED)

    config.LR_CLASSIFIER = trial.suggest_float("lr_classifier", 1e-5, 1e-1, log=True)
    config.WEIGHT_DECAY_CLASSIFIER = trial.suggest_float("weight_decay_classifier", 1e-6, 1e-2, log=True)
    config.DROPOUT_LINEAR_CLASSIFIER = trial.suggest_float("dropout_linear_classifier", 0.0, 0.5)
    config.DROPOUT_TRANSFORMER = trial.suggest_float("dropout_transformer", 0.0, 0.5)
    config.NUM_LAYERS_ENCODER = trial.suggest_int("num_layers_encoder", 1, 4)
    config.HIDDEN_DIM_CLASSIFIER1 = trial.suggest_int("hidden_dim_classifier1", 4, 128)
    config.REG_LAMBDA = trial.suggest_float("reg_lambda", 0.0, 1.0)
    config.CONTRASTIVE_LAMBDA = trial.suggest_float("contrastive_lambda", 0.0, 1.0)
    config.CONTRASTIVE_MARGIN = trial.suggest_float("contrastive_margin", 0.1, 5.0)
    config.LR_FINE_TUNE_W = trial.suggest_float("lr_fine_tune_w", 1e-5, 1e-1, log=True)
    config.NHEAD_CLASSIFIER = trial.suggest_categorical("nhead_classifier", [1, 2, 4, 8])

    config.FLAG_TYPE_RUN = 'single_tag'
    config.RANDOM_INITIALIZE_EMBEDDINGS = False
    config.WANDB_NAME_CLASSIFIER = 'optuna_classifier'

    # Define dataset here
    dataset_group = "Parkinson_16S"
    dataset_ids = [1]

    mean_metrics = process_dataset(dataset_group, dataset_ids, task='single_tag', config=config)

    val_auc = mean_metrics.get("val_auc_tag", 0.0)
    train_auc = mean_metrics.get("train_auc_tag", 0.0)
    test_auc = mean_metrics.get("test_auc_tag", 0.0)

    if os.path.exists(filename):
        with open(filename, "r") as f:
            try:
                best_data = json.load(f)
                current_best_auc = best_data.get("AUC val", 0)
            except json.JSONDecodeError:
                current_best_auc = 0
    else:
        current_best_auc = 0

    if val_auc > current_best_auc:
        with open(filename, "w") as f:
            json.dump({
                "trial": trial.number,
                "params": trial.params,
                "AUC train": train_auc,
                "AUC val": val_auc,
                "AUC test": test_auc
            }, f, indent=4)

    return val_auc  # Optuna will maximize this

import optuna

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize", study_name="microbiome_classifier_opt")
    study.optimize(objective, n_trials=100)

    print("Best trial:")
    trial = study.best_trial
    print(f"Value (val_auc): {trial.value}")
    print("Params:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
