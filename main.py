# main.py
import torch
#HEREREEEEEE TODO
from configs.config import Config
import sys

from src.utils.common import set_seed
from src.data.loader import load_preprocessed_data, prepare_loaders
from src.models.embedding import run_shared
from src.models.classifier import MicrobiomeClassifier
from src.training.trainer import train_classifier
from src.data.dataset import MicrobiomeDataset
from src.models.score_model import  MicrobeModel
from src.data.utils import load_data, stratified_by_task
import os
import numpy as np
import warnings
warnings.filterwarnings("ignore")


def process_dataset(name_of_dataset, dataset_ids, task, config, load_preprocessed_data_flag=False,
                    path_of_dataset_to_lodo=None, wandb_logger=None):
    set_seed(config.SEED)
    print(f"Processing: {name_of_dataset}, Dataset IDs: {dataset_ids}, Task: {task}")
    if path_of_dataset_to_lodo is not None:
        print(f"LODO IS {path_of_dataset_to_lodo}")

    if load_preprocessed_data_flag:
        data = load_preprocessed_data(name_of_dataset, dataset_ids, task, config)
    else:
        data = load_data(name_of_dataset, dataset_ids, task, config, lodo_path=path_of_dataset_to_lodo)
    bacteria_names, train_data, test_data, train_tag, test_tag, train_which_dataset_col, \
        test_which_dataset_col, num_of_partition, train_padded, test_padded = data

    bacteria_names_train, bacteria_names_test = bacteria_names

    # Pre-train embeddings
    if config.RANDOM_INITIALIZE_EMBEDDINGS:
        train_embedding = None
    else:
        train_embedding = run_shared(
            len(bacteria_names_train), train_padded, bacteria_names_train,
            name_of_dataset, dataset_ids, config
        )

        # train_embedding = torch.load(f"embedding_10_dim_trained_embeddings_{name_of_dataset}_{dataset_ids}.pt").to(config.DEVICE)

    # Stratified split
    skf, partition_tag_train, train_which_dataset_col = stratified_by_task(task, train_tag,
                                                                           train_which_dataset_col, config)

    train_which_dataset_col_original = train_which_dataset_col.unsqueeze(1).to(config.DEVICE)
    test_which_dataset_col_original = test_which_dataset_col.unsqueeze(1).to(config.DEVICE)
    train_data = train_data.to(config.DEVICE)
    train_data_with_which_dataset_col = torch.cat([train_data, train_which_dataset_col_original], dim=1)
    test_data_with_which_dataset_col = torch.cat([test_data, test_which_dataset_col_original], dim=1)

    if path_of_dataset_to_lodo is None:
        for dataset_id in range(len(dataset_ids)):
            metrics_over_folds = {}
            print(f"Training on Dataset ID: {dataset_ids[dataset_id]}")
            dataset_id_tensor = torch.tensor(dataset_id, dtype=torch.int64, device=config.DEVICE)

            # Create boolean mask for the current dataset ID- to get the subset of the data
            train_mask = torch.isin(train_which_dataset_col.squeeze(), dataset_id_tensor)
            test_mask = torch.isin(test_which_dataset_col.squeeze(), dataset_id_tensor)

            # Apply the mask to filter the rows
            train_data_subset = train_data_with_which_dataset_col[train_mask]
            test_data_subset = test_data_with_which_dataset_col[test_mask]
            train_tag_subset = train_tag[train_mask.to('cpu')]
            test_tag_subset = test_tag[test_mask]
            train_which_dataset_col_subset = train_which_dataset_col[train_mask]
            test_which_dataset_col_subset = test_which_dataset_col[test_mask]

            train_data_subset = train_data_subset[:, :-1]
            test_data_subset = test_data_subset[:, :-1]
            all_subset_data = torch.cat([train_data_subset, test_data_subset], dim=0).to(config.DEVICE)

            # Padded columns are padded with zeros- this way we can filter them out
            columns_to_keep = torch.any(all_subset_data != 0, dim=0)
            train_data_subset = train_data_subset[:, columns_to_keep].to('cpu')
            test_data_subset = test_data_subset[:, columns_to_keep]
            train_embedding_subset = train_embedding[columns_to_keep] if train_embedding is not None else None
            valid_columns_indices = torch.where(columns_to_keep)[0].tolist()
            bacteria_names_train_subset = [bacteria_names_train[i] for i in valid_columns_indices]
            num_microbes = len(bacteria_names_train_subset)
            partition_tag_train_subset = partition_tag_train[train_mask.to('cpu')].to('cpu')

            # Prepare test loader
            test_dataset = MicrobiomeDataset(test_data_subset, test_tag_subset, test_which_dataset_col_subset,
                                             config.DEVICE)
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)


            # Training loop
            for fold, (train_idx, val_idx) in enumerate(skf.split(train_data_subset, partition_tag_train_subset)):

                train_loader, val_loader = prepare_loaders(train_data_subset, train_tag_subset,
                                                           train_which_dataset_col_subset, train_idx, val_idx,
                                                           config)

                if config.MODEL_TYPE != 'score_model':
                    model = MicrobiomeClassifier(
                        num_microbes_train=len(bacteria_names_train_subset),
                        bacteria_names_train=bacteria_names_train_subset, bacteria_names_test=bacteria_names_test,
                        config=config, num_of_partition=num_of_partition,
                        lodo_flag=True if path_of_dataset_to_lodo is not None else False,
                        pretrained_embedding=train_embedding_subset
                    ).to(config.DEVICE)
                else:
                    print('score model')
                    model= MicrobeModel(num_microbes_train=len(bacteria_names_train_subset),
                                        num_samples=len(train_idx),config=config,
                                        pretrained_embedding=train_embedding_subset).to(config.DEVICE)

                run_id = f"run_{fold}_{name_of_dataset}"
                fold_metrics = train_classifier(model, train_loader, val_loader, test_loader, config, name_of_dataset,
                                                dataset_ids[dataset_id], fold, run_id, wandb_logger)

                # Store fold metrics
                for key, value in fold_metrics.items():
                    if key not in metrics_over_folds:
                        metrics_over_folds[key] = []  # Initialize list for each metric
                    metrics_over_folds[key].append(value)

                # embedding= model.microbe_embedding.weight.cpu().detach().numpy()
                # np.save(f"embedding_after_transformer__{name_of_dataset}_{dataset_ids[dataset_id]}_fold_{fold}.npy", embedding)
                # np.save(f"bacteria_names_train_{name_of_dataset}_{dataset_ids[dataset_id]}.npy", bacteria_names_train)
                print(f"Fold {fold + 1}: {fold_metrics}")

            # Compute mean of each metric across folds
            mean_metrics = {key: np.mean(values) for key, values in metrics_over_folds.items()}

            print(f"Dataset {dataset_ids[dataset_id]} DONE. \nMean Metrics Over All 10 Folds:\n", mean_metrics)

    else:
            # Leave one dataset out
            metrics_over_folds = {}
            train_data = train_data_with_which_dataset_col[:, :-1]
            test_data = test_data_with_which_dataset_col[:, :-1]

            # Prepare test loader
            test_dataset = MicrobiomeDataset(test_data, test_tag, test_data_with_which_dataset_col,
                                             config.DEVICE)
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

            # Training loop
            for fold, (train_idx, val_idx) in enumerate(skf.split(train_data, partition_tag_train)):

                train_loader, val_loader = prepare_loaders(train_data, train_tag,
                                                           train_which_dataset_col, train_idx, val_idx,
                                                           config)

                model = MicrobiomeClassifier(
                    num_microbes_train=len(bacteria_names_train),
                    bacteria_names_train=bacteria_names_train, bacteria_names_test=bacteria_names_test, config=config,
                    pretrained_embedding=train_embedding, num_of_partition=num_of_partition,
                    lodo_flag=True if path_of_dataset_to_lodo is not None else False
                ).to(config.DEVICE)

                run_id = f"run_{fold}_{name_of_dataset}"
                fold_metrics = train_classifier(model, train_loader, val_loader, test_loader, config, name_of_dataset,
                                                path_of_dataset_to_lodo.split('/')[-1], fold, run_id)

                # Store fold metrics
                for key, value in fold_metrics.items():
                    if key not in metrics_over_folds:
                        metrics_over_folds[key] = []
                    metrics_over_folds[key].append(value)

                print(f"Fold {fold + 1}: {fold_metrics}")

            # Compute mean of each metric across folds
            mean_metrics = {key: np.mean(values) for key, values in metrics_over_folds.items()}
            print(f"Dataset {path_of_dataset_to_lodo.split('/')[-1]} DONE. \nMean Metrics Over All 10 Folds:\n",
                  mean_metrics)

    return mean_metrics




def run_simulation():

    config = Config()
    load_preprocessed_data_flag = False
    random_initialize_embeddings = False
    # path_of_dataset_to_lodo = '/home/eng/finkels9/transformer_update/Data/Parkinson_16S/1'
    path_of_dataset_to_lodo = None
    flag_type_run = 'single_tag'
    # flag_type_run = 'which_dataset'


    config.FLAG_TYPE_RUN = flag_type_run
    config.RANDOM_INITIALIZE_EMBEDDINGS = random_initialize_embeddings
    config.path_of_dataset_to_lodo = path_of_dataset_to_lodo
    config.WANDB_NAME_CLASSIFIER = 'check_LODO_CLASSIFIER_one_linear_classifier' if path_of_dataset_to_lodo is not None else 'CLASSIFIER'
    datasets = {
        # 'GC':  [1,2, 4,5,6,7],

        # 'Parkinson_WGS': [1, 2, 3, 4,5,6],
        # 'Parkinson_16S':  [2,3,4, 5, 6,7,8],
        'Parkinson_16S':  [7],
        # 'gdm_saliva_t1':[1],
        # 'Sapir': [1]
    }

    # Case 1: Run each dataset as is, without shared information
    if flag_type_run == 'single_tag':
        for dataset_group, dataset_ids in datasets.items():
            for dataset_id in dataset_ids:
                process_dataset(dataset_group, [dataset_id], task='single_tag', config=config,
                                load_preprocessed_data_flag=load_preprocessed_data_flag)

    # Case 2: Run all datasets together in order to capture shared information
    if flag_type_run == 'which_dataset':
        process_dataset('Parkinson_16S', datasets['Parkinson_16S'], task='which_dataset', config=config,
                        load_preprocessed_data_flag=load_preprocessed_data_flag,
                        path_of_dataset_to_lodo=path_of_dataset_to_lodo)

if __name__ == "__main__":
    run_simulation()