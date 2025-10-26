# src/data/loader.py
import os
import torch
from .dataset import MicrobiomeDataset
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset


def load_preprocessed_data(name_of_dataset, dataset_ids, task, config):
    path = config.DATA_DIR
    prefix = f"{name_of_dataset}_{'_'.join(map(str, dataset_ids))}"
    files = [
        f"{prefix}_bacteria_names.pt", f"{prefix}_train_data.pt", f"{prefix}_test_data.pt",
        f"{prefix}_original_train_tag.pt", f"{prefix}_test_tag.pt",
        f"{prefix}_train_which_dataset_col.pt", f"{prefix}_test_which_dataset_col.pt",
        f"{prefix}_num_of_partition.pt", f"{prefix}_bact_contribution.pt",
        f"{prefix}_train_data_recognize_padded_bact_in_each_dataset.pt",
        f"{prefix}_test_data_recognize_padded_bact_in_each_dataset.pt"
    ]
    data = [torch.load(os.path.join(path, f)) for f in files]
    bacteria_names, train_data, test_data, train_tag, test_tag, train_which_dataset_col, \
    test_which_dataset_col, num_of_partition, bact_contribution, train_padded, test_padded = data
    return (
        bacteria_names, train_data.to(config.DEVICE), test_data.to(config.DEVICE),
        train_tag.to(config.DEVICE), test_tag.to(config.DEVICE),
        train_which_dataset_col.to(config.DEVICE), test_which_dataset_col.to(config.DEVICE),
        num_of_partition, bact_contribution, train_padded.to(config.DEVICE), test_padded.to(config.DEVICE)
    )

def create_balanced_sampler(which_dataset):
    # Calculate the class counts on the GPU
    unique_classes, class_counts = torch.unique(which_dataset, return_counts=True)

    # Compute weights for each class as the inverse of class counts
    class_weights = 1.0 / class_counts.float()

    # Map weights to each sample in `which_dataset`
    sample_weights = class_weights[which_dataset]

    # Create a WeightedRandomSampler using the calculated sample weights
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(which_dataset), replacement=True)
    return sampler

def prepare_loaders(train_data, train_tag, train_which_dataset_col, train_indices, val_indices, config):

    train_which_dataset = train_which_dataset_col[train_indices] if train_which_dataset_col is not None else None
    val_which_dataset = train_which_dataset_col[val_indices] if train_which_dataset_col is not None else None

    train_dataset = MicrobiomeDataset(
        train_data[train_indices], train_tag[train_indices], which_dataset=train_which_dataset, device=config.DEVICE
    )
    val_dataset = MicrobiomeDataset(
        train_data[val_indices], train_tag[val_indices], which_dataset=val_which_dataset, device=config.DEVICE
    )

    if train_which_dataset_col is not None and len(train_which_dataset_col.unique()) > 1:
        train_sampler = create_balanced_sampler(train_which_dataset)
        train_loader = DataLoader(
            train_dataset, batch_size=config.BATCH_SIZE, sampler=train_sampler
        )
    else:
        train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    return train_loader, val_loader

