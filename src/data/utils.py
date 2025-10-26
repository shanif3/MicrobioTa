# src/data/utils.py
import os

import numpy as np
import torch
import configs.config
import pandas as pd
from collections import Counter, defaultdict
from sklearn.model_selection import GroupShuffleSplit
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from src.data.preprocess import preprocess_dataframe



def load_data_which_dataset(data_paths, tag_paths, task: str, config: configs.config.Config, lodo_path=None):
    """Load and preprocess data for the 'which_dataset' task."""
    assert len(data_paths) > 1, f"For task '{task}', more than one dataset is required."
    assert len(tag_paths) == len(data_paths), f"For task '{task}', tag paths must match data paths."
    assert len(config.TARGET_COL_NAME) == 1, f"For task '{task}', only one target column is allowed."

    # Check if LODO path exists
    if lodo_path is not None:
        if not os.path.isdir(lodo_path):
            raise FileNotFoundError(f"LODO dataset path not found: {lodo_path}")

    train_data_list, test_data_list = [], []
    train_tag_list, test_tag_list = [], []
    for idx, (path, tag_path) in enumerate(zip(data_paths, tag_paths)):
        df = pd.read_csv(path, low_memory=False)
        tag = pd.read_csv(tag_path, index_col=0, low_memory=False).dropna()
        tag = pd.concat([tag[config.TARGET_COL_NAME], tag['Group']], axis=1)
        tag.index = tag.index.astype(str)
        df.index = df.index.astype(str)
        df = preprocess_dataframe(df)
        df['which_dataset'] = idx  # Add dataset index
        common_index = df.index.intersection(tag.index)
        df, tag = df.loc[common_index], tag.loc[common_index]
        groups = tag['Group'].values

        # Split the data and tag into train, val, and test sets by the group column
        if lodo_path is None:
            splitter = GroupShuffleSplit(test_size=0.2, n_splits=1)
            train_idx, test_idx = next(splitter.split(X=range(len(groups)), groups=groups))
            train_data_list.append(df.iloc[train_idx])
            test_data_list.append(df.iloc[test_idx])
            train_tag_list.append(tag.iloc[train_idx])
            test_tag_list.append(tag.iloc[test_idx])
        else:
            train_data_list.append(df)
            train_tag_list.append(tag)

    if lodo_path is not None:
        test_data_path = os.path.join(lodo_path, 'for_preprocess.csv')
        test_tag_path = os.path.join(lodo_path, 'tag.csv')
        if not os.path.exists(test_data_path) or not os.path.exists(test_tag_path):
            raise FileNotFoundError(f"LODO path: {lodo_path}, should consist for_preprocess.csv and tag.csv files.")

        test_df = pd.read_csv(test_data_path, low_memory=False)
        test_tag = pd.read_csv(test_tag_path, index_col=0, low_memory=False)
        test_df = preprocess_dataframe(test_df)

        test_df['which_dataset'] = idx + 1
        common_index = test_df.index.intersection(test_tag.index)
        test_df, test_tag = test_df.loc[common_index], test_tag.loc[common_index]
        test_data_list.append(test_df)
        test_tag_list.append(test_tag)

    return _finalize_data(train_data_list, test_data_list, train_tag_list, test_tag_list, config)


def _finalize_data(train_data_list, test_data_list, train_tag_list, test_tag_list, config: configs.config.Config):
    """Finalize data preprocessing and conversion to tensors."""
    for i, data in enumerate(train_data_list):
        new_cols = [
            col if col == 'which_dataset' else (
                'Bacteria;'+ col.split(';')[-1] + ';XXX;XXX' if len(col.split(';')) == 6 else
                'Bacteria;'+';'.join(col.split(';')[-2:]) + ';XXX' if len(col.split(';')) == 7 else
                'Bacteria;'+';'.join(col.split(';')[-3:]) if len(col.split(';')) == 8 else
                'Bacteria'+';'.join(col.split(';')[-2:])
            )
            for col in data.columns
        ]

        data.columns = new_cols
        # drop duplicates cols
        data = data.loc[:, ~data.columns.duplicated()]
        data = data.groupby(data.columns, axis=1).sum()
        train_data_list[i] = data



        # train_data_list[i].columns = [';'.join(col.split(';')[-2:]) for col in data.columns]

    for i, data in enumerate(test_data_list):
        # test_data_list[i].columns = [';'.join(col.split(';')[-2:]) for col in data.columns]
        new_cols = [
            col if col == 'which_dataset' else (
                'Bacteria;' + col.split(';')[-1] + ';XXX;XXX' if len(col.split(';')) == 6 else
                'Bacteria;' + ';'.join(col.split(';')[-2:]) + ';XXX' if len(col.split(';')) == 7 else
                'Bacteria;' + ';'.join(col.split(';')[-3:]) if len(col.split(';')) == 8 else
                'Bacteria' + ';'.join(col.split(';')[-2:])
            )
            for col in data.columns
            ]

        data.columns = new_cols
        data = data.loc[:, ~data.columns.duplicated()]
        data = data.groupby(data.columns, axis=1).sum()
        test_data_list[i]= data


    train_data = pd.concat(train_data_list, axis=0, ignore_index=True)
    test_data = pd.concat(test_data_list, axis=0, ignore_index=True)
    train_tag = pd.concat(train_tag_list, axis=0, ignore_index=True).drop(columns=['Group'])
    test_tag = pd.concat(test_tag_list, axis=0, ignore_index=True).drop(columns=['Group'])

    train_padded = train_data.copy()
    test_padded = test_data.copy()
    train_data, test_data = train_data.fillna(0), test_data.fillna(0)

    #make sure 'which_dataset' is the last column
    train_data_which_dataset = train_data['which_dataset']
    train_data = train_data.drop(columns=['which_dataset'])
    test_data_which_dataset = test_data['which_dataset']
    test_data = test_data.drop(columns=['which_dataset'])
    train_data= pd.concat([train_data,train_data_which_dataset], axis=1)
    test_data= pd.concat([test_data,test_data_which_dataset], axis=1)



    bacteria_names_train = train_data.columns.tolist()[:-1]
    bacteria_names_test = test_data.columns.tolist()[:-1]
    train_which_dataset = torch.tensor(train_data['which_dataset'].values, dtype=torch.int)
    test_which_dataset = torch.tensor(test_data['which_dataset'].values, dtype=torch.int, device=config.DEVICE)

    train_data = torch.tensor(train_data.drop(columns=['which_dataset']).values, dtype=torch.float32)
    test_data = torch.tensor(test_data.drop(columns=['which_dataset']).values, dtype=torch.float32,
                             device=config.DEVICE)
    train_padded = torch.tensor(train_padded.drop(columns=['which_dataset']).values, dtype=torch.float32)
    test_padded = torch.tensor(test_padded.drop(columns=['which_dataset']).values, dtype=torch.float32,
                               device=config.DEVICE)
    train_tag = torch.tensor(train_tag.values, dtype=torch.float32)
    test_tag = torch.tensor(test_tag.values, dtype=torch.float32, device=config.DEVICE)

    num_of_partition = len(train_data_list)

    return (
        (bacteria_names_train, bacteria_names_test), train_data, test_data, train_tag, test_tag,
        train_which_dataset, test_which_dataset, num_of_partition, train_padded, test_padded
    )

def custom_column_groupby(df):
    # Group column names
    grouped = {}
    for col in df.columns:
        grouped.setdefault(col, []).append(col)

    new_cols = {}
    for col, col_list in grouped.items():
        if len(col_list) == 1:
            new_cols[col] = df[col_list[0]]
        else:
            # Stack the columns to compare row-wise
            stacked = df[col_list].T
            # For each row, check if all values are equal
            all_equal = stacked.nunique() == 1
            # Use one of the values if all equal, else sum
            values = np.where(
                all_equal,
                stacked.iloc[0],
                stacked.sum()
            )
            new_cols[col] = pd.Series(values, index=df.index)

    return pd.DataFrame(new_cols)
def load_data_single_and_mixed(data_paths: list, tag_paths: list, task: str, config: configs.config.Config):
    """Load and preprocess data for 'single_tag' or 'mixed_tag' tasks."""

    cols_name = config.TARGET_COL_NAME
    assert len(data_paths) == 1, f"For task '{task}', only one dataset is required."
    assert len(tag_paths) == 1, f"For task '{task}', only one tag path is required."
    if task == 'single_tag':
        assert len(cols_name) == 1, f"For task '{task}', only one target column is required."
    elif task == 'mixed_tag':
        assert len(cols_name) > 1, f"For task '{task}', more than one target column is required."

    tag_df = pd.read_csv(tag_paths[0], index_col=0, low_memory=False)
    tag_df = pd.concat([tag_df[cols_name], tag_df['Group']], axis=1)
    tag_df.columns = [f'Tag_{i}' for i in range(len(cols_name))] + ['Group']

    df = pd.read_csv(data_paths[0], low_memory=False)
    df = preprocess_dataframe(df)
    df['which_dataset'] = 0

    tag_df.index = tag_df.index.astype(str)
    df.index = df.index.astype(str)
    common_index = df.index.intersection(tag_df.index)
    df, tag_df = df.loc[common_index], tag_df.loc[common_index]
    tag_df = tag_df[~tag_df.index.duplicated(keep='first')]
    tags = tag_df.filter(like='Tag').values
    groups = tag_df['Group'].values

    splitter = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=42)
    train_idx, test_idx = next(splitter.split(X=range(len(groups)), y=tags, groups=groups))

    bacteria_names = df.columns.tolist()[:-1]
    train_data = torch.tensor(df.drop(columns=['which_dataset']).values[train_idx], dtype=torch.float32)
    test_data = torch.tensor(df.drop(columns=['which_dataset']).values[test_idx], dtype=torch.float32,
                             device=config.DEVICE)
    train_tag = torch.tensor(tags[train_idx], dtype=torch.float32)
    test_tag = torch.tensor(tags[test_idx], dtype=torch.float32, device=config.DEVICE)
    train_which_dataset = torch.tensor(df.values[train_idx, -1], dtype=torch.int)
    test_which_dataset = torch.tensor(df.values[test_idx, -1], dtype=torch.int, device=config.DEVICE)

    num_of_partition = len(cols_name)

    return (
        bacteria_names,
        bacteria_names), train_data, test_data, train_tag, test_tag, train_which_dataset, test_which_dataset, num_of_partition, train_data, test_data


def load_data(name_of_dataset, dataset_ids, task, config: configs.config.Config, lodo_path=None):
    """Dispatch data loading based on task type."""
    data_paths, tag_paths = load_dataset(name_of_dataset, dataset_ids, config.DEVICE)
    if task in ['single_tag', 'mixed_tag']:
        return load_data_single_and_mixed(data_paths, tag_paths, task, config)
    elif task == 'which_dataset':
        return load_data_which_dataset(data_paths, tag_paths, task, config, lodo_path)


def stratified_by_task(task, original_train_tag, train_which_dataset_col, config: configs.config.Config):
    """Create stratified k-fold splits based on task."""
    if task == 'mixed_tag':
        skf = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=config.SEED)
        partition_tag = original_train_tag
    elif task in ['which_dataset', 'single_tag']:
        skf = MultilabelStratifiedKFold(n_splits=10, shuffle=True, random_state=config.SEED)
        partition_tag = torch.cat((original_train_tag, train_which_dataset_col.unsqueeze(1)), dim=1)
        train_which_dataset_col = train_which_dataset_col.to(config.DEVICE)
    else:
        raise ValueError(f"Unsupported task: {task}")
    return skf, partition_tag, train_which_dataset_col


def load_dataset(name_of_dataset, num_data, device):
    """Load file paths for a given dataset."""
    base_dir = configs.config.Config.DATA_DIR  # Hardcoded for now; consider config
    datasets = {
        'GC': lambda n: (
        [os.path.join(base_dir, 'Data', 'GC', f'{i}', 'for_preprocess.csv') for i in n],
        [os.path.join(base_dir, 'Data', 'GC', f'{i}', 'tag.csv') for i in n]),

        'Parkinson_WGS': lambda n:(
        [os.path.join(base_dir, 'Data', 'Parkinson_WGS', f"{i}", 'for_preprocess.csv') for i in n],
        [os.path.join(base_dir, 'Data', 'Parkinson_WGS', f"{i}", 'tag.csv') for i in n]),

        'Parkinson_16S': lambda n: (
        [os.path.join(base_dir, 'Data', 'Parkinson_16S', f"{i}", 'for_preprocess.csv') for i in n],
        [os.path.join(base_dir, 'Data', 'Parkinson_16S', f"{i}", 'tag.csv') for i in n]),

        'gdm_saliva_t1': lambda n: (
        [os.path.join(base_dir, 'Data', 'gdm_saliva_t1', f"{i}", 'for_preprocess.csv') for i in n],
        [os.path.join(base_dir, 'Data', 'gdm_saliva_t1', f"{i}", 'tag.csv') for i in n]),

        'Sapir': lambda n: (
            [os.path.join(base_dir, 'Data', 'Sapir', f"{i}", 'for_preprocess.csv') for i in n],
            [os.path.join(base_dir, 'Data', 'Sapir', f"{i}", 'tag.csv') for i in n]),





        'mixed_data': lambda n: ([os.path.join(base_dir, 'Data', 'mix_data', f'{i}', 'for_preprocess.csv')
                                  for i in n],
                                 [os.path.join(base_dir, 'Data', 'mix_data', f'{i}', 'tag.csv')
                                  for i in n]),

        'ben_goriun': lambda _: (
            [os.path.join(base_dir, 'Data', 'ben_goriun', 'less_nan', 'less_nans_otu_before_MIPMLP.csv')],
            [os.path.join(base_dir, 'Data', 'ben_goriun', 'less_nan', 'less_nans_tags_before_MIPMLP.csv')]),

        'IBD': lambda _: ([os.path.join(base_dir, 'Data', 'IBD', f"{i}", 'for_preprocess.csv')
                           for i in range(1, 5)],
                          [os.path.join(base_dir, 'Data', 'IBD', f"{i}", 'tag.csv')
                           for i in range(1, 5)]),

    }
    return datasets.get(name_of_dataset, lambda x: ([], []))(num_data)


def get_padded_columns_for_each_dataset(train_data, test_data, train_which_dataset_col, test_which_dataset_col):
    """Identify valid columns per dataset for padding."""
    data = pd.concat([train_data, test_data])
    which_dataset_col = pd.concat([train_which_dataset_col, test_which_dataset_col])
    valid_columns_per_dataset = defaultdict(list)
    for dataset_num in which_dataset_col.unique():
        specific_dataset = data[which_dataset_col == dataset_num]
        unique_counts = specific_dataset.nunique()
        valid_columns = unique_counts[unique_counts > 1].index.tolist()
        valid_column_indexes = [data.columns.get_loc(col) for col in valid_columns]
        valid_columns_per_dataset[dataset_num] = valid_column_indexes
    return valid_columns_per_dataset


