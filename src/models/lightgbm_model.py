import re
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split, KFold, GroupShuffleSplit, StratifiedKFold
from sklearn.metrics import roc_auc_score
from src.data.preprocess import preprocess_dataframe

def run_lightgbm(dataset_name, dataset_id):
    dataset_id = dataset_id[0]
    data_paths = [f"/home/eng/finkels9/transformer_update/Data/{dataset_name}/{dataset_id}/for_preprocess.csv"]
    tag_paths = [f"/home/eng/finkels9/transformer_update/Data/{dataset_name}/{dataset_id}/tag.csv"]
    cols_name= ['Tag']

    tag_df = pd.read_csv(tag_paths[0], index_col=0, low_memory=False)
    tag_df = pd.concat([tag_df[cols_name], tag_df['Group']], axis=1)
    tag_df.columns = [f'Tag_{i}' for i in range(len(cols_name))] + ['Group']

    df = pd.read_csv(data_paths[0], low_memory=False)
    df = preprocess_dataframe(df)

    tag_df.index = tag_df.index.astype(str)
    df.index = df.index.astype(str)
    common_index = df.index.intersection(tag_df.index)
    df, tag_df = df.loc[common_index], tag_df.loc[common_index]
    tag_df = tag_df[~tag_df.index.duplicated(keep='first')]
    groups = tag_df['Group'].values
    y= tag_df['Tag_0'].values
    df.columns = df.columns.str.replace('[^A-Za-z0-9_]', '_', regex=True)

    splitter = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=42)
    train_idx, test_idx = next(splitter.split(X=range(len(groups)),y=y, groups=groups))

    X_train, X_test = df.iloc[train_idx], df.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    # Split data into training and test sets

    # Initialize KFold
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    # Lists to store AUC scores
    train_auc_scores = []
    val_auc_scores = []
    test_auc_scores = []

    # LightGBM parameters
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'max_depth': -1,
        'n_estimators': 1000,
        'verbose': -1,
        'random_state': 42
    }

    # Perform 10-fold cross-validation
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
        print(f"Training fold {fold + 1}...")

        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]

        model = lgb.LGBMClassifier(**params)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], eval_metric='auc',callbacks=[lgb.early_stopping(10)])

        # Predict probabilities
        y_tr_pred = model.predict_proba(X_tr)[:, 1]
        y_val_pred = model.predict_proba(X_val)[:, 1]
        y_test_pred = model.predict_proba(X_test)[:, 1]

        # Compute AUC
        train_auc = roc_auc_score(y_tr, y_tr_pred)
        val_auc = roc_auc_score(y_val, y_val_pred)
        test_auc = roc_auc_score(y_test, y_test_pred)

        train_auc_scores.append(train_auc)
        val_auc_scores.append(val_auc)
        test_auc_scores.append(test_auc)

        # print(f"Fold {fold + 1} - Train AUC: {train_auc:.4f}, Val AUC: {val_auc:.4f}, Test AUC: {test_auc:.4f}")

    # Compute the average AUC scores
    train_auc_mean = np.mean(train_auc_scores)
    val_auc_mean = np.mean(val_auc_scores)
    test_auc_mean = np.mean(test_auc_scores)

    print(f"\nDataset {dataset_name}, ID: {dataset_id}, Average AUC Scores:")
    print(f"Train AUC: {train_auc_mean:.4f}")
    print(f"Validation AUC: {val_auc_mean:.4f}")
    print(f"Test AUC: {test_auc_mean:.4f}")

def main():
    datasets = {
        # 'GC': [1, 2, 3, 4, 5, 6, 7]
        # 'Parkinson_WGS': [1,2,3,4,5,6]
        'Parkinson_16S': [1, 2, 3, 4, 5, 6, 7, 8],

    }
    for dataset_group, dataset_ids in datasets.items():
        for dataset_id in dataset_ids:
            run_lightgbm(dataset_group, [dataset_id])

if __name__ == "__main__":
    main()