# src/models/embedding.py
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from torch.utils.data import DataLoader

import configs.config
from ..data.dataset import BacteriaDataset
from ..training.utils import calc_distance_in_sample
from torchmetrics.classification import BinaryAUROC
from ..training.utils import embedding_regularization
from torch.nn import functional as F
import wandb
from pytorch_lightning.loggers import WandbLogger

import pandas as pd
# Transformer-based Model
class TransformerPredictor(pl.LightningModule):
    def __init__(self, config: configs.config.Config, num_microbes, bacteria_names_train,
                 load_embedding=None):
        super(TransformerPredictor, self).__init__()
        device = config.DEVICE
        embedding_dim = config.EMBEDDING_DIM
        hidden_dim = config.HIDDEN_DIM_TRANSFORMER_EMBEDDING
        num_heads = config.NHEAD_EMBEDDING
        num_layers = config.NUM_LAYERS_EMBEDDING
        self.lr = config.LR_EMBEDDING
        self.weight_decay = config.WEIGHT_DECAY_EMBEDDING

        # 1) Microbe embedding => unique vector per microbe ID
        self.microbe_embedding = nn.Embedding(num_microbes, embedding_dim).to(device)
        if load_embedding is not None:
            self.microbe_embedding.weight.data = load_embedding.transpose(0, 1)

        # 2) Presence embedding => 3 possible codes: 0=absent, 1=present, 2=[MASK], -1=[PAD]
        self.presence_embedding = nn.Embedding(4, embedding_dim)

        # 3) Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers).to(device)

        # 4) Attention-based community-level aggregation
        self.attention_weights = nn.Parameter(torch.randn(embedding_dim, 1)).to(device)
        self.community_fc = nn.Linear(embedding_dim, embedding_dim).to(device)
        self.taxonomy_distance_matrix = calc_distance_in_sample(bacteria_names_train, device)
        self.community_embedding = None

        # 5) Final Linear layer- classification
        self.fc = nn.Linear(embedding_dim, 1)

        # Metrics
        self.train_auroc = BinaryAUROC()
        self.val_auroc = BinaryAUROC()

    def forward(self, existence_vector, attn_mask):
        batch_size, num_bacteria = existence_vector.shape  # [batch_size, num_microbes]

        # (A) Build a tensor of microbe IDs: shape (num_bacteria,)
        #     then expand to (batch_size, num_bacteria)
        microbe_ids = torch.arange(num_bacteria, device=existence_vector.device)
        microbe_ids = microbe_ids.unsqueeze(0).expand(batch_size,
                                                      -1)  # shape: [batch_size, num_bacteria, embedding_dim]

        # (B) Look up the microbe embedding => shape (batch_size, num_bacteria, embedding_dim)
        microbe_emb = self.microbe_embedding(microbe_ids)

        # (C) Look up presence embedding => shape (batch_size, num_bacteria, embedding_dim) (the embedding of the presence code)
        padded_input = existence_vector.clone()
        padded_input[padded_input == -1] = 3  # map -1 to index 3 (PAD) in presence_embedding
        presence_emb = self.presence_embedding(padded_input)

        # (D) Sum them => final embedding shape (batch_size, num_bacteria, embedding_dim)
        combined_emb = microbe_emb + presence_emb

        # (E) Optionally do your "community" attention
        #     Let's do a similar approach to your code, so we permute to (B, E, N)
        # attention_scores => (batch_size, num_bacteria)
        attention_scores = torch.einsum('bni,in->bn', combined_emb, self.attention_weights)
        attention_probs = F.softmax(attention_scores, dim=1).unsqueeze(-1)
        community_embedding = (combined_emb * attention_probs).sum(dim=1)
        community_embedding = F.relu(self.community_fc(community_embedding))
        # shape => (batch_size, embedding_dim)

        # If you want to incorporate community info back into each position:
        # expand => (batch_size, embedding_dim, num_bacteria)
        expanded_community = community_embedding.unsqueeze(-1).expand(-1, -1, num_bacteria)
        # add => (batch_size, embedding_dim, num_bacteria)
        combined_with_community = combined_emb.permute(0, 2, 1) + expanded_community
        # => shape (batch_size, num_bacteria, embedding_dim)
        combined_with_community = combined_with_community.permute(0, 2, 1)

        # (F) Pass through the Transformer.
        transformer_input = combined_with_community  # ( batch_size,num_bacteria, embedding_dim)
        transformer_output = self.transformer(transformer_input, src_key_padding_mask=attn_mask)  # same shape

        # (G) Final classification => shape (batch_size,num_bacteria, 1)
        predictions = torch.sigmoid(self.fc(transformer_output)).squeeze(-1)

        return predictions

    def training_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx, 'train')
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx, 'val')
        return loss

    def common_step(self, batch, batch_idx, stage):
        existence_vector = batch['existence_vector']
        y_true = batch['y_true']
        mask_vector = batch['mask_vector']
        attn_mask = batch['attn_mask']

        # Forward pass
        predictions = self(existence_vector, attn_mask)

        # Exclude NaN positions from loss
        valid_mask = mask_vector & ~attn_mask  # Only masked non-NaN positions
        if valid_mask.any():
            # Compute class weights
            num_class_0 = (y_true[valid_mask] == 0).sum().item()
            num_class_1 = (y_true[valid_mask] == 1).sum().item()
            self.log(f'{stage}_num_class_0', num_class_0, on_step=False, on_epoch=True)
            self.log(f'{stage}_num_class_1', num_class_1, on_step=False, on_epoch=True)
            sum_of_classes = num_class_0 + num_class_1

            class_weight_0 = (sum_of_classes - num_class_0) / sum_of_classes if num_class_0 > 0 else 0.0
            class_weight_1 = (sum_of_classes - num_class_1) / sum_of_classes if num_class_1 > 0 else 0.0
            # Create a weight tensor
            weights = torch.where(y_true[valid_mask] == 1, class_weight_1, class_weight_0).to(self.device)
            loss = F.binary_cross_entropy(predictions[valid_mask], y_true[valid_mask].float(), weight=weights)

            if stage == 'train':
                self.train_auroc.update(predictions[mask_vector], y_true[mask_vector])
            elif stage == 'val':
                self.val_auroc.update(predictions[mask_vector], y_true[mask_vector])

        else:
            loss = torch.tensor(0.0, device=self.device)

        loss_before_representation = embedding_regularization(self.microbe_embedding.weight.data,
                                                              self.taxonomy_distance_matrix, mode='exponent_on_h',
                                                              num_of_partition=1)
        total_loss = loss + loss_before_representation

        self.log(f'{stage}_bce_loss', loss, on_step=False, on_epoch=True)
        self.log(f'{stage}_reg_loss_before_representation', loss_before_representation, on_step=False, on_epoch=True)
        self.log(f'{stage}_total_loss', total_loss, on_step=False, on_epoch=True)
        return loss

    def on_train_epoch_end(self):
        train_auc = self.train_auroc.compute()
        self.log(f"train_auc", train_auc, on_step=False, on_epoch=True)

        true_vals = torch.cat(self.train_auroc.target)
        pred_vals = torch.cat(self.train_auroc.preds)

        mask_0 = true_vals == 0
        mask_1 = true_vals == 1

        if torch.sum(mask_0) > 0:
            accuracy_0 = (true_vals[mask_0] == (pred_vals[mask_0] > 0.5)).float().mean()
            self.log('train_accuracy_0', accuracy_0)

        if torch.sum(mask_1) > 0:
            accuracy_1 = (true_vals[mask_1] == (pred_vals[mask_1] > 0.5)).float().mean()
            self.log('train_accuracy_1', accuracy_1)

        # Precision and Recall
        pred_labels = (pred_vals > 0.5).float()
        true_positive = (pred_labels * true_vals).sum()
        predicted_positive = pred_labels.sum()
        actual_positive = true_vals.sum()

        precision = true_positive / predicted_positive if predicted_positive > 0 else torch.tensor(0.0,
                                                                                                   device=true_vals.device)
        recall = true_positive / actual_positive if actual_positive > 0 else torch.tensor(0.0, device=true_vals.device)

        self.log("train_precision", precision)
        self.log("train_recall", recall)

        self.train_auroc.reset()

    def on_validation_epoch_end(self):
        val_auc = self.val_auroc.compute()
        self.log(f"val_auc", val_auc, on_step=False, on_epoch=True)

        true_vals = torch.cat(self.val_auroc.target)
        pred_vals = torch.cat(self.val_auroc.preds)

        mask_0 = true_vals == 0
        mask_1 = true_vals == 1

        if torch.sum(mask_0) > 0:
            accuracy_0 = (true_vals[mask_0] == (pred_vals[mask_0] > 0.5)).float().mean()
            self.log('val_accuracy_0', accuracy_0)

        if torch.sum(mask_1) > 0:
            accuracy_1 = (true_vals[mask_1] == (pred_vals[mask_1] > 0.5)).float().mean()
            self.log('val_accuracy_1', accuracy_1)

            # Precision and Recall
        pred_labels = (pred_vals > 0.5).float()
        true_positive = (pred_labels * true_vals).sum()
        predicted_positive = pred_labels.sum()
        actual_positive = true_vals.sum()

        precision = true_positive / predicted_positive if predicted_positive > 0 else torch.tensor(0.0,
                                                                                                   device=true_vals.device)
        recall = true_positive / actual_positive if actual_positive > 0 else torch.tensor(0.0, device=true_vals.device)

        self.log("val_precision", precision)
        self.log("val_recall", recall)

        self.val_auroc.reset()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)


def extract_qkv(layer):
    """Extract query, key, and value weights and biases."""
    in_proj_weight = layer.self_attn.in_proj_weight  # Shape: (3*d_model, d_model)
    in_proj_bias = layer.self_attn.in_proj_bias  # Shape: (3*d_model,)
    d_model = layer.self_attn.embed_dim

    # Split into query, key, valueq
    query_weight = in_proj_weight[:d_model - 1, :-1]
    key_weight = in_proj_weight[d_model:2 * d_model - 1, :-1]
    value_weight = in_proj_weight[2 * d_model:-1, :-1]

    query_bias = in_proj_bias[:d_model - 1]
    key_bias = in_proj_bias[d_model:2 * d_model - 1]
    value_bias = in_proj_bias[2 * d_model:-1]

    in_proj_weight = torch.concat([query_weight, key_weight, value_weight], dim=0)
    in_proj_bias = torch.concat([query_bias, key_bias, value_bias], dim=0)
    return in_proj_weight, in_proj_bias


def run_shared(num_microbes, train_data, bacteria_train, name_of_dataset, dataset_ids,
               config: configs.config.Config):
    # wandb_name_project= f"{config.WANDB_NAME_EMBEDDING}_{name_of_dataset}_{dataset_ids}_{config.FLAG_TYPE_RUN}"
    # wandb.init(project=wandb_name_project, name=f"{name_of_dataset}_{dataset_ids}", config=config.to_dict())
    # wandb.run.log_code(".")
    # wandb_logger = WandbLogger(project=config.WANDB_NAME_EMBEDDING, name=f"{name_of_dataset}_{dataset_ids}",
    #                            id=f"{name_of_dataset}_{dataset_ids}")

    train_data = torch.where(torch.isnan(train_data), train_data, (train_data != -1).int()).to(config.DEVICE)
    split_ratio = 0.8
    train_size = int(split_ratio * train_data.shape[0])
    train_indices, val_indices = torch.utils.data.random_split(
        list(range(train_data.shape[0])), [train_size, train_data.shape[0] - train_size]
    )
    train_dataset = BacteriaDataset(train_data[train_indices])
    val_dataset = BacteriaDataset(train_data[val_indices])
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    model = TransformerPredictor(config,
                                 num_microbes, bacteria_names_train=bacteria_train
                                 ).to(config.DEVICE)
    early_stopping_callback = EarlyStopping(
        monitor='val_total_loss', patience=config.EARLY_STOPPING_PATIENCE_EMBEDDING, mode='min', verbose=True, min_delta=0.001
    )
    trainer = pl.Trainer(
        max_epochs=config.MAX_EPOCHS_EMBEDDING, accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1, num_sanity_val_steps=0, logger=[], callbacks=[early_stopping_callback]
    )
    trainer.fit(model, train_loader, val_loader)
    # wandb.finish()
    # torch.save(model.microbe_embedding.weight.data.clone(), f"embedding_8_dim_trained_embeddings_{name_of_dataset}_{dataset_ids}.pt")
    # pd.to_pickle(bacteria_train, f"embedding_8_bacteria_train_{name_of_dataset}_{dataset_ids}.pkl")
    return model.microbe_embedding.weight.data.clone().to(config.DEVICE)
