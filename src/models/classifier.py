import torch
import torch.nn as nn
import pytorch_lightning as pl
import inspect

import configs.config_lodo
from .components import MultiHeadTransformerAggregator
from ..training.utils import embedding_regularization, adjust_embedding_for_test_lodo, calc_distance_in_sample
from torchmetrics.classification import BinaryAUROC
from torch.nn import functional as F


class MicrobiomeClassifier(pl.LightningModule):
    def __init__(
            self,
            num_microbes_train,
            bacteria_names_train, bacteria_names_test,
            config: configs.config.Config, num_of_partition, lodo_flag=False,
            pretrained_embedding=None,  # Could be an nn.Embedding or a raw tensor
    ):
        """
        :param num_microbes_train: number of distinct microbes
        :param embedding_dim: dimension of each microbe embedding
        :param pretrained_embedding:
            if an nn.Embedding, we use that directly.
            if a Tensor, we wrap it in nn.Embedding.from_pretrained(...).
            if None, we create a new random embedding.
        :param freeze_embedding: whether to freeze the microbe embedding
        :param aggregator_dropout: dropout in the aggregator
        :param hidden_dim: MLP hidden dimension for final classification
        :param lr: learning rate
        :param weight_decay: weight decay for optimizer
        :param label_smoothing: how much label smoothing to apply
        :param taxonomy_distance_matrix: optional distance matrix for the reg
        :param reg_mode: used in embedding_regularization
        :param reg_lambda: coefficient for the embedding reg loss
        """
        super().__init__()
        # 1) Set up microbe embedding
        if isinstance(pretrained_embedding, nn.Embedding):
            self.microbe_embedding = pretrained_embedding
        elif isinstance(pretrained_embedding, torch.Tensor):
            self.microbe_embedding = nn.Embedding.from_pretrained(
                pretrained_embedding,
                freeze=config.FREEZE_EMBEDDING_CLASSIFIER  # do we want to FREEZE THE EMBEDDING?
            )
        else:
            # initialize new embedding if none is provided to load
            self.microbe_embedding = nn.Embedding(num_microbes_train, config.EMBEDDING_DIM)

        self.num_bacteria_train = num_microbes_train
        self.bacteria_names_train = bacteria_names_train
        self.bacteria_names_test = bacteria_names_test

        # If we want to freeze AFTER we wrap it in from_pretrained
        if config.FREEZE_EMBEDDING_CLASSIFIER:
            for param in self.microbe_embedding.parameters():
                param.requires_grad = False

        # 2) Attention-based aggregator
        self.aggregator = MultiHeadTransformerAggregator(config)
        self.generator = torch.Generator().manual_seed(config.SEED)
        # # 3) Classifier head => MLP
        self.lodo_flag = lodo_flag
        self.num_of_partition = num_of_partition
        # if not lodo_flag:
        self.classifier = nn.Sequential(
            nn.Linear(config.EMBEDDING_DIM, config.HIDDEN_DIM_CLASSIFIER1),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT_LINEAR_CLASSIFIER),
            nn.Linear(config.HIDDEN_DIM_CLASSIFIER1, config.HIDDEN_DIM_CLASSIFIER2),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT_LINEAR_CLASSIFIER),
            nn.Linear(config.HIDDEN_DIM_CLASSIFIER2, 1)  # final logit
        )
        # 4) Hyperparams
        self.lr = config.LR_CLASSIFIER
        self.weight_decay = config.WEIGHT_DECAY_CLASSIFIER
        self.label_smoothing = config.LABEL_SMOOTHING


        # 5) Regularization
        self.taxonomy_distance_matrix = calc_distance_in_sample(bacteria_names_train, config.DEVICE).to('cuda')
        self.reg_mode = config.REG_MODE
        self.reg_lambda = config.REG_LAMBDA

        # 6) Metrics to evaluate
        self.train_auroc = BinaryAUROC()
        self.val_auroc = BinaryAUROC()
        self.test_auroc = BinaryAUROC()

        self.flag_type_run = config.FLAG_TYPE_RUN
    def forward(self, abundances, att_mask, mode):
        """
        :param abundances: (batch_size, num_microbes) with real-valued abundances
        This function:
          1) Creates microbe IDs => 0..(num_microbes-1)
          2) Looks up microbe embeddings
          3) Weigh them by 'abundance' in a learned manner (here we do simple multiplication or something more advanced).
          4) Aggregates them with attention
          5) Feeds them into the classifier
        """

        batch_size, num_bacteria = abundances.size()
        device = abundances.device

        # if self.lodo_flag and mode=='test':
        #     # Create a mapping from test bacteria names to their indices
        #     test_bacteria_to_index = {name: idx for idx, name in enumerate(self.bacteria_names_test)}
        #     mutual_bacteria = set(self.bacteria_names_train).intersection(set(self.bacteria_names_test))
        #
        #     # Prepare a tensor of zeros in the shape of training bacteria
        #     # if 0.75 <= len(mutual_bacteria)/len(self.bacteria_names_train):
        #         # aligned_abundances = torch.zeros((batch_size, len(self.bacteria_names_train)), device=device)
        #     aligned_abundances = torch.full((batch_size, len(self.bacteria_names_train)), 1, device=device)
        #     # Fill in the abundances where bacteria are found in both train and test
        #     for i, name in enumerate(self.bacteria_names_train):
        #         if name in test_bacteria_to_index:
        #             test_idx = test_bacteria_to_index[name]
        #             aligned_abundances[:, i] = abundances[:, test_idx]
        #     abundances = aligned_abundances
        #     num_bacteria = len(self.bacteria_names_train)
        #
        #     # (A) Microbe IDs => shape (batch_size, num_bacteria)
        #     microbe_ids = torch.arange(num_bacteria, device=device).unsqueeze(0).expand(batch_size, -1)

            # else:
            #     aligned_abundances = torch.full((batch_size, len(mutual_bacteria)),-1, device=device)
            #
            #     for i, name in enumerate(mutual_bacteria):
            #         test_idx = self.bacteria_names_test.index(name)
            #         aligned_abundances[:, i] = abundances[:, test_idx]
            #
            #     abundances = aligned_abundances
            #     num_bacteria = len(mutual_bacteria)
            #     indexes_of_mutual_bacteria_in_train = [self.bacteria_names_train.index(bacteria) for bacteria in mutual_bacteria]
            #     microbe_ids= torch.tensor(indexes_of_mutual_bacteria_in_train, device=device).unsqueeze(0).expand(batch_size, -1)

        # (A) Microbe IDs => shape (batch_size, num_bacteria)
        if self.flag_type_run == 'which_dataset':
            if mode== 'train':
                fraction=0.3
                non_genus_indices = [
                    i for i, name in enumerate(self.bacteria_names_train)
                    if name.count('XXX') < 2
                ]

                num_bacteria_to_mask = int(len(non_genus_indices) * fraction)
                # Randomly select a subset of bacteria
                permuted_indices = torch.randperm(len(non_genus_indices), generator=self.generator)
                selected_bacteria_indices = [non_genus_indices[i] for i in permuted_indices[:num_bacteria_to_mask].tolist()]
                # Create a mask for the selected indices
                selected_bacteria_names = [self.bacteria_names_train[i] for i in selected_bacteria_indices]
                microbe_ids = torch.arange(num_bacteria, device=device).unsqueeze(0).expand(batch_size, -1)
                # (B) Look up embedding => (batch_size, num_bacteria, embedding_dim)
                self.microbe_embedding = self.microbe_embedding.to(device)
                name_embeddings = self.microbe_embedding(microbe_ids)
                microbe_emb_weighted_train,att_mask = adjust_embedding_for_test_lodo(att_mask,abundances, name_embeddings, batch_size,
                                                                            self.bacteria_names_train,
                                                                            self.bacteria_names_test,
                                                                            device,
                                                                            microbes_to_mask=selected_bacteria_names, microbes_to_mask_index=selected_bacteria_indices)

                microbe_emb_weighted = microbe_emb_weighted_train


            elif mode == 'val':
                microbe_ids = torch.arange(num_bacteria, device=device).unsqueeze(0).expand(batch_size, -1)
                # (B) Look up embedding => (batch_size, num_bacteria, embedding_dim)
                self.microbe_embedding = self.microbe_embedding.to(device)
                microbe_emb = self.microbe_embedding(microbe_ids)

                # (C) Multiply embedding by abundance (simple approach)
                expanded_abund = abundances.unsqueeze(-1)  # => (batch_size, num_bacteria, 1)
                microbe_emb_weighted = microbe_emb * expanded_abund

            elif mode == 'test':
                microbe_ids = torch.arange(len(self.bacteria_names_train), device=device).unsqueeze(0).expand(batch_size, -1)
                # (B) Look up embedding => (batch_size, num_bacteria, embedding_dim)
                self.microbe_embedding = self.microbe_embedding.to(device)
                name_embeddings = self.microbe_embedding(microbe_ids)
                microbe_emb_weighted_train,att_mask = adjust_embedding_for_test_lodo(att_mask,abundances, name_embeddings, batch_size,
                                                                            self.bacteria_names_train,
                                                                            self.bacteria_names_test,
                                                                            device)

                microbe_emb_weighted = microbe_emb_weighted_train
                # do not pad the test set
                # att_mask= torch.zeros(batch_size, microbe_emb_weighted.shape[1], dtype=torch.bool).to(device)

            # (D) Aggregator => [CLS] embedding

            self.aggregator = self.aggregator.to(device)
            cls_emb, hidden_states = self.aggregator(microbe_emb_weighted,att_mask)

            # TODO adding residual connection
            cls_emb = cls_emb # + microbe_emb_weighted.mean(dim=1)

            # (E) Classifier => logit
            self.classifier = self.classifier.to(device)
            logits = self.classifier(cls_emb).squeeze(-1)

        elif self.flag_type_run == 'single_tag':
            microbe_ids = torch.arange(num_bacteria, device=device).unsqueeze(0).expand(batch_size, -1)
            # (B) Look up embedding => (batch_size, num_bacteria, embedding_dim)
            self.microbe_embedding = self.microbe_embedding.to(device)
            microbe_emb = self.microbe_embedding(microbe_ids)

            # (C) Multiply embedding by abundance (simple approach)
            expanded_abund = abundances.unsqueeze(-1)  # => (batch_size, num_bacteria, 1)
            microbe_emb_weighted = microbe_emb * expanded_abund
            # (D) Aggregator => [CLS] embedding
            self.aggregator = self.aggregator.to(device)
            cls_emb, hidden_states = self.aggregator(microbe_emb_weighted,att_mask)
            # TODO adding residual connection
            cls_emb = cls_emb # + microbe_emb_weighted.mean(dim=1)
            # (E) Classifier => logit
            self.classifier = self.classifier.to(device)
            logits = self.classifier(cls_emb).squeeze(-1)


        return logits

    def _common_step(self, batch, batch_idx, mode):
        abundances, labels, which_dataset_num,sample_ids,attn_mask = batch

        logits = self(abundances, attn_mask,mode)

        total_loss = self._compute_loss(logits, labels)


        preds = torch.sigmoid(logits)

        metric = \
        {'train': self.train_auroc, 'val': self.val_auroc, 'test': self.test_auroc}[
            mode]
        metric.update(preds, labels.long())

        # # Possibly add embedding reg
        # if self.taxonomy_distance_matrix is not None and (not self.microbe_embedding.weight.requires_grad == False):
        #     # Only do if embeddings are trainable
        #     reg = embedding_regularization(
        #         self.microbe_embedding.weight,
        #         self.taxonomy_distance_matrix,
        #         self.reg_mode,
        #         num_of_partition=1
        #     )
        #     self.log(f"{mode}_reg_loss", reg, on_step=False, on_epoch=True)
        #     total_loss = total_loss + self.reg_lambda * reg


        self.log(f"{mode}_loss", total_loss, on_step=False, on_epoch=True, prog_bar=True)
        return total_loss

    def check_tensor_devices(scope=None):
        if scope is None:
            frame = inspect.currentframe().f_back
            scope = {**frame.f_globals, **frame.f_locals}

        print("Checking devices of all tensors:")

        if isinstance(scope, dict):
            items = scope.items()
        else:
            # Treat scope as a single object
            items = [(scope.__class__.__name__, scope)]

        for var_name, var in items:
            try:
                if isinstance(var, torch.Tensor):
                    print(f"{var_name}: Tensor on {var.device} with shape {var.shape}")
                elif hasattr(var, 'named_parameters'):
                    for name, param in var.named_parameters():
                        print(f"{var_name}.{name}: Tensor on {param.device} with shape {param.shape}")
                elif isinstance(var, object):
                    # Look for tensor attributes inside a custom class
                    for attr_name in dir(var):
                        attr = getattr(var, attr_name)
                        if isinstance(attr, torch.Tensor):
                            print(f"{var_name}.{attr_name}: Tensor on {attr.device} with shape {attr.shape}")
            except Exception as e:
                print(f"Could not inspect {var_name}: {e}")
    def training_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, 'train')

    def validation_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, 'val')

    def test_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, 'test')

    def on_train_epoch_end(self):
        auc_train = self.train_auroc.compute()
        self.log("train_auc_tag", auc_train, prog_bar=True, on_step=False, on_epoch=True)
        self.train_auroc.reset()

    def on_validation_epoch_end(self):
        auc_val = self.val_auroc.compute()
        self.log("val_auc_tag", auc_val, prog_bar=True, on_step=False, on_epoch=True)
        self.val_auroc.reset()

    def on_test_epoch_end(self):
        auc_test = self.test_auroc.compute()
        self.log("test_auc_tag", auc_test, prog_bar=True, on_step=False, on_epoch=True)
        self.test_auroc.reset()


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        # Example Cosine Annealing with Warm Restarts
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2
        )
        return [optimizer], [scheduler]

    def _compute_loss(self, logits, labels):
        """Compute BCE with optional label smoothing."""
        # if self.label_smoothing > 0.0:
        #     # simple approach: shift labels slightly toward 0.5
        #     smoothed_labels = labels * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing
        #     return F.binary_cross_entropy_with_logits(logits.unsqueeze(1), smoothed_labels)
        # else:
        return F.binary_cross_entropy_with_logits(logits.unsqueeze(1), labels)
