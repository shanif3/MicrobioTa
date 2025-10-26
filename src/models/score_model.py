# model.py
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics.classification import BinaryAUROC
import configs.config
from torch.nn import functional as F
import wandb


class MicrobeModel(pl.LightningModule):
    def __init__(
            self,
            num_microbes_train,
            num_samples,
            config: configs.config.Config,
            pretrained_embedding=None,
    ):
        super().__init__()
        self.save_hyperparameters()
        # self.early_stopping_abundance = config.EARLY_STOPPING_ABUNDANCE
        self.reg_lambda = config.REG_LAMBDA
        # 0) Microbe embeddings (sample-specific)
        self.num_microbes = num_microbes_train
        self.num_samples = num_samples  # Training samples; test samples will append to this
        self.embedding_dim = config.EMBEDDING_DIM
        self.contrast_lambda = config.CONTRASTIVE_LAMBDA
        self.contrast_margin = config.CONTRASTIVE_MARGIN
        self.lr_fine_tune_w = config.LR_FINE_TUNE_W

        if isinstance(pretrained_embedding, nn.Embedding):
            base_embedding = pretrained_embedding.weight.detach().clone()
        elif isinstance(pretrained_embedding, torch.Tensor):
            base_embedding = pretrained_embedding.clone()
        else:
            base_embedding = torch.randn(self.num_microbes, config.EMBEDDING_DIM)

        self.microbe_embedding = base_embedding  # ( num_microbes, embedding_dim)

        # 1) Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.EMBEDDING_DIM,
            nhead=config.NHEAD_CLASSIFIER,
            dim_feedforward=4 * config.EMBEDDING_DIM,
            dropout=config.DROPOUT_TRANSFORMER,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.NUM_LAYERS_ENCODER)

        self.abundance_mlp = nn.Sequential(
            nn.Linear(4 * self.embedding_dim, self.embedding_dim),
            nn.ReLU(),
            nn.Linear(self.embedding_dim, 1)
        )
        self.abundance_mlp2inputs = nn.Sequential(
            nn.Linear(2 * self.embedding_dim, 1)
        )

        # 2) Learnable sample-specific W_k
        self.W = nn.Parameter(torch.ones(self.num_samples, config.EMBEDDING_DIM))
        # NEW: Batch embeddings e_b (bias vector per batch)
        self.num_batches = 1  # config.NUM_BATCHES

        # 3) Classification head
        # self.classifier = nn.Sequential(
        #     nn.Linear(config.EMBEDDING_DIM,5),  # Reduced units
        #     nn.ReLU(),
        #     nn.Dropout(0.8),
        #     nn.Linear(5, 1)  # Single layer to output
        # )
        self.classifier = nn.Sequential(
            nn.Linear(config.EMBEDDING_DIM, config.HIDDEN_DIM_CLASSIFIER1),  # Reduced units
            nn.ReLU(),
            nn.Dropout(config.DROPOUT_LINEAR_CLASSIFIER),
            nn.Linear(config.HIDDEN_DIM_CLASSIFIER1, 1)  # Single layer to output
        )
        # 4) Hyperparams
        self.lr = config.LR_CLASSIFIER
        self.weight_decay = config.WEIGHT_DECAY_CLASSIFIER  # *10

        # Loss functions
        self.reg_loss_fn = nn.MSELoss()
        self.cls_loss_fn = nn.BCEWithLogitsLoss()
        self.train_auroc = BinaryAUROC()
        self.val_auroc = BinaryAUROC()
        self.test_auroc = BinaryAUROC()
        self.device_to_use = config.DEVICE

    # def forward(self, sample_ids):
    #     Z_k = self.microbe_embedding
    #     Z_prime = self.encoder(Z_k)  # Z'_ik: (num_microbes, embedding_dim)
    #     W_k = self.W[sample_ids]
    #     abundance_hat= torch.einsum('bn,dn->bd', Z_prime, W_k) # (batch size, num_microbes)
    #     cls_output = self.classifier(W_k)
    #     return abundance_hat.transpose(1,0), cls_output, Z_prime

    def forward(self, sample_ids, which_dataset):
        W_k = self.W[sample_ids]  # shape: (batch, d)

        Z_prime = self.encoder(self.microbe_embedding.unsqueeze(0)).squeeze(0)  # shape: (n_microbes, d)

        # Abundance prediction: dot product Z_i · (W_k + e_b)
        # abundance_hat = torch.einsum('dn,bn->bd', Z_prime, W_k)  # (batch size, num_microbes)

        # # NEW: MLP-based abundance prediction
        batch_size, num_microbes = W_k.size(0), Z_prime.size(0)
        Z_expand = Z_prime.unsqueeze(0).expand(batch_size, -1, -1)  # (B, M, D)
        W_expand = W_k.unsqueeze(1).expand(-1, num_microbes, -1)  # (B, M, D)
        # input_features = torch.cat([Z_expand, W_expand, Z_expand - W_expand, Z_expand * W_expand], dim=-1)  # (B, M, 4D)
        input_features = torch.cat([Z_expand, W_expand], dim=-1)  # (B, M, 2D)
        abundance_hat = self.abundance_mlp2inputs(input_features).squeeze(-1)  # (B, M)

        # Classification from W_k
        cls_logits = self.classifier(W_k)  # (batch,)

        return abundance_hat, cls_logits, Z_prime

    def _common_step(self, batch, batch_idx, mode):
        abundance_true, labels, which_dataset, sample_ids, _ = batch
        abundance_hat, cls_logits, Z_prime = self(sample_ids, which_dataset)

        reg_loss = self.reg_loss_fn(abundance_hat, abundance_true)
        cls_loss = self._compute_loss(cls_logits, labels)
        total_loss = (1 - self.reg_lambda) * reg_loss + self.reg_lambda * cls_loss

        # # NEW: L2 regularization on W_k and e_b
        # W_k = self.W[sample_ids]
        # l2_reg = W_k.norm(p=2, dim=1).mean()
        # total_loss += (l2_reg / 2)
        # self.log(f"{mode}_l2_reg", l2_reg / 2, on_step=False, on_epoch=True, prog_bar=True)
        # if mode == "train":
        #     labels_exp = labels.unsqueeze(1)  # (B, 1)
        #     matches = (labels_exp == labels_exp.T).float()  # (B, B) same class = 1, else 0
        #
        #     dists = torch.cdist(W_k, W_k, p=2)  # pairwise L2 distances (B, B)
        #
        #     pos_loss = (matches * dists.pow(2)).sum() / matches.sum().clamp(min=1.0)
        #     neg_mask = 1 - matches
        #     neg_loss = (neg_mask * F.relu(self.contrast_margin - dists).pow(2)).sum() / neg_mask.sum().clamp(min=1.0)
        #
        #     # contrastive loss on W so that we will keep close samples together
        #     contrast_loss = self.contrast_lambda * (pos_loss + neg_loss)
        #     total_loss += contrast_loss / 2
        #     self.log(f"{mode}_contrast_loss", contrast_loss / 2, on_step=False, on_epoch=True, prog_bar=True)

        metric = {'train': self.train_auroc, 'val': self.val_auroc, 'test': self.test_auroc}[mode]
        metric.update(cls_logits.sigmoid(), labels.long())

        # Plot scatter plot of abundance_true vs abundance_hat
        # abundance_true_mean = abundance_true.mean(dim=0)
        # abundance_hat_mean = abundance_hat.mean(dim=0)

        # Limit number of microbes to avoid clutter
        N = min(200, abundance_true.shape[1])


        table = wandb.Table(columns=["abundance_true", "abundance_hat"])
        for i in range(N):
            for j in range(abundance_true.shape[0]):  # Iterate over batch size
                table.add_data(
                    abundance_true[j, i].item(),
                    abundance_hat[j, i].item()
                )
        #
        self.logger.experiment.log({
            f"{mode}/abundance_scatter": wandb.plot.scatter(
                table, "abundance_true", "abundance_hat",
                title=f"{mode.capitalize()} Abundance Scatter (Mean Over Batch)"
            )
        })

        ss_res = torch.sum((abundance_true - abundance_hat) ** 2)
        ss_tot = torch.sum((abundance_true - abundance_true.mean()) ** 2)
        r_square = 1 - (ss_res / ss_tot)

        # Log R²
        self.log(f'{mode}_r_square', r_square, on_step=False, on_epoch=True, prog_bar=True)

        self.log(f'{mode}_loss', total_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(f'{mode}_reg_loss', reg_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(f'{mode}_cls_loss', cls_loss, on_step=False, on_epoch=True, prog_bar=True)
        return total_loss

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, 'train')

    def validation_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, 'val')

    def test_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, 'test')

    def fine_tune_W(self, batch, mode):
        """
        For a new sample, fine-tune W_k with frozen Transformer, then classify.
        """
        abundance_true, labels, which_dataset, sample_ids, _ = batch
        sample_ids = sample_ids.to(self.device_to_use)

        # # reset W of sample_ids
        if mode == 'val':
            self.W[sample_ids] = torch.ones(len(sample_ids), self.embedding_dim).to(self.device_to_use)
        else:
            with torch.no_grad():
                self.W[sample_ids] = torch.ones(len(sample_ids), self.embedding_dim).to(self.W.device)

        self.encoder.eval()
        self.classifier.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.classifier.parameters():
            param.requires_grad = False
        self.microbe_embedding.requires_grad_(False)

        # pre-compute Z_prime
        with torch.no_grad():
            Z = self.microbe_embedding
            Z_prime = self.encoder.to(self.device_to_use)(Z)  # (num_microbes, embedding_dim)

        # Fine-tune W_k
        sample_indices = sample_ids.to(self.W.device)
        W_k_to_optimize = self.W[sample_indices].clone().detach().requires_grad_(True)

        optimizer = torch.optim.AdamW([W_k_to_optimize], lr=self.lr_fine_tune_w, weight_decay=self.weight_decay)

        patience, min_delta, best_loss, wait = 30, 0.1, float('inf'), 0
        # patience= self.early_stopping_abundance
        num_steps = 1000
        with torch.enable_grad():
            for _ in range(num_steps):
                optimizer.zero_grad()
                # torch.set_grad_enabled(True)
                # abundance_hat = torch.einsum('bn,dn->db', Z_prime.to(W_k_to_optimize), W_k_to_optimize)

                # Get batch embeddings
                corrected_W = W_k_to_optimize  # (batch, dim)

                # Use MLP to predict abundance
                batch_size, num_microbes = corrected_W.size(0), Z_prime.size(0)
                Z_expand = Z_prime.unsqueeze(0).expand(batch_size, -1, -1).to(self.W.device)  # (B, M, D)
                W_expand = corrected_W.unsqueeze(1).expand(-1, num_microbes, -1).to(self.W.device)  # (B, M, D)
                # input_features = torch.cat([Z_expand, W_expand, Z_expand - W_expand, Z_expand * W_expand],
                #                            dim=-1)  # (B, M, 4D)
                input_features = torch.cat([Z_expand, W_expand],dim=-1)  # (B, M, 2D)
                # abundance_hat = self.abundance_mlp(input_features).squeeze(-1).to(self.W.device)  # (B, M)
                abundance_hat= self.abundance_mlp2inputs(input_features).squeeze(-1).to(self.W.device)  # (B, M)
                reg_loss = self.reg_loss_fn(abundance_hat, abundance_true.to(W_k_to_optimize.device))
                reg_loss.backward()
                optimizer.step()

                # Early stopping
                if reg_loss < best_loss - min_delta:
                    best_loss = reg_loss
                    wait = 0
                else:
                    wait += 1
                    if wait >= patience:
                        break
            # print('stopping at step:', _)
            # Update the global W parameter with the optimized values
            with torch.no_grad():
                self.W[sample_indices] = W_k_to_optimize.detach().clone()

    def on_validation_epoch_start(self):
        # Fine-tune W for validation samples at the start of each validation epoch
        self.train()
        for batch in self.trainer.val_dataloaders:
            self.fine_tune_W(batch, 'val')
        self.eval()

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
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2
        )
        return [optimizer], [{'scheduler': scheduler, 'interval': 'epoch'}]

    def _compute_loss(self, logits, labels):
        return F.binary_cross_entropy_with_logits(logits, labels.float())
