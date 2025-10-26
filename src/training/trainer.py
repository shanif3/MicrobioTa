# src/training/trainer.py
import pytorch_lightning as pl
import wandb
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
import torch
import uuid
from datetime import datetime
import os
import configs.config


def train_classifier(model, train_loader, val_loader,test_loader, config: configs.config.Config, name_of_dataset,
                     dataset_id, fold, run_id,wandb_logger=None):
    if config.RANDOM_INITIALIZE_EMBEDDINGS:
        wandb_name_project = f"{config.WANDB_NAME_CLASSIFIER}_{name_of_dataset}_{dataset_id}_{config.FLAG_TYPE_RUN}_random_initialize_embedding"
    else:
        wandb_name_project= f"2inputs_one_layer_see_MODEL_scatter{config.WANDB_NAME_CLASSIFIER}_{name_of_dataset}_{dataset_id}_{config.FLAG_TYPE_RUN}"
    # # #
    # if wandb_logger is None:
    #     wandb.init(project=wandb_name_project, name=run_id, id=run_id, config=config.to_dict())
    #     wandb.run.log_code(".")  # Log the code
    #     wandb_logger = WandbLogger(project=f"{wandb_name_project}",
    #                                name=f"fold_{fold + 1}_{name_of_dataset}_{dataset_id}")

    # checkpoint_dir = f"{config.CHECKPOINT_DIR}/{wandb_name_project}/{run_id}/{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    # os.makedirs(checkpoint_dir, exist_ok=True)
    #
    # checkpoint_callback = ModelCheckpoint(
    #     monitor='val_loss',
    #     dirpath=checkpoint_dir,
    #     filename='best-checkpoint-{epoch:02d}-{val_loss:.4f}',
    #     save_top_k=1, mode='min'
    # )

    # early_stopping_callback = EarlyStopping(
    #     monitor='val_loss', patience=config.EARLY_STOPPING_PATIENCE_CLASSIFIER, mode='min', verbose=True,min_delta=0.001
    # )

    trainer = pl.Trainer(
        max_epochs=900, logger=[],
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1, callbacks=[], num_sanity_val_steps=0,enable_checkpointing=False
    )
    trainer.fit(model, train_loader, val_loader)
    # training and validation metrics
    metrics = {key: val.item() for key, val in trainer.callback_metrics.items()}  # Ensure metrics are in standard format
    if config.MODEL_TYPE == 'score_model':
        print("Fine-tuning W for test samples...")
        model.train()
        for batch in test_loader:
            model.fine_tune_W(batch, mode='test')
        model.eval()


    trainer.test(model, dataloaders=test_loader)#,ckpt_path=checkpoint_callback.best_model_path)

    # test metrics
    test_metrics = {key: val.item() for key, val in trainer.callback_metrics.items()}  # Ensure metrics are in standard format
    # test_metrics['test_auc_tag'] = auc_test
    metrics = {**metrics, **test_metrics} # Combine training, validation and test metrics into one dictionary
    wandb.finish()
    return metrics
