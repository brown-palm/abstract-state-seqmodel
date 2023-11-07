import gym
import os

import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.callbacks import TQDMProgressBar, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
import torch
import wandb

from data.precomputed_internal_activation_data import PrecomputedInternalActivationDataModule
import babyai.utils as utils

OPTIONAL_LAST_STATE = True
device = "cuda" if torch.cuda.is_available() else "cpu"


@hydra.main(version_base=None, config_path="../config", config_name="train_precompute_probe")
def main(cfg: DictConfig):

    assert cfg.level_name in ["GoToLocal", "MiniBossLevel"]
    cfg.env_name = f"BabyAI-{cfg.level_name}-v0"
    
    pl.seed_everything(cfg.data_seed)
    dm = PrecomputedInternalActivationDataModule(cfg, device)
    dm.setup()
    
    model_type_dict = {
        "causal_all": 1,
        "causal_withold_state": 2,
    }
    model_type = model_type_dict[cfg.model_type]
    
    experiment_string = f"probe_{cfg.probe_type}_{cfg.probe_layer}|r_{1 if cfg.is_randomly_initialized else 0}|g_{1 if cfg.is_goal else 0}|P-{model_type}|{cfg.level_name}|n_train-{cfg.train_size}|seed-{cfg.seed}|d_s-{cfg.data_seed}|lr-{cfg.lr}|bs-{cfg.batch_size * len(cfg.devices)}|E-{cfg.epochs}|d_emb-{cfg.probe_intermediate_dim}|{cfg.info}"

    pl.seed_everything(cfg.seed)
    probe_model = hydra.utils.instantiate(
        cfg.model,
        probe_type=cfg.probe_type,
        probe_intermediate_dim=cfg.probe_intermediate_dim,
        level_name=cfg.level_name,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        betas=cfg.betas,
        epochs=cfg.epochs,
        eval_interval_epochs=cfg.eval_interval_epochs, 
        embed_dim=768,  # TODO: get rid of hard-coded embed_dim
        save_dir_root=cfg.save_dir_root,
        experiment_string=experiment_string,
    )

    working_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    wandb_logger = WandbLogger(project=cfg.wandb.project, save_dir=os.path.join(working_dir, "wandb_logs"), version=experiment_string)

    progressbar = TQDMProgressBar()
    lr_monitor = LearningRateMonitor(logging_interval="step")

    checkpoint = ModelCheckpoint(
        monitor="valid/acc",
        filename=cfg.env_name + "epoch:{epoch}-val_acc:{valid/acc:.3f}",
        mode="max",
        save_top_k=1,
        auto_insert_metric_name=False,
        every_n_epochs=cfg.eval_interval_epochs
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=cfg.devices, 
        max_epochs=cfg.epochs,
        logger=wandb_logger,
        callbacks=[progressbar, lr_monitor, checkpoint],
        check_val_every_n_epoch=cfg.eval_interval_epochs
    )

    trainer.fit(model=probe_model, train_dataloaders=dm.train_dataloader(), val_dataloaders=dm.val_dataloader())

    trainer.test(ckpt_path="best", dataloaders=dm.test_dataloader())

if __name__ == "__main__":
    main()