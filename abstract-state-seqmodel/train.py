import gym
import os

import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.callbacks import TQDMProgressBar, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
import torch
import wandb

from data.sequence import BabyAiDataModule
import babyai.utils as utils


@hydra.main(version_base=None, config_path="../config", config_name="train")
def main(cfg: DictConfig):
    pl.seed_everything(cfg.seed)

    assert cfg.level_name in ["GoToLocal", "MiniBossLevel"]
    cfg.env_name = f"BabyAI-{cfg.level_name}-v0"
    env = gym.make(cfg.env_name)
    obs_dim = 768
    action_dim = env.action_space.n + 2  # for BOS and PAD tokens

    dm = BabyAiDataModule(cfg)
    dm.setup()
    
    mtm = hydra.utils.instantiate(
        cfg.model,
        level_name=cfg.level_name,
        env_name=cfg.env_name,
        obs_dim=obs_dim,
        action_dim=action_dim,
        goal_dim=30522,  # WARNING: hard-coded tokenizer.vocab_size
        epochs=cfg.epochs,
        lr=cfg.lr,
        ctx_size=cfg.ctx_size,
        eval_interval_epochs=cfg.eval_interval_epochs,
        max_goal_length=cfg.max_goal_length,
        obs_type=cfg.obs_type,
        max_step_threshold=cfg.max_step_threshold,
        data_root=cfg.data_root,
        is_goal=cfg.is_goal
    )

    working_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
    model_type_dict = {
        "causal_all": 1,
        "causal_withold_state": 2,
    }
    model_type = model_type_dict[cfg.model.model_config.model_type]
    
    wandb_logger = WandbLogger(project=cfg.wandb.project, save_dir=os.path.join(working_dir, "wandb_logs"), version=f"P-{model_type}|{cfg.env_name.split('-')[1]}|g_{1 if cfg.is_goal else 0}|{cfg.obs_type}|n_train-{cfg.train_size}|seed-{cfg.seed}|ctx-{cfg.ctx_size}|lr-{cfg.lr}|bs-{cfg.batch_size * len(cfg.devices)}|E-{cfg.epochs}|d_emb-{cfg.model.model_config.embed_dim}|d_sym-{cfg.model.model_config.symbolic_embed_dim}|L-{cfg.model.model_config.n_enc_layers}|H-{cfg.model.model_config.n_head}|D-{cfg.model.model_config.pdrop}|{cfg.info}")

    progressbar = TQDMProgressBar()
    lr_monitor = LearningRateMonitor(logging_interval="step")


    checkpoint = ModelCheckpoint(
        monitor='valid/success_rate_over_all',
        filename=cfg.env_name + '-epoch:{epoch}-val_sr:{valid/success_rate_over_all:.3f}',
        mode='max',
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
    )

    if cfg.resume:
        trainer.fit(model=mtm, train_dataloaders=dm.train_dataloader(), val_dataloaders=dm.val_dataloader(), ckpt_path=cfg.seqmodel_ckpt_path)
    else:
        trainer.fit(model=mtm, train_dataloaders=dm.train_dataloader(), val_dataloaders=dm.val_dataloader())

    trainer.test(ckpt_path="best", dataloaders=dm.test_dataloader())

if __name__ == "__main__":
    main()