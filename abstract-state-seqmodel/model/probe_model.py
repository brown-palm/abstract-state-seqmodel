from typing import Tuple, Union, List
import math
import os

from contextlib import (
    contextmanager,
    redirect_stderr,
    redirect_stdout,
)
import gym
import numpy as np
import pickle as pkl
import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning.utilities.types import EPOCH_OUTPUT
from sklearn.metrics import average_precision_score, precision_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, device
from tqdm import tqdm
from transformers import AutoTokenizer, ViTFeatureExtractor, ViTModel
import wandb

import babyai.utils as utils
from gym_minigrid.minigrid import Grid
from vlm_rule_learning.data.babyai_utils import COLOR_TO_IDX, OBJECT_TO_IDX, OBJECT_IN_LEVEL_FOR_PROBE
from vlm_rule_learning.model.block import TransformerEncoder
from vlm_rule_learning.model.probe_utils import AgentStateProbe, OccupancyBoardRecoveryProbe, ClassificationProbe

os.environ["TOKENIZERS_PARALLELISM"] = "false"
Device = Union[device, str, int, None]

TILE_SIZE = 32
OBJECT_DIM = 11
COLOR_DIM = 6
STATE_DIM = 3 + 4  # +4 for agent orientations
ORIENTATION_DIM = 4
LOCATION_DIM = {
    "GoToLocal": 36,
    "MiniBossLevel": 49,
}

OBS_HEIGHT_MAP = {"partial_symbolic": 7, "full_symbolic": 8}
NUM_CELLS_MAP = {
    "GoToLocal":{
        "agent_loc": None,
        "board_type": 36,
        "board_color": 36,
        "neighbor_type": 9,
        "neighbor_color": 9,
        "board_occupy": 36,
        "neighbor_occupy": 9,
    },
    "MiniBossLevel":{
        "agent_loc": None,
        "board_type": 49,
        "board_color": 49,
        "neighbor_type": 9,
        "neighbor_color": 9,
        "board_occupy": 49,
        "neighbor_occupy": 9,
    },
}


@contextmanager
def suppress_output():
    """
        A context manager that redirects stdout and stderr to devnull
        https://stackoverflow.com/a/52442331
    """
    with open(os.devnull, "w") as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)


class ProbeCausalforMTM(pl.LightningModule):
    def __init__(
        self, 
        probe_type: str,
        probe_intermediate_dim: int,
        level_name: str,
        lr: float,
        weight_decay: float,
        betas: List[float],
        epochs: int,
        eval_interval_epochs: int, 
        embed_dim: int,
        save_dir_root: str,
        experiment_string: str,
    ):
        super().__init__()
        self.save_hyperparameters()  # for WandBLogger to log hyperparameters
        self.probe_type = probe_type
        self.probe_intermediate_dim = probe_intermediate_dim
        self.level_name = level_name
        self.num_cells = NUM_CELLS_MAP[self.level_name][self.probe_type]
        self.lr = lr
        self.weight_decay = weight_decay
        self.betas = betas
        self.num_epochs = epochs
        self.eval_interval_epochs = eval_interval_epochs
        self.embed_dim = embed_dim
        self.save_dir_root = save_dir_root
        self.experiment_string = experiment_string

        # Initialize probe
        if self.probe_type == "agent_loc":
            self.probe = AgentStateProbe(
                device = self.device, 
                num_locations = LOCATION_DIM[self.level_name], 
                intermediate_dim = self.probe_intermediate_dim, 
                input_dim = self.embed_dim,
            )

        elif self.probe_type in ["board_type", "neighbor_type"]:
            self.all_classes_object = list(OBJECT_IN_LEVEL_FOR_PROBE[self.level_name][self.probe_type])
            self.all_class_ids_object = [OBJECT_TO_IDX[c] for c in self.all_classes_object]
            self.probe = ClassificationProbe(
                device = self.device,
                num_cells = self.num_cells,
                num_classes = OBJECT_DIM,
                intermediate_dim = self.probe_intermediate_dim,
                input_dim = self.embed_dim,
            )

        elif self.probe_type in ["board_color", "neighbor_color"]:
            self.num_classes = COLOR_DIM
            self.probe = ClassificationProbe(
                device = self.device,
                num_cells = self.num_cells,
                num_classes = self.num_classes,
                intermediate_dim = self.probe_intermediate_dim,
                input_dim = self.embed_dim,
            )

        elif self.probe_type in ["board_occupy", "neighbor_occupy"]:
            self.num_classes = 2  # Occupied/Un-occupied cells
            self.probe = OccupancyBoardRecoveryProbe(
                device = self.device, 
                num_cells = self.num_cells,
                num_classes = self.num_classes,
                intermediate_dim = self.probe_intermediate_dim, 
                input_dim = self.embed_dim,
            )

        if self.probe_type in ["board_type", "neighbor_type", "board_color", "neighbor_color", "board_occupy", "neighbor_occupy"]:
            self.test_preds = []
            self.test_labels = []
            self.test_size_list = []
            self.test_ap = []  # For saving average precision metrics
            
    def forward(
        self, 
        inputs, 
        labels
    ):
        return self.probe(inputs, labels)

    def training_step(self, batch, batch_idx):
        hidden_features, labels = batch
        labels = labels.long()
        _, loss = self(hidden_features, labels)
        self.log(f"train/loss", loss, sync_dist=True)
        return loss
    
    def wandb_define_metrics(self):
        wandb.define_metric(f"valid/acc", summary="max")
        wandb.define_metric(f"test/acc", summary="max")
            
        if self.probe_type in ["board_type", "board_color", "neighbor_type", "neighbor_color", "board_occupy", "neighbor_occupy"]:
            wandb.define_metric(f"valid/mAP", summary="max")
            wandb.define_metric(f"test/mAP", summary="max")
        
    def validation_step(self, batch, batch_idx):
        if self.global_step == 0:
            self.wandb_define_metrics()
        
        hidden_features, labels = batch
        labels = labels.long()
        logits, loss = self(hidden_features, labels)
        acc, _, _ = self.evaluate_performance(logits, labels)

        self.log(f"valid/loss", loss, sync_dist=True)
        self.log(f"valid/acc", acc, sync_dist=True)

        # Code for Mean average precision evaluation
        if self.probe_type in ["board_type", "neighbor_type"]:
            ap_per_class = self.evaluate_average_precision(logits, labels, num_classes=OBJECT_DIM)
            ap_per_class_filtered = self.filter_ap_list_object(ap_per_class)
            mAP = ap_per_class_filtered.mean()
            self.log(f"valid/mAP", mAP, sync_dist=True)

        elif self.probe_type in ["board_color", "neighbor_color"]:
            ap_per_class = self.evaluate_average_precision(logits, labels, num_classes=COLOR_DIM)
            mAP = ap_per_class.mean()
            self.log(f"valid/mAP", mAP, sync_dist=True)

        elif self.probe_type in ["board_occupy", "neighbor_occupy"]:
            ap_per_class = self.evaluate_average_precision(logits, labels, num_classes=2)
            mAP = ap_per_class.mean()
            self.log(f"valid/mAP", mAP, sync_dist=True)

    def test_step(self, batch, batch_idx):
        hidden_features, labels = batch
        labels = labels.long()
        logits, _ = self(hidden_features, labels)
        acc, preds, _ = self.evaluate_performance(logits, labels)

        self.log(f"test/acc", acc, sync_dist=True)
        
        # Code for Mean average precision evaluation
        if self.probe_type in ["board_type", "neighbor_type"]:
            ap_per_class = self.evaluate_average_precision(logits, labels, num_classes=OBJECT_DIM)
            ap_per_class_filtered = self.filter_ap_list_object(ap_per_class)
            mAP = ap_per_class_filtered.mean()
            self.log(f"test/mAP", mAP, sync_dist=True)

        elif self.probe_type in ["board_color", "neighbor_color"]:
            ap_per_class = self.evaluate_average_precision(logits, labels, num_classes=COLOR_DIM)
            mAP = ap_per_class.mean()
            self.log(f"test/mAP", mAP, sync_dist=True)

        elif self.probe_type in ["board_occupy", "neighbor_occupy"]:
            ap_per_class = self.evaluate_average_precision(logits, labels, num_classes=2)
            mAP = ap_per_class.mean()
            self.log(f"test/mAP", mAP, sync_dist=True)

        if self.probe_type in ["board_type", "neighbor_type", "board_color", "neighbor_color", "board_occupy", "neighbor_occupy"]:
            self.test_preds.append(preds.copy())
            self.test_labels.append(labels.detach().cpu().numpy().copy())
            self.test_ap.append(ap_per_class.copy())
            self.test_size_list.append(logits.size(0))

    def filter_ap_list_object(self, ap_list):
        assert ap_list.shape[-1] == OBJECT_DIM, "Can only filter an ap_list for objects."
        ap_list_filtered = ap_list[self.all_class_ids_object]
        return ap_list_filtered

    def on_test_epoch_end(self):
        # on_test_epoch_end() for mean average precision computation.
        
        if self.probe_type in ["board_type", "neighbor_type"]:
            test_ap = np.stack(self.test_ap)  # [n_test_batch, OBJECT_DIM]
            
            # Process average precision for objects
            all_classes = list(OBJECT_IN_LEVEL_FOR_PROBE[self.level_name][self.probe_type])
            all_class_ids = [OBJECT_TO_IDX[c] for c in all_classes]
            ap_list_filtered = test_ap[:, all_class_ids]

            ap_per_class = np.zeros(len(all_classes))
            for ap_list, size in zip(ap_list_filtered, self.test_size_list):
                # Weighted sum over all lists of average precision per class
                ap_per_class = ap_per_class + ap_list * size
            ap_per_class = ap_per_class / sum(self.test_size_list)
            mAP = ap_per_class.mean()

        elif self.probe_type in ["board_color", "neighbor_color"]:
            test_ap = np.stack(self.test_ap)  # [n_test_batch, COLOR_DIM]

            # Process average precision for colors
            all_classes = list(COLOR_TO_IDX.keys())
            all_class_ids = [COLOR_TO_IDX[c] for c in all_classes]
            ap_per_class = np.zeros(len(all_classes))
            for ap_list, size in zip(test_ap, self.test_size_list):
                # Weighted sum over all lists of average precision per class
                ap_per_class = ap_per_class + ap_list * size
            ap_per_class = ap_per_class / sum(self.test_size_list)
            mAP = ap_per_class.mean()

        elif self.probe_type in ["board_occupy", "neighbor_occupy"]:
            all_classes = ["empty", "occupied"]
            all_class_ids = [0, 1]
            test_ap = np.stack(self.test_ap)  # [n_test_batch, 2]

            ap_per_class = np.zeros(len(all_classes))
            for ap_list, size in zip(test_ap, self.test_size_list):
                # Weighted sum over all lists of average precision per class
                ap_per_class = ap_per_class + ap_list * size
            ap_per_class = ap_per_class / sum(self.test_size_list)
            mAP = ap_per_class.mean()

        if self.probe_type in ["board_type", "neighbor_type", "board_color", "neighbor_color", "board_occupy", "neighbor_occupy"]:
            # Save predictions & metrics to local
            save_dir = os.path.join(self.save_dir_root, self.experiment_string)
            os.makedirs(save_dir, exist_ok=True)
            
            with open(os.path.join(save_dir, "mean_average_precision.pkl"), "wb") as f:
                pkl.dump({
                    "class": all_classes,
                    "class_ids": all_class_ids,
                    "ap_list_raw": test_ap,
                    "ap_per_class": ap_per_class,
                    "mAP": mAP,
                    "test_predictions": np.concatenate(self.test_preds),
                    "test_labels": np.concatenate(self.test_labels),
                }, f)
                
            self.log(f"test/mAP", mAP, sync_dist=True)
            
    def evaluate_performance(self, logits, labels):
        logits = logits.view(-1, logits.size(-1))  # [N, num_class]
        preds = logits.argmax(dim=-1).detach().cpu().numpy()
        labels = labels.detach().cpu().flatten().numpy()
        acc = sum(preds == labels) / len(labels)
        return acc, preds, labels
    
    def evaluate_average_precision(self, logits, labels, num_classes):
        """
        Arguments:
            logits: [batch_size, num_cells, num_classes]
            labels: [batch_size, num_cells]
        """
        probs = F.softmax(logits.view(-1, num_classes), dim=-1).detach().cpu().numpy()  # [batch_size * num_cells, num_classes]
        labels = F.one_hot(labels.flatten(), num_classes=num_classes).detach().cpu().numpy()  # [batch_size * num_cells, num_classes]
        with suppress_output():
            ap_per_class = average_precision_score(labels, probs, average=None)
        ap_per_class = np.nan_to_num(ap_per_class)
        return ap_per_class

    def configure_optimizers(self):
        optimizer, scheduler = self.probe.configure_optimizers(
            lr = self.lr,
            weight_decay = self.weight_decay,
            betas = self.betas,
        )
        return {
            "optimizer": optimizer,
        }
