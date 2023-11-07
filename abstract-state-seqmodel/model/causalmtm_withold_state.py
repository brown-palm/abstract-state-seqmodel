from typing import Tuple, Union
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
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, device
from tqdm import tqdm
from transformers import AutoTokenizer, ViTFeatureExtractor, ViTModel
import wandb

import babyai.utils as utils
from gym_minigrid.minigrid import Grid
from vlm_rule_learning.data.sequence import rotate_state
from vlm_rule_learning.model.block import TransformerEncoder

os.environ["TOKENIZERS_PARALLELISM"] = "false"
Device = Union[device, str, int, None]

OPTIONAL_LAST_STATE = True
NUM_OBSERVATIONS = 2 if OPTIONAL_LAST_STATE else 1

TILE_SIZE = 32
OBJECT_DIM = 11
COLOR_DIM = 6
STATE_DIM = 3 + 4  # +4 for agent orientations

OBS_HEIGHT_MAP = {
    "full_symbolic": {"GoToLocal": 8, "MiniBossLevel": 9}
}
EVAL_HORIZON_MAP = {
    "BabyAI-GoToObj-v0": 64,
    "BabyAI-GoToLocal-v0": 64,
    "BabyAI-MiniBossLevel-v0": 64,
    "BabyAI-PutNextLocal-v0": 128,
    "BabyAI-GoToObjMaze-v0": 200,
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

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, goals: Tensor, observations: Tensor, actions: Tensor, max_goal_length: int) -> Tensor:
        """
        Args:
            [batch_size, seq_len, embedding_dim]
        """
        goals += self.pe[:, :max_goal_length]
        observations += self.pe[:, max_goal_length: max_goal_length + observations.shape[1]]
        actions += self.pe[:, max_goal_length: max_goal_length + actions.shape[1]]
        return goals, observations, actions


def triangular_mask(size, device, diagonal_shift=1):
    """
    generate upper triangular matrix filled with ones
    """
    square = torch.triu(torch.ones(size, size, device=device), diagonal=diagonal_shift)
    square = square.masked_fill(square == 1., float("-inf"))
    return square


class CausalTransformer(nn.Module):
    def __init__(
        self, 
        obs_dim: int, 
        action_dim: int, 
        goal_dim: int, 
        ctx_size: int, 
        max_goal_length: int, 
        obs_type: str,
        level_name: str,
        config: DictConfig
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.goal_dim = goal_dim  # define by tokenizer
        self.embed_dim = config.embed_dim
        self.symbolic_embed_dim = config.symbolic_embed_dim
        self.max_goal_length = max_goal_length
        self.obs_type = obs_type
        self.level_name = level_name

        self.postional_encoding = PositionalEncoding(d_model=self.embed_dim)
        self.goal_embed = nn.Embedding(self.goal_dim, self.embed_dim, padding_idx=0)

        self.action_embed = nn.Embedding(self.action_dim, self.embed_dim, padding_idx=0)

        self.action_decoder = nn.Linear(self.embed_dim, self.action_dim)

        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            dim_feedforward=self.embed_dim * 4, 
            nhead=config.n_head, 
            dropout=config.pdrop, 
            activation=F.gelu, 
            norm_first=True, 
            batch_first=True
        )

        self.transformer = TransformerEncoder(self.transformer_layer, config.n_enc_layers)
        
        self.obs_height = OBS_HEIGHT_MAP[self.obs_type][self.level_name]
        self.object_embed = nn.Embedding(OBJECT_DIM, self.symbolic_embed_dim)
        self.color_embed = nn.Embedding(COLOR_DIM, self.symbolic_embed_dim)
        self.state_embed = nn.Embedding(STATE_DIM, self.symbolic_embed_dim)

        self.image_encoder = nn.Sequential(
            nn.Conv2d(self.symbolic_embed_dim*3, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            nn.Flatten(),
            nn.Linear(128 * self.obs_height * self.obs_height, 1024),
            nn.ReLU(),
            nn.Linear(1024, 768),
            nn.ReLU(),
            nn.Linear(768, self.embed_dim),
        )

    def causal_attn_mask(self, ctx_size: int, device: Device):
        max_len = self.max_goal_length + NUM_OBSERVATIONS + ctx_size  # goal + obs + action
        # goal only attend to itself
        goal_to_all = torch.ones([self.max_goal_length, max_len]) * float("-inf")
        goal_to_all[:, :self.max_goal_length] = 0.

        obs_to_goal = torch.zeros(NUM_OBSERVATIONS, self.max_goal_length)
        if not OPTIONAL_LAST_STATE:
            obs_to_obs = torch.zeros(1, 1)
        else:
            obs_to_obs = torch.triu(torch.ones(2, 2), diagonal=1)
            obs_to_obs = obs_to_obs.masked_fill(obs_to_obs == 1., float("-inf"))
        obs_to_actions = torch.ones(NUM_OBSERVATIONS, ctx_size) * float("-inf")
        obs_to_all = torch.cat([obs_to_goal, obs_to_obs, obs_to_actions], dim=1)

        # actions attend to current/previous obs and goal
        action_to_goal = torch.zeros(ctx_size, self.max_goal_length)
        action_to_obs = torch.zeros(ctx_size, NUM_OBSERVATIONS)
        action_to_action = torch.triu(torch.ones(ctx_size, ctx_size), diagonal=1)
        action_to_action = action_to_action.masked_fill(action_to_action == 1., float("-inf"))
        action_to_all = torch.cat([action_to_goal, action_to_obs, action_to_action], dim=1)

        causal_mask = torch.cat([goal_to_all, obs_to_all, action_to_all], dim=0).to(device=device)
        return causal_mask

    def forward(self, observations: Tensor, actions: Tensor, goals: Tensor, padding_mask: Tensor, probe_layer=None):
        batch_size, ctx_size = actions.size()[:2]

        # goal: B X self.max_goal_length x embed_dim
        goal_embeddings = self.goal_embed(goals)
        
        batch_sz, epi_len, c, h, w = observations.size()
        observations = observations.view(-1, c, h, w)  # channel = [object, color, state]
        object_embeds = self.object_embed(observations[:,0,:,:])  # [num_obs, self.symbolic_embed_dim, h, w]
        color_embeds = self.color_embed(observations[:,1,:,:])
        state_embeds = self.state_embed(observations[:,2,:,:])
        cell_embeds = torch.cat((object_embeds, color_embeds, state_embeds), dim=-1)  # [num_obs, h, w, self.symbolic_embed_dim*3]
        cell_embeds = cell_embeds.permute(0,3,1,2)
        obs_embeddings = self.image_encoder(cell_embeds)
        obs_embeddings = obs_embeddings.view(batch_sz, epi_len, -1)

        # actions: B X L x action_dim
        action_embeddings = self.action_embed(actions)

        g, o, a = self.postional_encoding(goal_embeddings, obs_embeddings, action_embeddings, self.max_goal_length)
        input_embs = torch.cat([g, o, a], dim=1)

        goal_padding_mask = torch.zeros(size=(batch_size, self.max_goal_length), dtype=torch.bool, device=observations.device)
        obs_padding_mask = torch.zeros(size=(batch_size, NUM_OBSERVATIONS), dtype=torch.bool, device=observations.device)
        seq_padding_mask = torch.cat([goal_padding_mask, obs_padding_mask, padding_mask], dim=1)  # TODO: pad goal seqs
        causal_mask = self.causal_attn_mask(ctx_size, observations.device)

        outputs = self.transformer(input_embs, causal_mask, seq_padding_mask, probe_layer=probe_layer)
        pred_a = outputs[:, (self.max_goal_length + NUM_OBSERVATIONS):]
        
        if probe_layer is None:
            pred_a = self.action_decoder(pred_a)

        return pred_a


class CausalforMTM(pl.LightningModule):
    def __init__(
        self, 
        level_name: str,
        env_name: str, 
        obs_dim: int, 
        action_dim: int, 
        goal_dim: int, 
        lr: float, 
        epochs: int, 
        ctx_size: int, 
        eval_interval_epochs: int, 
        max_goal_length: int, 
        obs_type: str,
        max_step_threshold: int,
        data_root: str,
        is_goal: bool,
        model_config: DictConfig
    ):
        super().__init__()
        self.save_hyperparameters()  # for WandBLogger to log hyperparameters
        self.level_name = level_name
        self.env_name = env_name
        self.ctx_size = ctx_size
        self.lr = lr
        self.num_epochs = epochs
        self.eval_interval_epochs = eval_interval_epochs
        self.max_goal_length = max_goal_length
        self.obs_type = obs_type
        self.obs_height = OBS_HEIGHT_MAP[self.obs_type][self.level_name]
        self.max_step_threshold = max_step_threshold
        self.embed_dim = model_config.embed_dim
        self.data_root = data_root
        self.is_goal = is_goal
        
        self.model = CausalTransformer(obs_dim, action_dim, goal_dim, ctx_size, max_goal_length, obs_type, level_name, model_config)
        
        self.save_hyperparameters()

        self.env = gym.make(self.env_name)
        
        # load pretrained tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            
        # load validation/test inference splits
        self.validation_seed_list = self.load_inference_split(split="validation")
        self.test_seed_list = self.load_inference_split(split="test")

    def load_inference_split(self, split: str):
        data_path = os.path.join(self.data_root, self.level_name, f"{split}_inference_split.pkl")
        with open(data_path, "rb") as f:
            seed_list = pkl.load(f)
        return seed_list
    
    def forward(self, observations: Tensor, actions: Tensor, goals: Tensor, padding_mask: Tensor):
        if not self.is_goal:
            goals = torch.zeros_like(goals, device=goals.device).long()   # using pad token (0) from huggingface bert tokenizer
        return self.model.forward(observations, actions, goals, padding_mask)

    def loss(self, target_a, pred_a):
        loss_fn = nn.CrossEntropyLoss(reduction="mean", ignore_index=0)  # ignore padded actions
        loss = loss_fn(pred_a.view(-1, pred_a.shape[-1]), target_a.view(-1).long())
        return loss

    def training_step(self, batch, batch_idx):
        observations, actions, label_actions, goals, valid_lengths, padding_mask = batch
        goals = goals.long()
        actions = actions.long()
        label_actions = label_actions.long()
        observations = observations.long()
            
        pred_a = self(observations, actions, goals, padding_mask)
        loss = self.loss(label_actions, pred_a)

        self.log("train/loss", loss, sync_dist=True)

        return loss
    
    def wandb_define_metrics(self):
        wandb.define_metric("valid/mean_steps_over_all", summary="min")
        wandb.define_metric("valid/mean_steps_over_success", summary="min")
        wandb.define_metric("valid/success_rate_over_all", summary="max")
        wandb.define_metric("valid/success_rate_over_success", summary="max")
        wandb.define_metric("test/mean_steps_over_all", summary="min")
        wandb.define_metric("test/mean_steps_over_success", summary="min")
        wandb.define_metric("test/success_rate_over_all", summary="max")
        wandb.define_metric("test/success_rate_over_success", summary="max")
    
    def validation_step(self, batch, batch_idx):
        if self.global_step == 0:
            self.wandb_define_metrics()
        
        observations, actions, label_actions, goals, valid_lengths, padding_mask = batch
        goals = goals.long()
        actions = actions.long()
        label_actions = label_actions.long()
        observations = observations.long()

        pred_a = self(observations, actions, goals, padding_mask)
        loss = self.loss(label_actions, pred_a)

        self.log("valid/loss", loss, sync_dist=True)
    
    def get_state(self, obs):
        return obs["grid_raw"]
    
    def encode_state(self, obs):
        image = self.get_state(obs)
        # model forward() will encode the observation
        x, y = obs["agent_pos"]
        # d = obs["direction"] + 3  # +3 offset for original states for doors
        image[x, y] = np.array([10, 0, 0])  # [10,0,d] for agent+orientation
        image = rotate_state(obs["direction"], image)
        return torch.LongTensor(image.copy()).to(self.device)
    
    def encode_goal(self, goal):
        return self.tokenizer(
            goal, 
            add_special_tokens=False, 
            padding="max_length", 
            max_length=self.max_goal_length, 
            return_tensors="pt"
        )["input_ids"].long().to(self.device)
    
    def run_evaluation(self, split, eval_rollout):
        eval_horizon = EVAL_HORIZON_MAP[self.env_name]
        if split == "valid":
            seed_list = self.validation_seed_list
        elif split == "test":
            seed_list = self.test_seed_list

        steps_list, success_list = [], []
        for i in range(eval_rollout):

            with suppress_output():
                seed = seed_list[i]
                self.env.seed(seed)  # controll validation/test inference splits
                ini_obs = self.env.reset()

            goal = self.encode_goal(ini_obs["mission"])
            obs = self.encode_state(ini_obs)

            # prepare variables for model inputs
            obs, actions, goal = self.init_eval(obs, goal, self.model.obs_dim, self.model.goal_dim, eval_horizon)

            num_steps = 0
            for t in range(eval_horizon):
                logits = self.ar_step(t, obs, actions, goal)
                a = logits.argmax().cpu().numpy()
                new_obs, reward, done, _ = self.env.step(a)
                num_steps += 1
                
                if reward != 0:
                    # NOTE: BabyAI env returns non-zero reward when agent reahces the goal.
                    # NOTE: Reward is negative when num_steps > max_steps defined in env.
                    break

                next_obs = self.encode_state(new_obs)
                obs, actions = self.ar_step_end(t, next_obs, a, obs, actions)

            steps_list.append(num_steps)
            success_list.append(int(reward != 0))

        steps_list = np.array(steps_list).astype(np.int64)
        success_list = np.array(success_list).astype(np.int64)

        steps_over_all_list = steps_list
        steps_over_success_list = (steps_list * success_list).astype(np.int64)
        steps_over_success_list = steps_over_success_list[steps_over_success_list != 0]
        
        mean_steps_over_all = steps_over_all_list.mean()
        mean_steps_over_success = steps_over_success_list.mean()
        success_rate_over_all = success_list.mean()
        success_rate_within_steps = sum(steps_over_success_list < self.max_step_threshold) / eval_rollout
        
        self.log(f"{split}/mean_steps_over_all", mean_steps_over_all)
        self.log(f"{split}/mean_steps_over_success", mean_steps_over_success)
        self.log(f"{split}/success_rate_over_all", success_rate_over_all)
        self.log(f"{split}/success_rate_within_steps", success_rate_within_steps)

    def run_visualization(self, num_videos):
        eval_horizon = EVAL_HORIZON_MAP[self.env_name]
        
        for i in range(num_videos):
            
            images_list = []
            with suppress_output():
                self.env.seed(50000 + i)
                ini_obs = self.env.reset()

            goal = self.encode_goal(ini_obs["mission"])
            obs = self.encode_state(ini_obs)
            images_list.append(ini_obs["grid_rgb"])

            # prepare variables for model inputs
            obs, actions, goal = self.init_eval(obs, goal, self.model.obs_dim, self.model.goal_dim, eval_horizon)

            num_steps = 0
            for t in range(eval_horizon):
                logits = self.ar_step(t, obs, actions, goal)
                a = logits.argmax().cpu().numpy()
                new_obs, reward, done, _ = self.env.step(a)
                num_steps += 1
                
                images_list.append(new_obs["grid_rgb"])
                
                if reward != 0:
                    # NOTE: BabyAI env returns non-zero reward when agent reahces the goal.
                    # NOTE: Reward is negative when num_steps > max_steps defined in env.
                    break

                next_obs = self.encode_state(new_obs)
                obs, actions = self.ar_step_end(t, next_obs, a, obs, actions)
        
            images_list = np.array(images_list).transpose(0,3,1,2)
            wandb.log({f"visualizations/{ini_obs['mission']}": wandb.Video(images_list, fps=2, format="gif")})
            
    def test_step(self, batch, batch_idx):
        self.run_evaluation(split = "test", eval_rollout=108)
        self.run_visualization(num_videos=10)
    
    def on_validation_epoch_end(self):
        if not self.is_goal:
            return super().on_validation_epoch_end()

        if (self.current_epoch + 1) % self.eval_interval_epochs == 0:
            self.run_evaluation(split = "valid", eval_rollout=36)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.lr)
        return {
            "optimizer": optimizer,
        }

    # eval funcs
    def init_eval(self, ini_obs: Tensor, goal: Tensor, obs_dim: int, goal_dim:int, plan_horizon: int = None):
        # ini_obs: (obs_dim, )
        ini_obs = ini_obs.permute(2, 0, 1)  # H*W*C -> C*H*W
        observations = ini_obs.view(1, 1, 3, self.obs_height, self.obs_height)
        if OPTIONAL_LAST_STATE:
            last_obs = torch.zeros((1, 1, 3, self.obs_height, self.obs_height), dtype=observations.dtype).to(observations.device)
            observations = torch.cat((observations, last_obs), dim=1)
        
        # goal: (goal_dim, )
        goal = goal.view(1, -1)
        actions = torch.zeros(size=(1, 1), device=self.device, dtype=torch.long)  # [batch_sz, ctx_size]
        actions[0, 0] = 1  # BOS token
        return observations, actions, goal

    def make_mask(self, timestep: int):
        """
        uni-modality (obs/action) mask with no padding
        """
        if timestep < self.ctx_size:
            padding_mask = torch.zeros(size=(1, timestep + 1), dtype=torch.bool, device=self.device)
        else:
            padding_mask = torch.zeros(size=(1, self.ctx_size), dtype=torch.bool, device=self.device)
        return padding_mask

    def ar_step(self, timestep: int, observations: Tensor, actions: Tensor, goal: Tensor):
        padding_mask = self.make_mask(timestep)
        pred_a = self(observations, actions, goal, padding_mask)
        action = pred_a[0, -1][2:]  # ignore BOS and PAD tokens
        return action

    def ar_step_end(self, timestep: int, next_obs: Tensor, action: Tensor, obs_seq: Tensor, action_seq: Tensor):
        action = torch.LongTensor(np.array([action + 2])).view(1, -1).to(self.device)  # offset by 2 for BOS and PAD tokens
        new_action_seq = torch.cat([action_seq, action], dim=1)
        new_action_seq = new_action_seq[:, -self.ctx_size:]
        return obs_seq, new_action_seq
