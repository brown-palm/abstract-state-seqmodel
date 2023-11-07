import gym
import os

import hydra
import pickle
import numpy as np
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
import torch

from data.internal_activation_data import InternalActivationDataset, preprocess_babyai_episodes_from_path, OBS_HEIGHT_MAP, get_symbolic_images_from_bytes
import babyai.utils as utils

OPTIONAL_LAST_STATE = True
device = "cuda" if torch.cuda.is_available() else "cpu"


def load_pretrained_ckpt(cfg):
    if cfg.is_randomly_initialized:
        env = gym.make(cfg.env_name)
        obs_dim = 768
        action_dim = env.action_space.n + 2  # for BOS and PAD tokens
        
        if cfg.model_type == "causal_all":
            from vlm_rule_learning.model.causalmtm_all import CausalTransformer
            
        elif cfg.model_type == "causal_withold_state":
            from vlm_rule_learning.model.causalmtm_withold_state import CausalTransformer
        
        pl.seed_everything(cfg.model_seed)
        
        model = CausalTransformer(
            obs_dim=obs_dim, 
            action_dim=action_dim, 
            goal_dim=30522,  # WARNING: hard-coded tokenizer.vocab_size
            ctx_size=cfg.pretrained_model_config.ctx_size, 
            max_goal_length=cfg.pretrained_model_config.max_goal_length, 
            obs_type=cfg.obs_type, 
            config=cfg.pretrained_model_config.model_config,
            level_name=cfg.level_name
        )

    else:
        ckpt_path = cfg.seqmodel_ckpt_path  # absolute path
        if cfg.model_type == "causal_all":
            from vlm_rule_learning.model.causalmtm_all import CausalforMTM
            model = CausalforMTM.load_from_checkpoint(ckpt_path, data_root=cfg.data_root, is_goal=cfg.is_goal).model
        
        elif cfg.model_type == "causal_withold_state":
            from vlm_rule_learning.model.causalmtm_withold_state import CausalforMTM
            model = CausalforMTM.load_from_checkpoint(ckpt_path, data_root=cfg.data_root, is_goal=cfg.is_goal).model
        
    return model


def precompute_and_save(save_path: str, dataset: InternalActivationDataset, device, is_goal):
    subfile_cnt = 0
    precompute_data = []
    for idx in tqdm(range(len(dataset.traj_indices)), total=len(dataset.traj_indices)):
        batch = dataset.get_batch_item(idx)
        observations, actions, label_actions, goals, valid_length, padding_mask, agent_pos, agent_dir = batch
        epi_idx, start, end = dataset.traj_indices[idx]

        observations = torch.LongTensor(observations).unsqueeze(0).to(device)
        actions = torch.LongTensor(actions).unsqueeze(0).to(device)
        goals = torch.LongTensor(goals).unsqueeze(0).to(device)
        padding_mask = torch.BoolTensor(padding_mask).unsqueeze(0).to(device)

        if not is_goal:
            goals = torch.zeros_like(goals).long().to(device)

        # Predict hidden features
        with torch.no_grad():
            all_hidden_features = dataset.pretrained_model(observations, actions, goals, padding_mask, probe_layer=dataset.probe_layer).detach().cpu().numpy()  # [1, ctx_size, embed_dim]

        target_step_id = np.random.choice(range(1, valid_length))  # Get rid of sampling the initial state
        hidden_feature = all_hidden_features[0, target_step_id]  # [embed_dim]

        # Generate labels
        epi_length = dataset.epi_buffers["episode_lengths"][epi_idx]
        obs_height = OBS_HEIGHT_MAP[dataset.obs_type][dataset.env_name.split("-")[1]]
        curr_observations = np.zeros((dataset.max_episode_length, 3, obs_height, obs_height), dtype=np.float32)
        curr_observations[:epi_length] = get_symbolic_images_from_bytes(dataset.epi_buffers["observations"][epi_idx])  # [episode_len, 3, H, W]
        target_obs = curr_observations[start:end][target_step_id].transpose(1,2,0)  # [H,W,3], withold_state has no access to future obs

        precompute_item = {
            "traj_id": epi_idx,
            "start, end": (start, end),
            "target_step_id": target_step_id,
            "hidden_feature": hidden_feature.copy(),
            "target_obs": target_obs.copy()
        }

        precompute_data.append(precompute_item)

        if ((idx + 1) % 100000 == 0) or (idx == len(dataset.traj_indices) - 1):
            subfile_save_path = os.path.join(save_path, f"subfile_{subfile_cnt}.pkl")
            save_dir = os.path.dirname(subfile_save_path)
            if not os.path.isdir(save_dir):
                os.makedirs(save_dir, exist_ok=True)
            
            with open(subfile_save_path, "wb") as f:
                pickle.dump(precompute_data, f)
            print(f"Save to {subfile_save_path}", flush=True)
            subfile_cnt += 1
            precompute_data = []
            


@hydra.main(version_base=None, config_path="../config", config_name="data_precompute")
def main(cfg: DictConfig):
    assert cfg.level_name in ["GoToLocal", "MiniBossLevel"]
    cfg.env_name = f"BabyAI-{cfg.level_name}-v0"
    pretrained_model = load_pretrained_ckpt(cfg).to(device)
    pretrained_model.eval()
    
    pl.seed_everything(cfg.data_seed)

    model_type = f"{cfg.model_type}" +  ("_no_goal" if not cfg.is_goal else "") + ("_random_init" if cfg.is_randomly_initialized else "")
    dataset_split_size = {"train": cfg.train_size, "valid": cfg.valid_size, "test": cfg.test_size}

    max_episode_length = cfg.pretrained_model_config.max_episode_length
    max_goal_length = cfg.pretrained_model_config.max_goal_length
    ctx_size = cfg.pretrained_model_config.ctx_size

    for split, split_size in dataset_split_size.items():
        print(f"Pre-computing probing data for {model_type}-layer{cfg.probe_layer}-{split}", flush=True)
        dataset_path_split = os.path.join(cfg.data_root, cfg.level_name, cfg.obs_type, f"{split}.pkl")
        epi_buffer_split = preprocess_babyai_episodes_from_path(dataset_path_split, max_episode_length, max_goal_length, cfg.obs_type, ctx_size, number=split_size)
        dataset = InternalActivationDataset(pretrained_model, cfg.probe_type, cfg.probe_layer, cfg.env_name, epi_buffer_split, max_episode_length, max_goal_length, ctx_size, cfg.model_type, device)

        # precompute and save
        # skip processing probe_type and directly save target_obs
        save_path = os.path.join(cfg.data_root, f"precompute_seed-{cfg.data_seed}", cfg.level_name, model_type, f"probe_layer-{cfg.probe_layer}", split)
        precompute_and_save(save_path, dataset, device=device, is_goal=cfg.is_goal)
        


if __name__ == "__main__":
    main()