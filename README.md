# Emergence of Abstract State Representations in Embodied Sequence Modeling
This repository provides the code for pretraining and probing embodied sequence models in *Emergence of Abstract State Representations in Embodied Sequence Modeling*, to be present at EMNLP 2023.

## Table of Contents
1. [Installation](#installation)
2. [Data Preparation](#data-preparation)
3. [Pretraining](#pretraining)
4. [Probing Data Precomputation](#probing-data-precomputation)
5. [Probing with Precomputed Internal Activations](#probing-with-precomputed-internal-activations)
6. [How to Cite](#how-to-cite)

## Installation and Data Preparation
1. `git clone` this repo with `--recurse-submodules` to git clone with submodules.

2. Install `babyai`, `gym` and `gym-minigrid` submodules.
```
cd babyai
conda env create -f environment.yaml

source activate babyai
cd ../gym-minigrid
pip install -e .

cd ../babyai
pip install -e .

cd ..
pip install -e .
```

3. Install remainining dependencies.
```
conda install pytorch==1.10.2 cudatoolkit=11.3 -c pytorch -c conda-forge

pip install wandb==0.14.2 transformers==4.18.0 hydra-core==1.3.2 pytorch-lightning==1.5.0 matplotlib==3.3.4
```


## Data Preparation

1. Download the generated BabyAI GoToLocal and MiniBossLevel trajectories for pretraining and probing from [this link](https://drive.google.com/drive/folders/14R8CTrXPYiqr26tRfHtXk5nkdiJ-Zo-Y?usp=sharing). After finishitng the downloads, create `data` folder under the root directory of the repo and unzip data files under your `data` folder.
```
mkdir data
mv *.zip data/

cd data
unzip *.zip
```

2. The following paths in `/config/*.yaml` need to be specified before running experiments. **NOTE: All paths are absolute paths".**
```
`data_root`:  # Path of babyai-related data folder. E.g., ${PATH_TO_REPO}/data

`ckpt_root`:  # Path of checkpoint folder based on `wandb`. E.g., ${PATH_TO_REPO}/wandb_logs/babyai

`save_dir_root``:  # Path of training output folder based on `wandb`. Eg., ${PATH_TO_REPO}/wandb_logs/outputs
```


## Pretraining 
Pretraining experiments with the language instruction:
```shell
python abstract_state_seqmodel/train.py level_name=${level_name} \
    devices=[0] \
    seed=${seed} \
    model=${model} \
    max_goal_length=${max_goal_length} \
    epochs=${epoch} \
    lr=${lr} \
    is_goal=True
```
Pretraining experiments without the language instruction:
```shell
python abstract_state_seqmodel/train_without_goal.py level_name=${level_name} \
    devices=[0] \
    seed=${seed} \
    model=${model} \
    max_goal_length=${max_goal_length} \
    epochs=${epoch} \
    lr=${lr} \
    is_goal=False \
```
We implemented two types of sequence models: Complete-State and Missing-State models:
- To run pretraining experiments on Complete-State models, pass `causal_all` to the "model" argument.
- To run pretraining experiments on Missing-State models, pass `causal_withold_state` to the "model" argument.

We supported two levels in BabyAI: `GoToLocal` and `MiniBossLevel`, and their corresponding `max_goal_length`s are listed below:

| Level           | GoToLocal | MiniBossLevel |
|-----------------|-----------|---------------|
| Max Goal Length | 10        | 50            |


## Probing Data Precomputation
To accelerate probing experiments, we precompute the internal activations from the pre-trained/randomly initialized sequence models using the following command:
```shell
python abstract_state_seqmodel/precompute_probe_data.py level_name=${level_name} \
    devices=[0] \
    model="probe_model" \
    probe_layer=${probe_layer} \
    model_seed=${model_seed} \
    is_randomly_initialized=${is_randomly_initialized} \
    is_goal=${is_goal} \
    model_type=${model_type} \
    pretrained_model_config.max_goal_length=${max_goal_length} \
    seqmodel_ckpt_path=${seqmodel_ckpt_path} \
```
Arguments:
- `probe_layer`: the *i*-th layer we are extracting the internal activations from. The range of *i* is [0, 5].
- `is_randomly_initialized`: if `True`, then we compute internal activations from a randomly initialized sequence model. Otherwise, we extract internal activations from a pre-trained sequence model.
- `model_seed` is used to when precomputing internal activations from randomly initialized models.
- `seqmodel_ckpt_path` is the absolute checkpoint path for the pre-trained model, and is used when `is_randomly_initialized` is `False`.


## Probing with Precomputed Internal Activations
To run probing experiments with internal activations precomputed from `probe_layer`, we use the following command:
```shell
python abstract_state_seqmodel/train_precompute_probe.py level_name=${level_name} \
    devices=[0] \
    model="probe_model" \
    probe_type=${probe_type} \
    probe_layer=${probe_layer} \
    seed=${seed} \
    is_goal=${is_goal} \
    is_randomly_initialized=${is_randomly_initialized} \
    model_type=${model_type} \
    pretrained_model_config.max_goal_length=${max_goal_length}
```

We implemented 5 different probe types/metrics: `agent_loc`, `board_type`, `board_color`, `neighbor_type`, `neighbor_color`.


## How to Cite
```
@inproceedings{yun2023emergence,
	title={Emergence of Abstract State Representations in Embodied Sequence Modeling},
	author={Tian Yun and Zilai Zeng and Kunal Handa and Ashish V Thapliyal and Bo Pang and Ellie Pavlick and Chen Sun},
	booktitle={Proceedings of the Conference on Empirical Methods in Natural Language Processing},
	year={2023}
}
```