# mRNAdesignwithRL

## RNAplay: Training & Fine-Tuning Scripts

This repository contains scripts for running and training the **RNAplay** environment using reinforcement learning with PyTorch.  

It has two main workflows:
1. **run_env.py** → runs the environment (episodes, simulation, degradation model, etc.)  
2. **train_model.py** → runs the training loop to optimize the model  

---

## Prerequisites

### Install Conda
If you don’t already have conda (or `mamba`) installed, install [Miniconda](https://docs.conda.io/en/latest/miniconda.html).  

Check if conda is available:
```bash
conda --version
```

---

### Create a Conda Environment

From the project folder, create and activate the environment:

```
# Create new env from the provided environment.yml
conda env create -f environment.yml -n rnplay

# Activate it
conda activate rnplay
```

---

## Additional Dependencies

Some external tools and libraries are required.

### 1. Install [RiboGraphViz](https://github.com/DasLab/RiboGraphViz)

```
git clone https://github.com/DasLab/RiboGraphViz
cd RiboGraphViz
pip install -r requirements.txt
python setup.py install
```

### 2. Install [DegScore](https://github.com/eternagame/DegScore)

```bash
git clone https://github.com/eternagame/DegScore
```

### 3. Install [LinearPartition](https://github.com/LinearFold/LinearPartition)

```bash
git clone https://github.com/LinearFold/LinearPartition
cd LinearPartition
make
```

### 4. Install [LinearFold](https://github.com/LinearFold/LinearFold)

```bash
git clone https://github.com/LinearFold/LinearFold
cd LinearFold
make
```

---

## Project Structure

Key files/folders in this repo:

* `run_env.py` → runs episodes in the environment
* `train_model.py` → training loop that consumes the state from `run_env.py`
* `environment.yml` → file describing dependencies for conda
* `run_env.sh`, `train_model.sh` → shell scripts to launch experiments
* `Agent.py`, `Network.py`, `Functions.py`, `env.py`, etc. → core Python modules
* `async_comm/` → folder used for communication between environment and training loop
* `logs/`, `models/`, `finetune_memory/` → output directories

---

## Get Data

For now all necessary data has been placed in the shared directory:

```
/data/backup/RNAplay/data
```

---

## How to Run

The training pipeline involves **two processes running at the same time**:

1. **Environment process** (`run_env.py`) generates episodes and writes to `async_comm/`.
2. **Training process** (`train_model.py`) consumes from `async_comm/` and updates model weights.

This setup will:

* Continuously read from the `async_comm/` folder (produced by `run_env.py`)
* Train the neural network using reinforcement learning
* Save updated weights back into `async_comm/trained_weights.bin`
* Write logs to `logs/`

---

## Workflow

You can run both processes manually, or use the provided bash scripts.

### Option 1: Manual Run

1. **Open 2 terminals** and `conda activate rnplay` in both.
2. In terminal 1: run the environment:

   ```bash
   python run_env.py [arguments...]
   ```
3. In terminal 2: run the trainer:

   ```bash
   python train_model.py [arguments...]
   ```
4. The two processes communicate through the `async_comm/` folder.

### Option 2: Use Bash Scripts

Example scripts are included:

* **`run_env.sh`**

  ```bash
  #!/bin/bash
  conda activate rnplay
  python run_env.py \
    --episodes 1 \
    --k 64 \
    --EPS_START 0.5 \
    --reduce_k_epoch 10 \
    --gamma 0.5 \
    --episode_length 12 \
    --reps_per 64 \
    --gpu_id 0 \
    --linearpartition_path ./../LinearPartition \
    --linearfold_path ./../LinearFold \
    --codon_table_path ../data/RNA_codons.csv \
    --codon_usage_table_path ../data/h_sapiens_9606.csv \
    --fasta_file_path ../data/GFP.fasta \
    --use_deberta_attention \
    --cai_weight 0.05 \
    --cai_metric_weight 0.05 \
    --degradation_reward \
    --degradation_weight 2 \
    --degradation_model_weight_path ../data/degradation_model_w_loop
  ```

* **`train_model.sh`**

  ```bash
  #!/bin/bash
  conda activate rnplay
  python train_model.py \
    --episodes 60 \
    --k 32 \
    --EPS_START 0.5 \
    --reduce_k_epoch 10 \
    --gamma 0.5 \
    --episode_length 12 \
    --reps_per 64 \
    --gpu_id 1 \
    --linearpartition_path ./../../../../LinearPartition \
    --linearfold_path ./../../../../LinearFold \
    --codon_table_path ../../../data/RNA_codons.csv \
    --codon_usage_table_path /home/grads/s/shujun/RNAplay/codon-usage-tables/codon_usage_data/tables/h_sapiens_9606.csv \
    --fasta_file_path ../../../data/pdb_6M0J.fasta \
    --use_deberta_attention \
    --cai_weight 0.05 \
    --pretrained_weight_path ../data/pretrained_weights.bin \
    --degradation_reward \
    --degradation_weight 2
  ```

To run them:

```bash
bash run_env.sh
bash train_model.sh
```

you can also ```nohup``` the bash scripts so you don't have to open 2 terminals.

---

## Argument Descriptions

Both `run_env.py` and `train_model.py` accept a set of command-line arguments via `argparse`.

### Core Training/Environment Args

* `--episodes` → Number of episodes to run (outer loop of environment).
* `--episode_length` → Number of steps per episode (sequence mutation steps).
* `--epochs_per_episode` → Training epochs per episode.
* `--reps_per` → Number of parallel sequences to run in each episode.
* `--k` → Maximum number of mutations allowed.
* `--reduce_k_epoch` → Frequency (in episodes) to reduce `k`.

### Reinforcement Learning Args

* `--EPS_START` → Initial exploration rate (epsilon-greedy).
* `--EPS_END` → Final exploration rate.
* `--gamma` → Discount factor for rewards.
* `--gamma_reduce` → Amount to reduce `gamma` after each `reduce_k_epoch`.
* `--memory_capacity` → Maximum number of transitions to store in replay memory.
* `--batch_size` → Batch size used during training.
* `--batch_size_update_epoch` → Epoch frequency to double batch size.
* `--max_batch_size` → Maximum allowed batch size.

### Model Args

* `--gpu_id` → Which GPU to use.
* `--pretrained_weight_path` → Path to pretrained model weights.
* `--use_deberta_attention` → Use DeBERTa-style self-attention in transformer.
* `--use_nt_input` → Whether to use nucleotide input (instead of amino acid input).
* `--data_parallel` → Enable `nn.DataParallel` for multi-GPU training.

### Optimization Args

* `--weight_decay` → Weight decay (L2 regularization) for optimizer.
* `--lr_scale` → Scaling factor for learning rate.
* `--dropout` → Dropout probability.
* `--ntoken` → Vocabulary size (usually 4 nucleotides).
* `--nclass` → Number of classes from the linear decoder.
* `--ninp` → Embedding dimension.
* `--nhead` → Number of attention heads.
* `--nhid` → Hidden dimension size.
* `--nlayers` → Number of transformer layers.

### Biological / Data Args

* `--codon_table_path` → CSV file mapping codons to amino acids.
* `--codon_usage_table_path` → Codon usage frequency table.
* `--fasta_file_path` → Input protein FASTA file.
* `--linearpartition_path` → Path to LinearPartition executable.
* `--linearfold_path` → Path to LinearFold executable.
* `--custom_protein_mask` → Optional mask for protein sequence.

### Rewards & Metrics

* `--optimization_direction` → `max`, `min`, or `target` (for structure optimization).
* `--target` → Target structure score when using `optimization_direction target`.
* `--cai_weight` → Weight of CAI during optimization.
* `--cai_metric_weight` → Weight of CAI when choosing best sequence.
* `--mld_metric_weight` → Weight of MLD when choosing best sequence.
* `--degradation_reward` → Whether to use degradation model reward.
* `--degradation_weight` → Weight of degradation score.
* `--degradation_model_weight_path` → Path to degradation model weights.

---

## Outputs

* **Logs:** CSV files in `logs/` containing training metrics (loss, performance, CAI, degradation score, etc.)
* **Models:** saved neural network checkpoints in `models/`
* **Best sequences:** written into `best_sequence.txt`
* **Async state:** saved in `async_comm/state.json`

---

## Helpful Tips for Beginners

* To **check available conda envs**:

  ```bash
  conda env list
  ```
* To **switch GPU**: change `--gpu_id 0` → `--gpu_id 1` etc.
* If you get `ModuleNotFoundError`, make sure you’re inside the right folder and environment:

  ```bash
  conda activate rnplay
  cd RNAplay/test_deberta/git_test/test
  ```
* To stop training, just press **Ctrl+C** in both terminals.
* You can edit `run_env.sh` or `train_model.sh` to change hyperparameters without typing long commands.

---

```
```
