# mRNAdesignwithRL
# RNAplay: Training & Fine-Tuning Scripts

This repository contains scripts for running and training the **RNAplay** environment using reinforcement learning with PyTorch.  

It has two main workflows:
1. **run_env.py** → runs the environment (episodes, simulation, degradation model, etc.)  
2. **train_model.py** → runs the training loop to optimize the model  

---

## 1. Prerequisites

### Install Conda
If you don’t already have conda (or `mamba`) installed, install [Miniconda](https://docs.conda.io/en/latest/miniconda.html).  

Check if conda is available:
```bash
conda --version
Create a Conda Environment
From the project folder, create and activate the environment:

bash
Copy
Edit
cd RNAplay/test_deberta/git_test/test

# Create new env from the provided environment.yml
conda env create -f environment.yml -n rnplay

# Activate it
conda activate rnplay
If you want to install manually without the .yml file:

bash
Copy
Edit
conda create -n rnplay python=3.10
conda activate rnplay
pip install -r requirements.txt   # if requirements file exists
2. Project Structure
Key files/folders in this repo:

run_env.py → runs episodes in the environment

train_model.py → training loop that consumes the state from run_env.py

environment.yml → file describing dependencies for conda

run.sh, run_env.sh, train_model.sh → shell scripts to launch experiments

Agent.py, Network.py, Functions.py, env.py, etc. → core Python modules

async_comm/ → folder used for communication between environment and training loop

logs/, models/, finetune_memory/ → output directories

3. Running the Environment
Example command:

bash
Copy
Edit
python run_env.py \
  --episodes 1 \
  --k 64 \
  --EPS_START 0.5 \
  --reduce_k_epoch 10 \
  --gamma 0.5 \
  --gamma_reduce 0 \
  --episode_length 12 \
  --reps_per 64 \
  --gpu_id 0 \
  --linearpartition_path ./../LinearPartition \
  --MAX_THRE 24 \
  --codon_table_path ../data/RNA_codons.csv \
  --linearfold_path ./../LinearFold \
  --codon_usage_table_path ../data/h_sapiens_9606.csv \
  --fasta_file_path ../data/GFP.fasta \
  --optimization_direction max \
  --epochs_per_episode 5 \
  --use_deberta_attention \
  --cai_weight 0.05 \
  --cai_metric_weight 0.05 \
  --pretrained_weight_path ../../RNAplay_async_pretrain48/async_comm/trained_weights.bin \
  --degradation_reward \
  --degradation_weight 2 \
  --degradation_model_weight_path ../data/degradation_model_w_loop
What this does:

Runs 1 episode of optimization

Uses pretrained weights if provided

Applies degradation reward with weight 2

Reads codon usage and fasta sequence from ../data/

4. Training the Model
In a separate terminal window (also with the environment activated):

bash
Copy
Edit
python train_model.py \
  --episodes 60 \
  --k 32 \
  --EPS_START 0.5 \
  --reduce_k_epoch 10 \
  --gamma 0.5 \
  --gamma_reduce 0 \
  --episode_length 12 \
  --reps_per 64 \
  --gpu_id 1 \
  --linearpartition_path ./../../../../LinearPartition \
  --MAX_THRE 24 \
  --codon_table_path ../../../data/RNA_codons.csv \
  --linearfold_path ./../../../../LinearFold \
  --codon_usage_table_path /home/grads/s/shujun/RNAplay/codon-usage-tables/codon_usage_data/tables/h_sapiens_9606.csv \
  --fasta_file_path ../../../data/pdb_6M0J.fasta \
  --optimization_direction max \
  --epochs_per_episode 3 \
  --use_deberta_attention \
  --cai_weight 0.05 \
  --pretrained_weight_path ../data/pretrained_weights.bin \
  --degradation_reward \
  --degradation_weight 2
This will:

Continuously read from the async_comm/ folder (produced by run_env.py)

Train the neural network using reinforcement learning

Save updated weights back into async_comm/trained_weights.bin

Write logs to logs/

5. Workflow
Open 2 terminals and conda activate rnplay in both.

In terminal 1: run the environment (run_env.py).

In terminal 2: run the trainer (train_model.py).

The two processes communicate through the async_comm/ folder.

6. Outputs
Logs: CSV files in logs/ containing training metrics (loss, performance, CAI, degradation score, etc.)

Models: saved neural network checkpoints in models/

Best sequences: written into best_sequence.txt

Async state: saved in async_comm/state.json

7. Helpful Tips for Beginners
To check available conda envs:

bash
Copy
Edit
conda env list
To switch GPU: change --gpu_id 0 → --gpu_id 1 etc.

If you get ModuleNotFoundError, make sure you’re inside the right folder and environment:

bash
Copy
Edit
conda activate rnplay
cd RNAplay/test_deberta/git_test/test
To stop training, just press Ctrl+C in both terminals.

You can edit run.sh or train_model.sh to avoid typing long commands.