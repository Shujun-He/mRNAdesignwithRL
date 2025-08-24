python train_model.py --episodes 60 --k 32 --EPS_START 0.5 --reduce_k_epoch 10 --gamma 0.5 --gamma_reduce 0 --episode_length 12 --reps_per 64 --gpu_id 1 --linearpartition_path ./../../../../LinearPartition --MAX_THRE 24 \
--codon_table_path ../../../data/RNA_codons.csv --linearfold_path ./../../../../LinearFold \
--codon_usage_table_path /home/grads/s/shujun/RNAplay/codon-usage-tables/codon_usage_data/tables/h_sapiens_9606.csv \
--fasta_file_path ../../../data/pdb_6M0J.fasta  --optimization_direction max --epochs_per_episode 3 --use_deberta_attention \
--optimization_direction max --cai_weight 0.05 \
--pretrained_weight_path ../data/pretrained_weights.bin \
--degradation_reward --degradation_weight 2