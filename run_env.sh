python run_env.py --episodes 1 --k 64 --EPS_START 0.5 --reduce_k_epoch 10 --gamma 0.5 --gamma_reduce 0 --episode_length 12 --reps_per 64 --gpu_id 0 --linearpartition_path ./../LinearPartition --MAX_THRE 24 \
--codon_table_path ../data/RNA_codons.csv --linearfold_path ./../LinearFold \
--codon_usage_table_path ../data/h_sapiens_9606.csv \
--fasta_file_path ../data/GFP.fasta  --optimization_direction max --epochs_per_episode 5 --use_deberta_attention \
--optimization_direction max --cai_weight 0.05 --cai_metric_weight 0.05 \
--pretrained_weight_path ../../RNAplay_async_pretrain48/async_comm/trained_weights.bin \
--degradation_reward --degradation_weight 2 --degradation_model_weight_path ../data/degradation_model_w_loop