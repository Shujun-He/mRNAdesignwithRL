python pretrain.py --episodes 50 --reps_per 240 --k 8 --gpu_id 0 --linearpartition_path ./../../../LinearPartition --MAX_THRE 96 --epochs_per_episode 3 --batch_size 512 --EPS_START 0.75 \
--codon_table_path ../../data/RNA_codons.csv --linearfold_path ./../../../LinearFold \
--codon_usage_table_path /home/grads/s/shujun/RNAplay/codon-usage-tables/codon_usage_data/tables/h_sapiens_9606.csv \
--fasta_file_path ../../data/pdb_6M0J.fasta  --optimization_direction max --cai_weight 0 --use_deberta_attention

#--cai_weight 0.1 --degradation_reward --degradation_weight 1
