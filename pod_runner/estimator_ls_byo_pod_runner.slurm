#!/bin/bash -l
# ask for 40 cores on 1 node
#SBATCH --nodes=1 --ntasks-per-node=40
#SBATCH --time=48:00:00
#SBATCH --mail-user=yuningshen@ucsb.edu
#SBATCH --mail-type=start,end

cd $SLURM_SUBMIT_DIR

CORE=40
SAMPLE_NAME='byo_doped.pkl'
TABLE_NAME='table_filtered_reacted_frac_spike_in'
OUTPUT_BASE='/home/yshen/k-seq/results/byo-doped_2019-10/point-est-bounded_2019-11-23'
OUTPUT_NAME=''
OUTPUT_PREFIX=''

# Avaiable flags
# --simu_data
# --seq_table
# --table_name
# --exclude_zero
# --inverse_weight

python /home/yshen/k-seq/clt/estimator_ls_byo_runner.py \
    --pkg_path /home/yshen/k-seq/src/ \
    --seq_table /home/yshen/k-seq/datasets/$SAMPLE_NAME \
    --table_name $TABLE_NAME \
    --fit_partial -1 \
    --bootstrap_num 0 \
    --bs_record_num 0 \
    --bs_method 'data' \
    --inverse_weight \
    --core_num $CORE \
    --output_dir $OUTPUT_BASE/$OUTPUT_PREFIX$OUTPUT_NAME \

cp ./estimator_ls_byo_pod_runner.slurm $OUTPUT_BASE/

