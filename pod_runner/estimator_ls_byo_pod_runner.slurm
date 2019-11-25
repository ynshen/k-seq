#!/bin/bash -l
# ask for 40 cores on 1 node
#SBATCH --nodes=1 --ntasks-per-node=40
#SBATCH --time=48:00:00
#SBATCH --mail-user=yuningshen@ucsb.edu
#SBATCH --mail-type=start,end

cd $SLURM_SUBMIT_DIR

CORE=6
OUTPUT_BASE='/home/yshen/k-seq/byo_doped/point-est-bounded_2019-11-23/'
# SAMPLE_NAME='on_doped_s1000_d40'
# OUTPUT_NAME='byo-doped_no-zero-True_inverse-weight-False'

python estimator_ls_byo_runner.py \
    --pkg_path /home/yshen/k-seq/ \
    --seq_table /home/yshen/k-seq/datasets/byo_doped.pkl \
    --table_name table_filtered_reacted_frac_spike_in \
    --fit_partial -1 \
    --bootstrap_num 0 \
    --bs_record_num 0 \
    --bs_method 'data' \
    --core_num $CORE \
    --output_dir $OUTPUT_BASE \
#    &> $OUTPUT_NAME.out

