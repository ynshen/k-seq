#!/bin/bash -l

CORE=6
OUTPUT_BASE='/mnt/storage/projects/k-seq/working/simu_data/least_squared/'
SAMPLE_NAME='on_doped_s1000_d40'
OUTPUT_NAME='on_doped_s1000_d40'

python estimator_ls_byo_runner.py \
    --pkg_path /home/yuning/research/k-seq/src/pkg/ \
    --simu_data /mnt/storage/projects/k-seq/datasets/simulated/$SAMPLE_NAME/ \
    --fit_partial -1 \
    --bootstrap_num 500 \
    --bs_record_num 100 \
    --bs_method 'data' \
    --core_num $CORE \
    --exclude_zero \
    --inverse_weight \
    --output_dir $OUTPUT_BASE/$OUTPUT_NAME/ \
    &> $OUTPUT_NAME.out
