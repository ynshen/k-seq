#!/bin/bash -l


CORE=6
OUTPUT_BASE='/mnt/storage/projects/k-seq/working/simu_data/least_squared'
SAMPLE_NAME='on_doped_s1000_d40_data_test'

python estimator_ls_byo_runner.py \
    --pkg_path /home/yuning/research/k-seq/src \
    --simu_data /mnt/storage/projects/k-seq/datasets/simulated/on_doped_s1000_d40/ \
    --fit_partial 10 \
    --bootstrap_num 10 \
    --bs_record_num 5 \
    --bs_method 'data' \
    --core_num $CORE \
    --output_dir $OUTPUT_BASE/$SAMPLE_NAME \
    &> $SAMPLE_NAME.log
