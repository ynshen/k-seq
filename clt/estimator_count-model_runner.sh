#!/bin/bash

DATA_PATH='/mnt/storage/projects/k-seq/datasets/simulated/'
SAMPLE_NAME='on_doped_s10_d40'
OUTPUT_PATH='/mnt/storage/projects/k-seq/working/simu_data/count_model_mle/cvxpy_scs/'
OUTPUT_NAME='on_doped_s10_d40'

# Available flags
# --seq_table and --table_name
# --input_pools
# --simu_data_path

python estimator_count-model.py \
  --simu_data_path $DATA_PATH/$SAMPLE_NAME \
  --output_path $OUTPUT_PATH/$OUTPUT_NAME