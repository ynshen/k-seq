#!/bin/sh

OUTPUT=./react-frac-test
N_SEQ=4
N_THREAD=8

# standard set: enriched pool x-data with 0.2 relative error
./simu-reacted-frac.py \
    --n_seq $N_SEQ \
    --x_data 0.000250 0.000050 0.000010 0.000002 \
    --replicates 3 \
    --rel_err 0.2 \
    --log_A \
    --output $OUTPUT/standard-relErr02 \
    --n_thread $N_THREAD \
    --n_bootstrap 100 \
    --n_converge 20 \
    --seed 23

# additional replicate: enriched pool x-data, 4 replicates, 0.2 relative error
./simu-reacted-frac.py \
    --n_seq $N_SEQ \
    --x_data 0.000250 0.000050 0.000010 0.000002 \
    --replicates 4 \
    --rel_err 0.2 \
    --log_A \
    --output $OUTPUT/additional-relErr02 \
    --n_thread $N_THREAD \
    --n_bootstrap 100 \
    --n_converge 20 \
    --seed 23

# extended set: + 1250 uM, 3 replicates, 0.2 relative error
./simu-reacted-frac.py \
    --n_seq $N_SEQ \
    --x_data 0.001250 0.000250 0.000050 0.000010 0.000002 \
    --replicates 3 \
    --rel_err 0.2 \
    --log_A \
    --output $OUTPUT/extended-relErr02 \
    --n_thread $N_THREAD \
    --n_bootstrap 100 \
    --n_converge 20 \
    --seed 23

# error effects
./simu-reacted-frac.py \
    --n_seq $N_SEQ \
    --x_data 0.001250 0.000250 0.000050 0.000010 0.000002 \
    --replicates 3 \
    --rel_err 0.0 \
    --log_A \
    --output $OUTPUT/extended-relErr0 \
    --n_thread $N_THREAD \
    --n_bootstrap 100 \
    --n_converge 20 \
    --seed 23

./simu-reacted-frac.py \
    --n_seq $N_SEQ \
    --x_data 0.001250 0.000250 0.000050 0.000010 0.000002 \
    --replicates 3 \
    --rel_err 0.5 \
    --log_A \
    --output $OUTPUT/extended-relErr05 \
    --n_thread $N_THREAD \
    --n_bootstrap 100 \
    --n_converge 20 \
    --seed 23

./simu-reacted-frac.py \
    --n_seq $N_SEQ \
    --x_data 0.001250 0.000250 0.000050 0.000010 0.000002 \
    --replicates 3 \
    --rel_err 1.0 \
    --log_A \
    --output $OUTPUT/extended-relErr10 \
    --n_thread $N_THREAD \
    --n_bootstrap 100 \
    --n_converge 20 \
    --seed 23
