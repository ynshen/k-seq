#!/bin/bash

R1L1='/mnt/storage/uli/hiSeq/UMA_S1_L001_R1_001.fastq.gz'
R2L1='/mnt/storage/uli/hiSeq/UMA_S1_L001_R2_001.fastq'
base='UMA_S1_L001_R1_001_default'
echo "Joining $base lane 1..."
pandaseq -f $R1L1 -r $R2L1 -F -p ATGTGAGACCGAGA -q GCTGGAGCTTAACT \
        -w /mnt/storage/uli/hiSeq/$base.joined.fastq -t 0.6 -T 6  
