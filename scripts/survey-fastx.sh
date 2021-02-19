#!/bin/bash
# Count all fasta/fastq files in the folder and export the number of reads for each file

# File naming format:
#     - contain 'fast[a/q]' for fasta/fastq files, or fast[a/q].gz for GNU zipped file
#     - contain '_L[lane]' to indicate line number
#     - contain `R[1/2]` to indicate forward (1) or reverse (2) read
#     Note: values in [...] will be extracted

# Usage:
#   survey-fastx.sh /path/to/fastx_dir /path/to/output

# Output format:
#   A TSV file contains columns: file lane R read
#     file - base file name
#     lane - lane number
#     r    - read direction (1 for forward, 0 for reverse)
#     read - number of reads
#
# NOTE:
#   this script uses GNU grep (not the grep from Mac coreutil),
#   if you are running on MacOS, ensure the GNU grep is installed


[ -z "$1" ] && FILE_DIR=$(pwd) || FILE_DIR=$1
[ -z "$2" ] && OUTPUT_DIR=$(pwd) || OUTPUT_DIR=$2

echo "Survey files in $FILE_DIR"
OUTPUT_FILE=$OUTPUT_DIR/sample_reads.tsv

FILES=$(find "$FILE_DIR" -type f -name '*fast*' | sort)
for fastx in $FILES
do
  if ! [ -f "$fastx" ]
    then
      echo "No fasta/fastq file found in $FILE_DIR"
      exit 1
  fi
  fastx_name=$(basename "$fastx")
  sample=${fastx_name/_L0*}
  lane=$(echo "$fastx_name" | grep -oP '(?<=_L)\d+')
  direction=$(echo "$fastx_name" | grep -oP '(?<=_R)\d')
  gz=$(echo "$fastx_name" | grep -oP 'gz')
  fasta=$(echo "$fastx_name" | grep -oP 'fasta')

  [ -z "$gz" ] && reads=$(wc -l "$fastx" | awk '{print $1}') || reads=$(gunzip -c "$fastx" | wc -l)
  [ -z "$fasta" ] && reads=$((reads/4)) || reads=$((reads/2))
  echo "$fastx_name:$reads"
  if ! [ -f "$OUTPUT_FILE" ]; then
    printf "file\t%s%sreads\n" \
      "$([ -z "$lane" ] && echo "" || printf "lane\t")" \
      "$([ -z "$direction" ] && echo "" || printf "R\t")" \
      >> "$OUTPUT_FILE"
  fi
  printf "$sample\t%s%s$reads\n" \
    "$([ -z "$lane" ] && echo "" || printf "%s\t" "$lane")" \
    "$([ -z "$direction" ] && echo "" || printf "%s\t" "$direction")" \
    >> "$OUTPUT_FILE"
done

echo "Results saved to $OUTPUT_FILE"
