#!/usr/bin/env bash
# Whole pipeline, from the raw training / test data to output/matching_results.tsv and
# output/candidate_pairs.tsv (about 7 h on 16 CPU cores, <= 16 GB RAM per step).
set -e
cd "$(dirname "$0")"
./run_v9.sh
./run_v10.sh
