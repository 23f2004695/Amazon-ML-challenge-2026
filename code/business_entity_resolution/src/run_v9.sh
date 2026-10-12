#!/usr/bin/env bash
# v9 chain (France rows of the final file), on top of the base chain (run_v6.sh) and the frozen
# v4 / v5 stage-2 predictions (legacy_members/, used by the US / India blend):
#   1. synthetic multi-record decoy families (US) -> <work>/aug; stage 2 retrained there (v8);
#   2. test stage 2 with the v8 models (label-free vocabulary for unseen countries);
#   3. stage 3: uncertain region re-scored with group context from p2, house-number arithmetic,
#      noise markers and raw-name formatting (trained on <work>/aug, same query folds);
#   4. residual stacker on the final US / India probabilities;
#   5. decision (make_final.py): stage-3 / v5 / v4 blend for US and India, threshold 0.70,
#      rescue of link-less S1 (p > 0.55), US nearby-house-number near-twins never raised above
#      stage 2, residual stacker (removals; additions only for India); France: v8 stage 2 +
#      role-aware vocabulary fix + chance house-number collision fix, threshold 0.66.
#   -> output/matching_results_v9d.tsv, output/candidate_pairs_v9d.tsv
set -e
cd "$(dirname "$0")"
PY=${PY:-python}
W="${BER_WORK:-${BER_ROOT:-$(cd ../../.. && pwd)}/work}"
mkdir -p "$W"
for v in v4 v5; do [ -e "$W/$v" ] || cp -r legacy_members/$v "$W/$v"; done
./run_v6.sh
BER_SIM=1 MEM_MAX=14G ./run_limited.sh v8_make_aug $PY make_decoy_aug.py 0.28
BER_WORK="$W/aug" BER_SIM=1 MEM_MAX=14G ./run_limited.sh v8_stage2 $PY stage2.py
cp "$W"/aug/lgb2_fold{0,1}_sim.txt "$W"/aug/model2_meta_sim.json "$W"/
rm -f "$W"/test_pred2_sim_tx.parquet
BER_SIM=1 BER_DECISION=thr BER_UNSEEN=tx MEM_MAX=13G ./run_limited.sh v8_predict_tx $PY predict.py
BER_WORK="$W/aug" BER_SIM=1 MEM_MAX=14G ./run_limited.sh v9_stage3 $PY stage3.py
cp "$W"/aug/lgb3_fold{0,1}_sim.txt "$W"/aug/model3_meta_sim.json "$W"/
BER_SIM=1 BER_UNSEEN=tx MEM_MAX=15G ./run_limited.sh v9_stage3_test $PY stage3.py test
BER_WORK="$W/aug" BER_SIM=1 MEM_MAX=10G ./run_limited.sh v9_resid $PY resid_stack.py
cp "$W"/aug/resid_fold{0,1}_sim.txt "$W"/
BER_SIM=1 MEM_MAX=14G ./run_limited.sh v9_final $PY make_final.py --name v9d \
  --members test_pred2_sim_tx.parquet --stage3 test_pred3_sim_tx.parquet --blend-country India US \
  --no-raise-near US --fr-role-fix --fr-collision --thr 0.66 --thr-seen 0.70 --rescue 0.55 --resid
