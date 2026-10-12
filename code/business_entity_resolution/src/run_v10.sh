#!/usr/bin/env bash
# v10 chain (US / India rows of the final file): the v9 chain rerun with numbered streets kept as
# address tokens (BER_ORD_STREET=1: "18th Street" / "Eighth St" / "5th Cross" become a street token
# instead of a house-number candidate), in its own work directory (default <root>/work2).
# It ends with the final composition: US / India rows from v10, France rows from v9d (run_v9.sh),
# new US near-twin links (house numbers 1-30 apart) dropped -> output/matching_results.tsv and
# output/candidate_pairs.tsv.
set -e
cd "$(dirname "$0")"
PY=${PY:-python}
ROOT="${BER_ROOT:-$(cd ../../.. && pwd)}"
export BER_WORK=${BER_WORK_V10:-$ROOT/work2}
export BER_ORD_STREET=1
W="$BER_WORK"
mkdir -p "$W/logs"
for v in v4 v5; do [ -e "$W/$v" ] || cp -r legacy_members/$v "$W/$v"; done
step() { echo "$(date +%H:%M:%S) $1" >> "$W/logs/v10_chain.log"; }
step start
./run_limited.sh v10_build_dict $PY build_dict.py
./run_limited.sh v10_prep_train $PY prepare.py train
./run_limited.sh v10_prep_test  $PY prepare.py test
step prepared
BER_QCHUNK=50000 ./run_limited.sh v10_deep_sample $PY blocking.py train sample
./run_limited.sh v10_ranker $PY train_ranker.py
BER_QCHUNK=50000 MEM_MAX=11G ./run_limited.sh v10_block_train $PY blocking.py train
BER_QCHUNK=40000 MEM_MAX=11G ./run_limited.sh v10_block_test  $PY blocking.py test
step blocked
./run_limited.sh v10_decoy_dict $PY decoy_dict.py
MEM_MAX=9G ./run_limited.sh v10_decoy_mixture $PY decoy_mixture.py
MEM_MAX=9G ./run_limited.sh v10_decoy_unseen  $PY decoy_dict_unseen.py
./run_limited.sh v10_feat_train $PY features.py train
./run_limited.sh v10_feat_test  $PY features.py test
step features
BER_DROP=0.18 ./run_limited.sh v10_make_drop $PY make_drop.py
BER_SIM=1 MEM_MAX=13G ./run_limited.sh v10_train $PY train.py
BER_SIM=1 MEM_MAX=13G ./run_limited.sh v10_stage2 $PY stage2.py
step stage2
BER_SIM=1 BER_DECISION=thr MEM_MAX=13G ./run_limited.sh v10_predict $PY predict.py
BER_SIM=1 BER_DECISION=thr BER_UNSEEN=tx MEM_MAX=13G ./run_limited.sh v10_predict_tx $PY predict.py
BER_SIM=1 MEM_MAX=14G ./run_limited.sh v10_make_aug $PY make_decoy_aug.py 0.28
BER_WORK="$W/aug" BER_SIM=1 MEM_MAX=14G ./run_limited.sh v10_stage2_aug $PY stage2.py
cp "$W"/aug/lgb2_fold{0,1}_sim.txt "$W"/aug/model2_meta_sim.json "$W"/
rm -f "$W"/test_pred2_sim_tx.parquet
BER_SIM=1 BER_DECISION=thr BER_UNSEEN=tx MEM_MAX=13G ./run_limited.sh v10_predict_tx_aug $PY predict.py
step aug
BER_WORK="$W/aug" BER_SIM=1 MEM_MAX=14G ./run_limited.sh v10_stage3 $PY stage3.py
cp "$W"/aug/lgb3_fold{0,1}_sim.txt "$W"/aug/model3_meta_sim.json "$W"/
BER_SIM=1 BER_UNSEEN=tx MEM_MAX=15G ./run_limited.sh v10_stage3_test $PY stage3.py test
BER_WORK="$W/aug" BER_SIM=1 MEM_MAX=10G ./run_limited.sh v10_resid $PY resid_stack.py
cp "$W"/aug/resid_fold{0,1}_sim.txt "$W"/
step stage3
BER_SIM=1 MEM_MAX=14G ./run_limited.sh v10_final $PY make_final.py --name v10 \
  --members test_pred2_sim_tx.parquet --stage3 test_pred3_sim_tx.parquet --blend-country India US \
  --no-raise-near US --fr-role-fix --fr-collision --thr 0.66 --thr-seen 0.70 --rescue 0.55 --resid
step done
BER_GUARD_NEAR=US MEM_MAX=16G ./run_limited.sh v10_merge $PY merge_countries.py v9d v10 v10g US India
cp "$ROOT/output/matching_results_v10g.tsv" "$ROOT/output/matching_results.tsv"
cp "$ROOT/output/candidate_pairs_v10g.tsv" "$ROOT/output/candidate_pairs.tsv"
