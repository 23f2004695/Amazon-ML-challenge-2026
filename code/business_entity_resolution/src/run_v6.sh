#!/usr/bin/env bash
# Base chain (v6): Indic token dictionary -> normalization (OCR digit repair, unit-aware house
# numbers, spelled ordinals) -> learned blocking re-ranker -> decoy dictionaries (train +
# label-free unseen-country) -> features -> withheld-S1 simulation -> stage 1 (20% of queries)
# -> stage 2 -> test predictions (train vocabulary everywhere, and label-free vocabulary for
# unseen countries). Writes to $BER_WORK (default <root>/work).
set -e
cd "$(dirname "$0")"
PY=${PY:-python}
W="${BER_WORK:-${BER_ROOT:-$(cd ../../.. && pwd)}/work}"
./run_limited.sh build_dict $PY build_dict.py
./run_limited.sh v6_prep_train $PY prepare.py train
./run_limited.sh v6_prep_test  $PY prepare.py test
BER_QCHUNK=50000 ./run_limited.sh v6_deep_sample $PY blocking.py train sample
./run_limited.sh v6_ranker $PY train_ranker.py
rm -rf "$W/train_cand" "$W/test_cand"
BER_QCHUNK=50000 MEM_MAX=11G ./run_limited.sh v6_block_train $PY blocking.py train
BER_QCHUNK=40000 MEM_MAX=11G ./run_limited.sh v6_block_test  $PY blocking.py test
./run_limited.sh v6_decoy_dict $PY decoy_dict.py
MEM_MAX=9G ./run_limited.sh v6_decoy_mixture $PY decoy_mixture.py
MEM_MAX=9G ./run_limited.sh v6_decoy_unseen  $PY decoy_dict_unseen.py
rm -rf "$W/train_feat" "$W/test_feat" "$W/train_drop_feat"
./run_limited.sh v6_feat_train $PY features.py train
./run_limited.sh v6_feat_test  $PY features.py test
BER_DROP=0.18 ./run_limited.sh v6_make_drop $PY make_drop.py
BER_SIM=1 MEM_MAX=13G ./run_limited.sh v6_train $PY train.py
BER_SIM=1 MEM_MAX=13G ./run_limited.sh v6_stage2 $PY stage2.py
rm -f "$W"/test_pred1_sim*.parquet "$W"/test_pred2_sim*.parquet
BER_SIM=1 BER_DECISION=thr MEM_MAX=13G ./run_limited.sh v6_predict $PY predict.py
BER_SIM=1 BER_DECISION=thr BER_UNSEEN=tx MEM_MAX=13G ./run_limited.sh v6_predict_tx $PY predict.py
