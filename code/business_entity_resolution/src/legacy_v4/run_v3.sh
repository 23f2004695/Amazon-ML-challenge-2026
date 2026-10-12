#!/usr/bin/env bash
# v3 chain: features (with distinctive-name features) -> withheld-S1 simulation ->
# stage 1 + stage 2 trained on the simulation -> test prediction.
set -e
cd "$(dirname "$0")"
PY=${PY:-python}
./run_limited.sh v3_feat_train $PY features.py train
./run_limited.sh v3_feat_test  $PY features.py test
BER_DROP=0.18 ./run_limited.sh v3_make_drop $PY make_drop.py
BER_SIM=1 ./run_limited.sh v3_train  $PY train.py
BER_SIM=1 MEM_MAX=11G ./run_limited.sh v3_stage2 $PY stage2.py
BER_SIM=1 MEM_MAX=11G ./run_limited.sh v3_predict $PY predict.py
