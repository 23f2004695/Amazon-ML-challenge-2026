# Business Entity Resolution — reproducible pipeline

Blocking (IDF-weighted inverted index) → pairwise features → stage-1 LightGBM (2-fold,
grouped by query) → group-aware stage-2 LightGBM → one-S1-per-record assignment with an
F0.5-tuned threshold.

## Environment

```bash
conda create -n amlc python=3.11 -y
conda activate amlc
pip install -r requirements.txt
```

CPU only; no GPU and no external data or services are used. Peak memory per step is about
6–10 GB. Every step is resumable: finished chunks and prediction checkpoints in `work/` are reused. `run_limited.sh` runs each step in a memory-capped `systemd-run` scope
(`MEM_MAX`, default 10G), so an out-of-memory condition kills only that step. You can also
run each step directly with `python <script>.py` from `src/`.

## Layout

The code expects to sit at `student_resource/code/business_entity_resolution/`, with the data
in `student_resource/dataset/{train,test}/`. Set `BER_ROOT` to point elsewhere.
Intermediate files go to `student_resource/work/` (`BER_WORK` overrides). Final files go to
`student_resource/output/`.

| file | stage |
|---|---|
| `src/config.py` | paths, thread count |
| `src/normalize.py` | name / address normalization (transliteration, legal forms, abbreviations, states) |
| `src/build_dict.py` | learns an Indic-script → Latin token dictionary from train matches |
| `src/prepare.py` | normalizes every source file → parquet |
| `src/blocking.py` | candidate generation (top-15 S1 per S2/S3 record) |
| `src/blocking_recall.py` | diagnostic: blocking recall@K on train |
| `src/features.py` | pairwise features for the top-10 candidates of every record |
| `src/evaluate.py` | ground-truth loading and the exact macro F0.5 metric |
| `src/train.py` | stage-1 2-fold LightGBM, out-of-fold predictions, threshold search |
| `src/stage2.py` | stage-2 group-aware LightGBM on out-of-fold stage-1 scores, threshold search |
| `src/error_analysis.py` | diagnostic: OOF errors by country / group size / error type |
| `src/decoy_dict.py` | learns which added name words mark a decoy / sister business (out-of-fold) |
| `src/oracle_analysis.py` | diagnostic: upper bound of each error class |
| `src/make_drop.py` | withheld-S1 simulation: test has ~5.75 S2/S3 per S1 vs 4.68 in train, as if ~18% of S1 were withheld |
| `src/stress_test.py` | diagnostic: scores an existing model set on the simulation |
| `src/predict.py` | scores test pairs, writes `matching_results.tsv` and `candidate_pairs.tsv` |

## Run end-to-end (≈2 h on 16 cores)

```bash
cd code/business_entity_resolution/src
python build_dict.py          # Indic token dictionary (train labels only)
python prepare.py train
python prepare.py test
python blocking.py train
BER_QCHUNK=40000 python blocking.py test   # smaller query chunks: test keys fan out more
python decoy_dict.py                # decoy-word log-odds (out-of-fold on train)
python features.py train
python features.py test
BER_DROP=0.18 python make_drop.py   # withheld-S1 simulation of the test distribution
BER_SIM=1 python train.py           # stage 1 on the simulation (OOF F0.5 per threshold)
BER_SIM=1 python stage2.py          # stage 2 on the simulation, saves threshold
BER_SIM=1 python predict.py         # writes ../../../output/*.tsv
cd ../../..
python3 utils/validate_submission.py --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv --test-dir dataset/test
```

`run_v4.sh` runs the decoy dictionary → feature → simulation → stage 1 → stage 2 → predict chain in one go
(set `PY` to your python). `BER_SIM=1` makes train/stage2/predict use the simulated train
features and write `*_sim` model files.
