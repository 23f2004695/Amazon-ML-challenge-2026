# Business Entity Resolution: reproducible pipeline

The pipeline has five steps:
1. Blocking with an IDF-weighted inverted index and a learned re-ranker.
2. Pairwise features.
3. A stage-1 LightGBM, trained with 2 folds grouped by query.
4. A group-aware stage-2 LightGBM, retrained with synthetic decoy families.
5. A stage-3 re-scoring of uncertain pairs, then an F0.5-tuned decision that links each record to at most one S1.

All code is in `src/`. `src/run_all.sh` regenerates `output/matching_results.tsv` and `output/candidate_pairs.tsv` from the raw training and test data.

## Environment

```bash
conda create -n amlc python=3.11 -y
conda activate amlc
pip install -r requirements.txt
```

- Hardware: CPU only (16 cores used). No GPU.
- Data: no external data, APIs or pretrained models.
- Models: LightGBM (MIT license), far below the 8B-parameter limit.

The run scripts wrap every step in a memory-capped `systemd-run --user --scope` (`run_limited.sh`, cap set by `MEM_MAX`). Peak memory is at most 16 GB per step.
- Logs are written to `<work>/logs/<step>.log`.
- Without systemd, run the same commands directly from `src/`, as listed in the step-by-step section below.

## Data and paths

| What | Where | Override |
|---|---|---|
| This package | unzipped as `student_resource/`, next to the provided `dataset/` folder | `BER_ROOT` = the folder that contains `dataset/{train,test}/` |
| Intermediate files | `<root>/work`, plus `<root>/work2` for the final (v10) chain | `BER_WORK` (v9 chain), `BER_WORK_V10` (v10 chain) |
| Final files | `<root>/output/` | |

## Run end-to-end (about 7 h on 16 cores)

```bash
cd code/business_entity_resolution/src
PY=python ./run_all.sh          # = ./run_v9.sh, then ./run_v10.sh
cd ../../..
python3 utils/validate_submission.py --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv --test-dir dataset/test --check-ids
```

### What the scripts do

- **`run_v6.sh`: base chain.**
  1. Indic token dictionary.
  2. Normalization.
  3. Deep candidates and the blocking re-ranker.
  4. Blocking: top 60 by IDF, re-ranked, top 15 kept.
  5. Decoy-word dictionaries: out-of-fold on train, and label-free for countries absent from train.
  6. Features.
  7. Withheld-S1 simulation of the test distribution: 18% of train S1 removed.
  8. Stage 1, then stage 2.
  9. Test predictions.
- **`run_v9.sh`: the France rows of the final file.** It runs `run_v6.sh`, then:
  1. Synthetic multi-record US decoy families, and stage 2 retrained on them (`make_decoy_aug.py`).
  2. Test stage 2.
  3. Stage 3 (`stage3.py`).
  4. Residual stacker (`resid_stack.py`).
  5. The decision (`make_final.py`), which writes `output/*_v9d.tsv`.
     - France: role-aware fix of the label-free decoy vocabulary (`fr_role_fix.py`) and chance house-number collision fix (`fr_collision_fix.py`).
     - US/India: blend of stage 3 with two older stage-2 versions (v5, v4).
- **`run_v10.sh`: the US/India rows of the final file.**
  - It repeats the same chain in `work2/` with `BER_ORD_STREET=1`. Numbered streets ("18th Street", "Eighth St", "5th Cross") then become address tokens instead of house-number candidates. This raises blocking recall@10 from 0.9861 to 0.9874.
  - It ends with `merge_countries.py`, which builds the final file:
    - US and India rows come from v10;
    - France rows come from v9d;
    - new US links between records whose house numbers differ by 1–30 are dropped (the decoy-family signature);
    - the result is copied to `output/matching_results.tsv` and `output/candidate_pairs.tsv`.

### Blend members from earlier pipeline versions (`src/legacy_members/`)

The US/India blend averages stage 3 (weight 0.5) with the stage-2 probabilities of two earlier versions of this pipeline, v5 and v4 (weight 0.25 each). These versions used different normalization, candidates and features. Their test predictions and simulation out-of-fold predictions are frozen in `src/legacy_members/{v4,v5}/`:
- `test_pred2_sim*.parquet`: columns qid, s1, p;
- `train_oof2_sim.parquet`: columns qid, s1, label, p.

The run scripts copy these into the work directories.

| Member | Code | How to rebuild it |
|---|---|---|
| v4 | exact snapshot in `src/legacy_v4/`, with its own README and run scripts | from `src/legacy_v4/`, run its chain with `BER_ROOT=<root>` and a separate `BER_WORK`; the outputs are `test_pred2_sim.parquet` and `train_oof2_sim.parquet` |
| v5 | intermediate state between v4 and the current code, not archived separately | use the frozen files |

v5 differs from the current code by the v6 input fixes: OCR digit repair, unit-aware house numbers, spelled ordinals, the S1-house-number-in-record feature, the per-source room feature and the 20% stage-1 sample.

## Step by step (what `run_v9.sh` runs; v10 is the same chain with `BER_ORD_STREET=1` and `BER_WORK=<root>/work2`)

```bash
cd code/business_entity_resolution/src
python build_dict.py                                # Indic -> Latin token dictionary (train labels only)
python prepare.py train && python prepare.py test   # normalization -> parquet
BER_QCHUNK=50000 python blocking.py train sample    # deep candidates of 1/40 train queries
python train_ranker.py                              # blocking re-ranker
BER_QCHUNK=50000 python blocking.py train           # candidates: top-60 by IDF -> re-ranked -> top-15
BER_QCHUNK=40000 python blocking.py test
python decoy_dict.py && python decoy_mixture.py && python decoy_dict_unseen.py
python features.py train && python features.py test
BER_DROP=0.18 python make_drop.py                   # withheld-S1 simulation
BER_SIM=1 python train.py                           # stage 1
BER_SIM=1 python stage2.py                          # stage 2
BER_SIM=1 BER_DECISION=thr python predict.py        # test stage 1+2 (train vocabulary)
BER_SIM=1 BER_DECISION=thr BER_UNSEEN=tx python predict.py   # label-free vocabulary for unseen countries
BER_SIM=1 python make_decoy_aug.py 0.28             # synthetic US decoy families -> <work>/aug
BER_WORK=<work>/aug BER_SIM=1 python stage2.py      # stage 2 on decoy families (copy lgb2_*, model2_meta to <work>)
BER_SIM=1 BER_DECISION=thr BER_UNSEEN=tx python predict.py   # (after deleting <work>/test_pred2_sim_tx.parquet)
BER_WORK=<work>/aug BER_SIM=1 python stage3.py      # stage 3 (copy lgb3_*, model3_meta to <work>)
BER_SIM=1 BER_UNSEEN=tx python stage3.py test
BER_WORK=<work>/aug BER_SIM=1 python resid_stack.py # residual stacker (copy resid_fold* to <work>)
BER_SIM=1 python make_final.py --name v9d --members test_pred2_sim_tx.parquet \
  --stage3 test_pred3_sim_tx.parquet --blend-country India US --no-raise-near US \
  --fr-role-fix --fr-collision --thr 0.66 --thr-seen 0.70 --rescue 0.55 --resid
```

Validation: every model is trained and its threshold chosen on the withheld-S1 simulation. Query folds are fixed (hash of the record id, seed 11) and shared by all stacked stages, so out-of-fold predictions are leak-free.

| Simulation F0.5 (US + India) | Score |
|---|---|
| Stage 2 | 0.9847 |
| Stage 3 | 0.9854 |
| Final decision | 0.9864 |

## Files in `src/`

| file | role |
|---|---|
| `config.py` | paths (`BER_ROOT`, `BER_WORK`), simulation switch (`BER_SIM`), thread count |
| `normalize.py` | name / address normalization (transliteration, legal forms, abbreviations, states, OCR digit repair, house numbers, numbered streets with `BER_ORD_STREET=1`) |
| `build_dict.py` | Indic-script → Latin token dictionary learned from train matches |
| `prepare.py` | normalizes every source file → parquet |
| `blocking.py`, `train_ranker.py` | candidate generation (IDF inverted index, learned re-ranker) |
| `decoy_dict.py`, `decoy_mixture.py`, `decoy_dict_unseen.py` | decoy-word log-odds (out-of-fold on train; label-free for unseen countries via the house-number mismatch mixture) |
| `features.py` | pairwise features |
| `make_drop.py` | withheld-S1 simulation |
| `train.py` | stage-1 LightGBM, 2-fold out-of-fold predictions |
| `stage2.py` | stage-2 group-aware LightGBM (query / S1 competition, sibling agreement, decoy families, sibling expansion) |
| `predict.py` | test stage 1 + stage 2 |
| `make_decoy_aug.py` | synthetic multi-record US decoy businesses for stage-2 retraining |
| `stage3.py` | stage 3: re-scores 0.01 < p2 < 0.99 with group context from p2, house-number arithmetic, noise markers, raw-name formatting |
| `resid_stack.py` | residual stacker on the final US/India probabilities (removals; additions only for India) |
| `fr_role_fix.py` | France: role-aware fix of the label-free decoy vocabulary |
| `fr_collision_fix.py` | France: unlink chance house-number collisions |
| `make_final.py` | decision: blends, thresholds, rescue, caps (5 S2 / 6 S3 links per S1) → `output/matching_results_<name>.tsv`, `output/candidate_pairs_<name>.tsv` |
| `merge_countries.py` | final composition by S1 country (v10 for US/India, v9d for France) with the US near-twin guard |
| `evaluate.py` | ground truth and the exact macro F0.5 (singletons included) |
| `run_all.sh`, `run_v6.sh`, `run_v9.sh`, `run_v10.sh`, `run_limited.sh` | the chain described above |
| `legacy_v4/`, `legacy_members/` | code snapshot of the v4 blend member; frozen v4 / v5 member predictions |

These `src/` files are diagnostics and rejected experiments, not used by the final chain. They are kept for the methodology write-up:
- `oracle_analysis.py`, `error_analysis.py`, `fn_anatomy.py`, `blocking_recall.py`, `blocking_miss_analysis.py`, `expected_f.py`, `stress_test.py`, `aug_eval.py`
- `decision.py`, `prior_shift.py`, `bucket_shift.py`, `decoy_shift.py`, `make_variant.py`, `stage3_bag.py`
