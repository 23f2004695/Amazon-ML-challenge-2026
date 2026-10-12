# Amazon ML Challenge 2026 - Business Entity Resolution

My solution for the Amazon ML Challenge 2026, submitted under the team name **DuoML**.

The task was to link business records from three different sources to the same real-world entity. Source 1 is the reference list. Every record in Source 2 and Source 3 has to be linked to the Source 1 entity it belongs to, or left unlinked if it doesn't belong to any. The data is messy: typos, shuffled words, junk prefixes, names written in 9 Indic scripts, abbreviated or half-missing addresses. The test set also includes a country (France) that never appears in training.

**Final score:** 0.9864 macro F0.5 on my test-like validation setup
**Leaderboard progression:** 0.961 → 0.975 → 0.977 → 0.978 → 0.979

---

## The main idea

Early on I noticed that in the ground truth, every S2/S3 record belongs to **at most one** S1 entity. So instead of clustering everything, I treated it as an assignment problem: for each S2/S3 record, find its single best S1 candidate and decide whether to link it or not.

The pipeline looks like this:

```
raw TSVs
   │
   ▼
normalize  ──►  blocking (IDF inverted index + re-ranker)  ──►  ~10 candidates per record
                                                                    │
                                                                    ▼
                                         stage 1: pairwise LightGBM (70 features)
                                                                    │
                                                                    ▼
                                stage 2: group-aware LightGBM (how do the siblings agree?)
                                                                    │
                                                                    ▼
                                     stage 3: re-score the uncertain pairs only
                                                                    │
                                                                    ▼
                       decision: best S1 per record, F0.5-tuned threshold, per-S1 caps
```

Everything runs on CPU with LightGBM. I didn't use any external data, APIs or pretrained models.

## What actually moved the score

Most of the gains came from understanding the data, not from bigger models.

- **Blocking keys built from the noise I saw.** Plain name tokens weren't enough. Adding street-name pairs and name × locality keys raised recall@10 from 93.3% to 97.5%. A small learned re-ranker pushed it to about 98.7% in the final version.
- **Letting linked siblings vouch for a record.** Records of the same business inside one source often share the exact same address string, even when the name has been swapped for something unrelated. Stage 2 asks: do the records already confidently linked to this S1 look like this one?
- **Simulating the test distribution.** My first leaderboard score (0.961) was way below my OOF score (0.973). Test has ~23% more S2/S3 records per S1, which means more "orphan" records with no true match. I reproduced that on train by withholding 18% of S1 entities (`make_drop.py`), then trained and tuned everything on that simulation.
- **Decoy detection (+0.010, the biggest single jump).** The data has sister businesses on the same street with an extra word ("Holdings", "Group", "Exports") and a nearby house number. I added house-number relation features (equal vs. close vs. typo vs. truncated) and a learned dictionary of which added words point to decoys and which are just noise.
- **France without labels.** I estimated a French decoy vocabulary without any labels by using the house-number mismatch rate as a proxy. I also fixed dotted legal forms like "S.A.R.L." and handled French names that contain the city name ("Lille Club" vs "Lille Comite").
- **Numbered streets.** "309 18th Street" and "309 5th Street" were collapsing to the same thing in normalization. Fixing that and rerunning the whole chain gave the last bit of improvement.

The full write-up, with every experiment, feature list, ablation and error analysis, is in [`docs/solution_writeup.md`](docs/solution_writeup.md).

## Results

| Version | What changed | Simulated F0.5 |
|---|---|---|
| v2 | two-stage model, trained on plain train | 0.9696 |
| v3 | distinctive-name features, trained on the simulation | 0.9715 |
| v4 | house-number relation + decoy-word features | 0.9814 |
| v5 | re-ranked blocking, sibling expansion, dictionary dropout | 0.9829 |
| v6 | OCR digit repair, unit-aware house numbers | 0.9845 |
| v9 | stage 3 + blend with older versions | 0.9858 |
| v9d | residual stacker, France collision fix | 0.9862 |
| **v10 (final)** | numbered-street normalization, full rerun | **0.9864** |

## Repo structure

```
├── code/business_entity_resolution/
│   ├── README.md            # detailed reproduction guide (step-by-step commands)
│   ├── requirements.txt
│   └── src/
│       ├── normalize.py, prepare.py, build_dict.py   # cleaning + Indic transliteration
│       ├── blocking.py, train_ranker.py              # candidate generation
│       ├── features.py, decoy_dict*.py               # pairwise features
│       ├── train.py, stage2.py, stage3.py            # the three LightGBM stages
│       ├── make_final.py, merge_countries.py         # final decision + output files
│       ├── run_all.sh                                # full pipeline
│       ├── legacy_v4/, legacy_members/               # older blend members
│       └── ...                                       # diagnostics and rejected experiments
├── docs/
│   └── solution_writeup.md  # full methodology
└── output/
    └── matching_results.tsv # final submission
```

`candidate_pairs.tsv` (the second submission file) is about 1.2 GB, which is too big for GitHub, so it isn't included here. Running the pipeline regenerates it.

## Running it

You'll need the competition dataset placed in `dataset/train` and `dataset/test` at the repo root. It isn't included here.

```bash
conda create -n amlc python=3.11 -y
conda activate amlc
pip install -r code/business_entity_resolution/requirements.txt

cd code/business_entity_resolution/src
PY=python ./run_all.sh
```

The full run takes about 7 hours on 16 CPU cores and stays under 16 GB RAM per step. The scripts use `systemd-run` to cap memory. If you're not on Linux, the [code README](code/business_entity_resolution/README.md) lists every command so you can run each step yourself.

## Tech

Python 3.11 · Polars · LightGBM · RapidFuzz · Unidecode · NumPy · scikit-learn

## Author

- **Prabhjot Kaur Batth** (team lead)
