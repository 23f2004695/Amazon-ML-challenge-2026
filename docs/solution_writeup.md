# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** DuoML
**Team Members:** Prabhjot Kaur Batth (Team Leader), Jashan Tiwari
**Submission Date:** September 2026

---

## 1. Executive Summary

We treat the task as *assignment*, not pairwise clustering. The ground truth shows that every Source-2/3 record belongs to **at most one** Source-1 entity, so each S2/S3 record ("query") is linked to its single best S1 candidate, or to none. The pipeline has three parts:
1. An IDF-weighted inverted-index blocker that reaches 97.5% pair recall at 10 candidates per query.
2. A pairwise LightGBM matcher.
3. A second, group-aware LightGBM stage that uses how confidently linked sibling records of the same S1 agree with the query.

The final system (v9) adds:
- a stage-2 model retrained with synthetic multi-record decoy businesses (v8);
- a third stage that re-scores the uncertain pairs with group context rebuilt from stage-2 probabilities and with evidence the earlier stages do not see (v9).

It scores **0.9864** out-of-fold on a *withheld-S1 simulation* that mirrors the test set's higher density of unmatched records. On the same simulation, the first model scored 0.9696.

Leaderboard history: v2 0.961, v4 0.975, v5 0.977, v6 0.978, v8 0.979.

Two further findings drove the final model: test has ~23% more S2/S3 records per S1 than train, and French names often embed the city name.

---

## 2. Methodology

### 2.1 Problem Analysis

- **Scale:** train has 2.2M S1, 5.0M S2 and 5.3M S3 records; test has 1.73M, 4.89M and 5.08M.
- **Structure:** 7.64M S2/S3 records are matched, and every one of them to exactly one S1 (no ID appears twice in the ground truth). 27% of S2/S3 records match nothing. These unmatched records are the main false-merge risk: many are exact-name clones at other addresses (e.g. "Cobalt Inc" in several cities). 5.6% of S1 entities are singletons, and a matched S1 has 3.5 matches on average.
- **Name noise:** typos, duplicated or shuffled words, legal-suffix changes, junk prefixes (`***`, `--`, `<<`, `M/s`), appended phone numbers and `(ID: …)` tags, website-style names (`empirecrystalrepublic.com`), and whole names written in 9 Indic scripts (Devanagari, Telugu, Kannada, Tamil, Gujarati, Bengali, Malayalam, Odia, Gurmukhi). About 18% of Indian S2/S3 names are in Indic script. Some matches carry a completely unrelated name and are linked only through the address.
- **Address noise:** S2 is upper-case and abbreviated; S3 spells out state names and reorders components. House numbers drift (325 → 32, 2817 → 2817C), components go missing (3% of S2 addresses are empty), state names appear in native script (16 distinct strings, all Indian states), and there are "N/A" fillers and landmark phrases.
- **Unseen country:** France is 15% of the test S1 records and never appears in train. Every feature is therefore country-agnostic, and `country` is used only to keep blocking keys within the same label.

### 2.2 Solution Strategy

**Approach type:** Blocking + two-stage gradient-boosted classifier + one-to-one assignment
**Core innovation:** Group-aware second-stage re-scoring. Once confident links exist, a doubtful query is judged partly by whether it shares the exact address variant or name of records already linked to the same S1. Records of one entity inside a source often share the same address string even when the name was replaced.

Pipeline: `prepare` (normalize) → `blocking` → `features` → `train` (stage 1) → `stage2` → `predict` (assignment).

---

### 2.3 Train/test shift and how we handled it

Our first leaderboard score (0.961) was well below the train OOF score (0.9727). Two causes were identified from the test inputs alone:

1. **More unmatched records.** Test has 5.5–5.8 S2/S3 records per S1 in every country, versus 4.68 in train. This matches train with ~18% of S1 entities withheld: their records become "orphans" that tempt the matcher toward sister businesses at the same address or same-name clones. In test, S1 entities that received more than 5 S2 links (impossible in the train truth) were 6–15× more frequent than in train OOF.
   - `make_drop.py` reproduces this on train: it removes a deterministic 18% of S1, recomputes each query's rank/gap features, and relabels the orphans as unmatched.
   - Scoring the v2 models on this simulation dropped their F0.5 from 0.9727 to 0.9696. Final models are **trained and thresholded on the simulation**.
2. **City names inside French names.** 16.5% of French S1 records share an exact address with another S1 (5–6% in the US/India), and names such as "Lille Club" / "Lille Comite" / "Dunkerque Centre" share the city token. We added *distinctive-name* features, computed after removing tokens that occur in either address, plus counts of name tokens without a fuzzy counterpart on the other side ("ZF Trading **Holdings**"). The same pattern exists in Indian train names ("Hyderabad Farms"), so the model learns it from labelled data, without a country-specific rule.
3. **Decoy signatures (largest single gain, +0.010).** Train contains decoy records: sister businesses of an S1 entity on the same street, with an added name word and/or a nearby house number. Two feature groups capture this:
   - **House-number relation.** Among pairs with similar names and addresses, 95% of equal-number pairs are true matches. Numbers that differ by ≤30 are true in only 6–14% of pairs (decoys). One-digit typos with a large numeric jump (325 → 425) are true 78% of the time, and truncations (325 → 32) 61% (noise). Features: log numeric difference, relative difference, digit edit distance, truncation flag, and the closest number pair.
   - **Decoy-word log-odds.** Some added words never occur in true matches but appear thousands of times in decoys: industries, public, holdings, group, ventures, exports, overseas… Others are harmless noise: services, llc, center, inc, the… `decoy_dict.py` learns log P(word added | non-match) − log P(word added | match) for query-side and S1-side added words. It is built **out of fold**: rows of query-fold r use the dictionary from fold 1 − r, and test uses all of train. Features: max, sum and count of the log-odds.
4. **France without labels (v5–v6).**
   - **Dotted legal forms.** "S.A.S.U.", "S.A.R.L." and "E.U.R.L." were split into letters, which the decoy vocabulary then scored as decoy words (~7k lost French links). They are now kept as one token.
   - **Unseen-country decoy vocabulary.** Vocabulary learned on US/India does not transfer to French. We estimate French decoy words label-free:
     - On train, a word that appears on one side of a top-1 pair is a decoy marker almost exactly when such pairs have a *different* house number. The rank correlation between a word's mismatch rate and its learned log-odds is 0.76.
     - The mismatch rate is a mixture of the true-pair and false-pair rates (0.15 / 0.85, learned on train). This gives the share of false pairs per word (validated on train words: corr 0.92) and, through a fitted map, a log-odds.
     - The resulting French vocabulary: groupe, participations, holding, développement, international → strong decoy words (8.6–9.1); SARL/SAS/EURL → mild (~2.3); et, fils, associés, services → noise.
   - **Dictionary dropout.** A quarter of train queries never see the dictionary, so the model also learns to judge unknown vocabulary.
   - **Diagnostic.** A label-free expected-F0.5 per country, computed from the model's calibrated probabilities, located the problem. France scored 0.967 (v4) vs 0.991 for US/India; v6 with the label-free vocabulary reaches 0.983.
5. **Input fixes (v6).**
   - OCR-style digits inside name tokens are repaired ("c0mpany" → company, "h0ldings" → holdings; 2.5% of S2/S3 names).
   - The house number is the first non-unit number: numbers after Unit/PMB/Apt/Suite/Floor/Box are unit numbers. The first number was wrong for 1.3% of US true pairs.
   - Spelled ordinals ("Thirteenth" → 13) are converted.
   - A new feature checks whether the S1 house number appears anywhere among the record's numbers. Noise prepends numbers such as "HN 791" to 17% of Indian true pairs.
6. **Learned blocking re-ranker (v5).** The top 60 S1 per record by IDF are re-ranked by a LightGBM on cheap string evidence before the top 15 are kept (recall@10 0.979 → 0.986).
7. **Sibling expansion (v5).** A record that shares an exact normalized address / name with a record confidently linked to S1 X gets X as a candidate, when that text points to ≤ 2 linked S1s.
8. **Cap:** at most 5 S2 and 6 S3 links per S1, the maxima observed in the train truth; the lowest-probability links are removed first.
9. **US decoy families (v8).** The US test has about 3.6× more multi-record decoy businesses per S1 than train: several records sharing one nearby, different house number. The stage-2 family features read them as a true entity's consistently drifted address.
   - `make_decoy_aug.py` duplicates 28% of the US unmatched near-twin records of the simulation (as a cross-source or within-S3 copy) into a separate work directory, and stage 2 is retrained there.
   - On this family-heavy population, US F0.5 rises from about 0.979 to 0.985; India is unchanged. Leaderboard: 0.978 → 0.979.
10. **Stage 3 (v9, `stage3.py`).** Pairs with 0.01 < p2 < 0.99 (1.1M train pairs) are re-scored by a third LightGBM. It uses:
    - all stage-2 features;
    - the stage-2 group features recomputed from p2, which gives cleaner sets of confidently linked siblings;
    - house-number arithmetic: absolute difference, zero padding, prefix and length relations;
    - query noise markers: `#1234` tags, `##`, OCR digits, junk prefixes, brackets, null tokens, PMB, web names, phone numbers, duplicated tokens. Unmatched decoy records carry less generator noise than true records.
    - query flags: no house number, domain-style name, d/b/a;
    - raw-name formatting agreement that normalization discards (case style, punctuation, commas, raw ratio);
    - how many of the query's candidates share the S1's normalized name.

    Trained on the decoy-family population with the stage-2 query folds (leak-free stacking), it scores 0.9852 vs 0.9844 for stage 2. A second iteration on top of stage 3 overfit (0.9840) and was rejected.
11. **Model blend for US and India (v9).** The stage-3 probability is averaged with the v5 and v4 stage-2 probabilities (0.5 / 0.25 / 0.25). These versions differ in normalization, candidates and features. The older candidate lists also contain true pairs that the v6 blocking re-ranker dropped, e.g. `#palmerdirect | #6 Morris Place` → `Palmer Direct Lightwave | 6 Morris Place`; such pairs are 90% true in the simulation. F0.5: India 0.9847 → 0.9853, US 0.9859 → 0.9862. France is not blended: v4 predates the French fixes.
12. **France: role-aware vocabulary fix (v9, `fr_role_fix.py`).** The label-free French vocabulary scored groupe / développement / france as strong decoy words. These words also serve as noise suffixes that replace a generic word ("Chasse Club SARL" → "Chasse SARL Groupe"), and in that role they keep the S1's house number (mismatch 6–17%, like noise words, vs 94–99% for decoy words).
    - For pairs in the noise role (the word is the only added token, the S1 has at most one extra token and it is not a legal form, equal house numbers, top candidate), the probability of the model without the unseen-country vocabulary is kept. This restores 18.9k French links.
    - The US dual-role word "partners" has the same signature in train: 99.9% true in the swap role, 88.6% in the added role.
13. **Decision (v9).**
    - Threshold 0.70 for US/India and 0.66 for France (v8).
    - An S1 left with no link receives its best still-unlinked record if p > 0.55 (+0.0002; 5.9% of S1 end with no link vs 5.6% true singletons).
    - For US pairs whose house numbers differ by 1–30, the stage-3 / blended probability may lower but never raise the v8 stage-2 probability (pairs v8 never scored are capped at 0).
14. **Residual stacker (v9d, `resid_stack.py`).** A small LightGBM re-scores the uncertain US/India pairs (0.02 < p < 0.98). Its inputs are the member probabilities and the decision context they imply:
    - the query's number of candidates and of strong candidates, whether the pair is the query's top, and the other option's p;
    - the records the S1 already holds, per source, and their summed p;
    - competing strong records for the S1;
    - source, empty address, country, and exact name equality.

    On the simulation it scores 0.9858 → 0.9863. On test, only its removals and its India additions are used, because US additions are exposed to the extra US decoy families (hardened version: 0.9862).
15. **France chance house-number collisions (v9d, `fr_collision_fix.py`).** 86% of French house numbers have 1–2 digits and French names repeat heavily. A record and a same-name S1 in another street of the same city therefore often share the house number by chance, and the US-trained model links them.
    - Among same-name, equal-number top pairs, the share with different streets rises from 0.06% for unique names to 7% for names shared by 51+ S1. It is flat at 0.3% in the US and 3.3% in India, and 4–10× lower for 3+ digit numbers. This is the signature of chance collisions, not of street noise.
    - Links with the same core name, equal number, different street and a name shared by ≥ 11 S1 are removed: 1,351 pairs, estimated precision 0.01–0.06, +0.0002 LB.
16. **Numbered streets (v10, `BER_ORD_STREET=1`).** The normalizer turned street ordinals into numbers.
    - "309 18th Street" and "309 5th Street" both became "st" + {309, 18/5}, so thousands of true pairs on numbered streets were never retrieved by blocking (e.g. 'Dental-Center LLC | 309 18TH SAINT' vs 'Dental Center LLC | 309 18th Street'), and "87TH STREET" was read as house number 87.
    - An ordinal followed by a street word now becomes an address token: 8% of US and 3–5% of Indian addresses.
    - The whole chain was rerun: blocking recall@10 0.9861 → 0.9874. Simulation scores rose at every stage: stage 2 +0.0003, stage 3 +0.0002, final US/India decision 0.9862 → 0.9864.
    - The final file takes US/India from v10 and France from v9d, since France has no numbered streets.
    - New US near-twin links that v9d did not have are dropped (1,255), because of the US test's extra decoy families. On test it would otherwise add 4× more such links per S1 than on the simulation, in the bucket where the US test holds extra decoys. Cost on the simulation: −0.0001.

## 3. Candidate Generation (Blocking)

Every record emits hashed keys, prefixed with its country label so records only meet records from the same label:

| key | example | purpose |
|---|---|---|
| core name token | `n|endocrinology` | rare name words |
| concatenated core name | `n|empirecrystalrepublic` | website-style names |
| unordered pair of core name tokens | `b|cardiology_health` | word re-ordering; common words become specific in pairs |
| house number × address token | `h|2817|16` | street address |
| adjacent address-token pair | `s|buck_ridge` | street names, robust to house-number drift |
| name token × address token | `x|golden|ghaziabad` | generic names inside a locality |
| address token | `a|bibavewadi` | rare localities |

- A key's weight is `log(N_country / df)`, where df is its document frequency in Source 1. Keys with df > 400 are dropped (> 100 for bare address tokens). Each query scores S1 records by the sum of weights of shared keys, split per key type. The top 15 are kept, and the **top 10 per query are fed to the model**. These are exactly the pairs listed in `candidate_pairs.tsv`.
- The join is done in chunks of 40–100k queries (Polars), so peak memory stays around 6 GB.
- **Recall (train, fraction of true pairs in the candidate list):** @1 94.7%, @3 96.5%, @10 **97.5%**, @15 97.7%. Adding the street-pair and name×address keys raised recall@10 from 93.3% to 97.5%.
- **Pairs scored by the model:** about 100M on train and on test (≤10 per S2/S3 record), against 1.7M × 10M = 1.7·10¹³ possible pairs.

---

## 4. Matching Model

### Normalization
- **Indic script:** a token dictionary is learned from the training matches by aligning Indic-script names with their S1 Latin names position by position and taking a majority vote (1,347 tokens, covering 96% of Indic token occurrences in test). Unknown tokens fall back to unidecode with repeated letters squashed.
- **Names:** accents folded; `&`/`+` → "and"; phone numbers, `(ID: …)` tags and URLs stripped (the domain body is kept); legal forms removed from the *core* name (US/India: LLC, Inc, Corp, Pvt, Ltd, LLP…; France: SARL, SAS, SASU, EURL, SA, SNC, EI, SCI, Cie…); repeated words de-duplicated.
- **Addresses:** street-type abbreviations canonicalized (English and French: rue, avenue, boulevard, impasse, allée, route, chemin…); ordinals reduced to digits ("41th" → 41, "Eighth" → 8); leading zeros stripped; digit/letter runs split ("2817C" → 2817 c); state components mapped to canonical codes (US and India, including native-script state names); "N/A", "Unit", "PMB", "No." and similar noise dropped.

### Stage-1 features (70)
- **Name:** rapidfuzz ratio, token-sort, token-set and partial ratio on core and full names; Jaro-Winkler on concatenated core names; token Jaccard; first-token equality; substring containment; token counts; website-domain vs concatenated-name similarity; legal-form agreement.
- **Address:** token-set, ratio and partial on normalized addresses; token Jaccard; first house number equal / contained; number-set Jaccard; state agreement; empty-address flags.
- **Blocking context:** total and per-key-type IDF scores, number of shared keys, rank within the query, gap to the query's best and second-best score, number of candidates, the S1's number of competing queries and its best score from any query.
- **Genericness:** how many S1 records share the exact core name (for both the query and the S1).
- **Distinctive name (v3):** ratio, token-set ratio and Jaccard of the core-name tokens that occur in neither address, their counts, and the number of name tokens on each side without a fuzzy (≥80) counterpart on the other.
- **Decoy signatures (v4–v6):** house-number log/relative difference, digit edit distance, truncation flag, closest-number difference, S1 house number contained in the record's numbers; decoy-word log-odds (max/sum/count, query side and S1 side); initials match ("BE" vs "Bordeaux Ecole").
- **Blocking context (v5):** re-ranker score and gaps, IDF rank, number of hits.
- **Other:** record source (S2/S3) and an Indic-script flag. No country feature is used.

### Stage-2 features (+23)
Stage 2 re-scores the pairs with stage-1 probability p1 ≥ 0.01 (9.3M train pairs, containing 99.85% of the positives that are in the candidate lists):
- **Query competition:** p1, the margin to the query's best other candidate, and the number of strong alternatives.
- **S1 competition:** the sum and maximum of p1 over other queries claiming the same S1, and how many of them are strong links.
- **Sibling agreement:** how many records confidently linked (p1 > 0.5) to the same S1 share the query's exact raw address, normalized address, core name or full name, plus the best fuzzy (token-set) address similarity, name similarity, and min(address, name) similarity to those records.
- **Family support (v5):** how many other candidate records of the S1 share the query's added word / its differing house number (noise is per record, a decoy business carries its word and number on all of its records).
- **Sibling expansion / room (v5–v6):** expansion flags and ambiguity count, links already held by the S1 from the query's source vs the 5-S2 / 6-S3 maximum.

**Model type:** LightGBM binary classifiers.
- Stage 1: 600 trees, 255 leaves, fitted on a 12% query sample.
- Stage 2: 500 trees, 127 leaves.

**Validation:** 2-fold cross-validation **grouped by query** (fold = hash(query id) mod 2). All pairs of a query share its label, so this is where leakage would come from. Stage-2 inputs are out-of-fold stage-1 predictions, so stacking is leak-free. At test time the two fold models are averaged.

**Decision rule:** each S2/S3 record keeps only its highest-scoring S1 and is linked when p2 > threshold. The threshold is chosen by grid search on the **exact macro F0.5** (one score per S1, singletons included) over all out-of-fold train predictions: **0.66**.

---

## 5. Results & Error Analysis

| system | OOF macro F0.5 (train, 2.2M S1) |
|---|---|
| Stage 1, best threshold (0.70) | 0.9657 |
| Stage 1 + stage 2, exact sibling features (0.66) | 0.9714 |
| Stage 1 + stage 2, exact + fuzzy sibling features (0.66) — v2 | 0.9727 |

On the withheld-S1 simulation (test-like; scored on the remaining 82% of S1):

| system | simulated OOF F0.5 |
|---|---|
| v2 models (trained on plain train) | 0.9696 |
| v3: + distinctive-name / extra-token features, trained on the simulation (thr 0.66) | 0.9715 |
| v4: + house-number relation and decoy-word features (thr 0.70) | 0.9814 |
| v5: + re-ranked blocking, sibling expansion, family support, acronym fix, dictionary dropout | 0.9829 |
| v6: + OCR repair, unit-aware house numbers, reverse number check, per-source room, 20% stage-1 sample (thr 0.66) | 0.9845 |
| v8: stage 2 retrained with synthetic US decoy families | 0.9844 (US on the family-heavy population: 0.979 → 0.985) |
| v9: + stage 3 (thr 0.70) | 0.9852 |
| v9: + rescue of link-less S1 | 0.9855 |
| v9b: + India blend with v5/v4, US near-twins not raised | 0.9856 |
| v9c: US and India blended with v5/v4, US near-twins not raised | 0.9858 (US 0.9862, India 0.9853) |
| v9d: + residual stacker (removals, India additions); France collision fix (label-free) | 0.9862 |
| **final (v10g): numbered-street normalization, full rerun; France from v9d; US near-twin guard** | **0.9864** |

Leaderboard: v2 0.961, v4 0.975, v5 0.977, v6 0.978, v8 0.979.

**Where the remaining loss is (v6, simulation).** Removing every false link gives +0.0036. Linking every true pair in the stage-2 set gives +0.0064. Blocking misses cost about 0.0053. Six diagnostic analyses, each checked by an independent verifier, found that most of this is irreducible from the records themselves:
- 85% of missed links for records without an address are ties between same-name S1 entities in different cities. 50% of S1 share a core name, and nothing in the record picks the right one.
- In the high-similarity cells, generated unmatched near-duplicates (same name, nearby house number, the usual noise) outnumber true pairs 2:1 with identical-looking edits.
- Most blocking misses are the same name ties.

Per-segment thresholds, prior-shift corrections, a "hungry S1" rule and twin tie-breaking were all tested and rejected.

Upper bounds on the simulation for v4 (oracle analysis):

| scenario | F0.5 |
|---|---|
| no false links | 0.9845 |
| every true pair in the stage-2 set found | 0.9878 |
| perfect model on all candidates | 0.9915 |

The ceiling comes from blocking recall (97.5%) and inherently ambiguous records: records with no address sharing a generic name with several S1 entities, and names replaced by a trade name.

The breakdown below is for the stage-2 model with exact sibling features (OOF 0.9714):
- **By country:** US 0.9738, India 0.9678.
- **By true match count:** singletons 0.973; S1 with 1 true match 0.892; 3 matches 0.975; 6+ matches 0.983.
- **Query-level outcomes (train OOF):** 7.20M correct links, 53k unmatched records linked by mistake, 7.6k linked to the wrong S1, 227k true links scored below the threshold, 200k true links not in the candidate list or with p1 < 0.01. Pair precision is about 99.2% and pair recall about 94.3%.
- **Distractor check:** train S2/S3 names containing Indic words absent from the learned dictionary have a 0% true match rate (synthetic distractors with an inserted business-type word). The model links 0.03% of the equivalent test records, i.e. it has learned to reject them.
- **Common false positives:** generic names ("Oncology Associates LLC", "Golden Foods Private Limited") at a nearby or partially matching address, and unmatched clones sharing a street with the true entity.
- **Common false negatives:** records whose name was replaced by an unrelated word ("Lumlyra", "Zephevo") while the address is also degraded (house number changed, street missing); Indic-script names using words never seen in train (e.g. "स्टोर्स" = stores), handled only by approximate transliteration; addresses reduced to city + state.

---

## 6. Conclusion

The biggest gains came from the structure of the data rather than from model complexity:
1. Treating matching as one-to-one assignment.
2. Blocking keys designed from observed noise (street pairs and name×locality combinations added 4 points of blocking recall).
3. A group-aware second stage that lets confidently linked siblings vouch for doubtful records.

All components are CPU-only, use only the provided data, and treat country as an open label, so France is handled without special cases.

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/` (all code in `src/`; see README.md). `src/run_all.sh` regenerates both output files from the raw data in about 7 h on 16 CPU cores, using ≤16 GB RAM per step:

| file | role |
|---|---|
| `src/normalize.py` | name / address normalization |
| `src/build_dict.py` | learns the Indic → Latin token dictionary from train matches |
| `src/prepare.py` | normalizes all sources → parquet |
| `src/blocking.py` | inverted-index candidate generation |
| `src/features.py` | stage-1 pairwise features |
| `src/train.py` | stage-1 2-fold LightGBM + OOF + threshold |
| `src/stage2.py` | group-aware stage-2 model + OOF + threshold |
| `src/predict.py` | test inference (stage 1 + stage 2) |
| `src/make_drop.py`, `src/make_decoy_aug.py` | withheld-S1 simulation; synthetic US decoy families (v8) |
| `src/stage3.py` | stage-3 re-scoring of the uncertain region (v9) |
| `src/fr_role_fix.py` | role-aware unseen-country vocabulary fix (v9) |
| `src/fr_collision_fix.py` | France chance house-number collision fix (v9d) |
| `src/resid_stack.py` | residual stacker on the final US/India probabilities (v9d) |
| `src/merge_countries.py`, `src/run_v10.sh` | v10 rerun with numbered-street tokens and the final country-wise composition |
| `src/make_final.py` | final decision (country-wise stage 3 / blend, thresholds, rescue, caps) → `output/matching_results.tsv`, `output/candidate_pairs.tsv` |
| `src/run_all.sh` (= `src/run_v9.sh` + `src/run_v10.sh`) | full chain of the final submission |
| `src/legacy_v4/`, `src/legacy_members/` | code snapshot of the v4 blend member; frozen v4 / v5 stage-2 predictions used by the blend |
| `src/evaluate.py` | exact macro F0.5 |
| `src/blocking_recall.py`, `src/error_analysis.py` | diagnostics |

Models used: LightGBM (MIT license), far below the 8B-parameter limit. No external data, APIs or pretrained models. Dependencies: polars, numpy, lightgbm, rapidfuzz, unidecode (see requirements.txt; Python 3.11).

### B. Additional Results

Blocking recall@K (train): 1 → 94.7%, 2 → 95.9%, 3 → 96.5%, 5 → 97.0%, 8 → 97.35%, 10 → 97.5%, 15 → 97.7%.

Final stage-2 threshold sweep (OOF F0.5): 0.50 → 0.9712, 0.58 → 0.9723, 0.62 → 0.97254, 0.64 → 0.97262, **0.66 → 0.97267**, 0.67 → 0.97268, 0.70 → 0.97263, 0.78 → 0.9720, 0.86 → 0.9701. The optimum is flat between 0.64 and 0.70.

Test-set sanity check (no labels). Links per non-empty S1 are consistent across countries and with train (train OOF 3.29; test US 3.35, India 3.39, France 3.48). About 6% of S1 entities get no match in every country, close to the 5.6% train singleton rate. France, which is unseen in training, behaves like the other countries.
