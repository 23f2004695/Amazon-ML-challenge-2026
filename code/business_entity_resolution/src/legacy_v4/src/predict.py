"""Stage 5: score test pairs, assign matches, write the two submission files.

output/matching_results.tsv  - final matches per Source-1 entity
output/candidate_pairs.tsv   - exactly the pairs the stage-1 model scored (superset of matches)

Stage 1 scores every candidate pair; stage 2 re-scores pairs with p1 >= P_FLOOR using
group context; each S2/S3 record is linked to its best S1 if p2 exceeds the tuned threshold.

Usage: python predict.py
"""
import glob
import json
import multiprocessing as mp
import os

import lightgbm as lgb
import numpy as np
import polars as pl

from config import N_JOBS, OUTPUT, TAG, work_path
from stage2 import attach_stage1_feats, group_feats
from train import assign, cap_links

QOFF = 8_000_000


def id_maps():
    s1 = pl.read_parquet(work_path("test_s1.parquet"), columns=["idx", "entity_id"]).rename({"idx": "s1"})
    qs = []
    for k, off in ((2, 0), (3, QOFF)):
        d = pl.read_parquet(work_path(f"test_s{k}.parquet"), columns=["idx", "entity_id"])
        qs.append(d.select((pl.col("idx") + off).alias("qid"), pl.col("entity_id").alias("qe")))
    return s1, pl.concat(qs)


def write_lists(pairs, s1, col, path):
    """pairs: DataFrame(s1, qid). Writes one row per S1 entity (empty list allowed)."""
    grouped = (pairs.join(QMAP, on="qid").group_by("s1")
                    .agg(pl.col("qe").unique().sort().str.join(",").alias(col)))
    out = (s1.join(grouped, on="s1", how="left").with_columns(pl.col(col).fill_null(""))
             .select(pl.col("entity_id").alias("source1_entity_id"), col))
    out.write_csv(path, separator="\t", quote_style="never")
    print(f"wrote {path}: {out.height:,} rows, {(out[col] != '').sum():,} non-empty")


def stage1():
    """Score every candidate pair with the averaged stage-1 fold models."""
    meta = json.load(open(work_path(f"model_meta{TAG}.json")))
    models = [lgb.Booster(model_file=work_path(f"lgb_fold{f}{TAG}.txt")) for f in (0, 1)]
    preds = []
    for p in sorted(glob.glob(work_path("test_feat/part_*.parquet"))):
        d = pl.read_parquet(p)
        X = d.select(meta["feats"]).to_numpy().astype(np.float32)
        pr = np.mean([m.predict(X) for m in models], axis=0)
        preds.append(d.select("qid", "s1").with_columns(pl.Series("p1", pr.astype(np.float32))))
    pl.concat(preds).write_parquet(work_path(f"test_pred1{TAG}.parquet"))


def stage2():
    """Group-aware re-scoring of plausible pairs (p1 >= P_FLOOR)."""
    meta2 = json.load(open(work_path(f"model2_meta{TAG}.json")))
    pool = mp.get_context("spawn").Pool(min(N_JOBS, 10))
    P = group_feats(pl.read_parquet(work_path(f"test_pred1{TAG}.parquet")), "test", pool)
    pool.close()
    P = attach_stage1_feats(P, "test")
    X = P.select(meta2["feats"]).to_numpy().astype(np.float32)
    keys = P.select("qid", "s1")
    del P
    models2 = [lgb.Booster(model_file=work_path(f"lgb2_fold{f}{TAG}.txt")) for f in (0, 1)]
    p2 = np.mean([m.predict(X) for m in models2], axis=0)
    keys.with_columns(pl.Series("p", p2.astype(np.float32))).write_parquet(work_path(f"test_pred2{TAG}.parquet"))


def main():
    global QMAP
    if not os.path.exists(work_path(f"test_pred1{TAG}.parquet")):
        stage1()
    print("stage 1 done", flush=True)
    if not os.path.exists(work_path(f"test_pred2{TAG}.parquet")):
        stage2()
    print("stage 2 done", flush=True)
    thr = json.load(open(work_path(f"model2_meta{TAG}.json")))["threshold"]
    s1, QMAP = id_maps()
    matches = cap_links(assign(pl.read_parquet(work_path(f"test_pred2{TAG}.parquet")), thr))
    print(f"threshold {thr}: {matches.height:,} linked S2/S3 records")
    write_lists(matches.select("s1", "qid"), s1, "matched_entity_ids", os.path.join(OUTPUT, "matching_results.tsv"))
    del matches
    cand = pl.read_parquet(work_path(f"test_pred1{TAG}.parquet"), columns=["s1", "qid"])
    write_lists(cand, s1, "candidate_entity_ids", os.path.join(OUTPUT, "candidate_pairs.tsv"))


QMAP = None

if __name__ == "__main__":
    main()
