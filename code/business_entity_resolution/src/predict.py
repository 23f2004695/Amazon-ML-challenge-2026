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
from decision import optimize_links
from features import DECOY_FEATS
from stage2 import attach_stage1_feats, group_feats
from train import assign, cap_links

QOFF = 8_000_000
# Decoy vocabulary for records of countries absent from the training data (BER_UNSEEN):
#   global  - the train vocabulary (default)
#   nodict  - no vocabulary (as in dictionary-dropout training rows)
#   tx      - vocabulary estimated label-free from the test inputs (decoy_dict_unseen.py)
UNSEEN = os.environ.get("BER_UNSEEN", "nodict" if os.environ.get("BER_UNSEEN_NODICT", "0") == "1" else "global")
VAR = {"global": "", "nodict": "_nd", "tx": "_tx"}[UNSEEN]
_TX = {}


def _tx_tables():
    if not _TX:
        seen = pl.read_parquet(work_path("train_s1.parquet"), columns=["country"])["country"].unique().implode()
        _TX["q"] = pl.concat([pl.read_parquet(work_path(f"test_s{k}.parquet"), columns=["idx", "name_full", "country"])
                                .filter(~pl.col("country").is_in(seen))
                                .select((pl.col("idx") + off).cast(pl.UInt32).alias("qid"), "name_full", "country")
                              for k, off in ((2, 0), (3, QOFF))])
        _TX["s"] = (pl.read_parquet(work_path("test_s1.parquet"), columns=["idx", "name_full", "country"])
                      .filter(~pl.col("country").is_in(seen))
                      .select(pl.col("idx").cast(pl.UInt32).alias("s1"), pl.col("name_full").alias("name_full_s")))
        _TX["d"] = pl.read_parquet(work_path("decoy_dict_unseen.parquet")).select(
            "country", pl.col("token").alias("x"), "lo")
    return _TX


def _unseen_values(keys):
    """keys: DataFrame(row, qid, s1) of unseen-country pairs -> DataFrame(row, <DECOY_FEATS>)."""
    if UNSEEN == "nodict":
        return keys.select("row", *[pl.lit(0.0, pl.Float32).alias(c) for c in DECOY_FEATS])
    t = _tx_tables()
    x = (keys.join(t["q"], on="qid").join(t["s"], on="s1")
             .select("row", "country", pl.col("name_full").str.split(" ").list.unique()
                     .list.set_difference(pl.col("name_full_s").str.split(" ").list.unique()).alias("x")))
    e = (x.explode("x").drop_nulls("x").join(t["d"], on=["country", "x"])
          .group_by("row").agg(pl.col("lo").max().alias("dq_max"), pl.col("lo").sum().alias("dq_sum"),
                               pl.len().cast(pl.Float32).alias("dq_n")))
    return (keys.select("row").join(e, on="row", how="left")
                .with_columns(pl.col("dq_max", "dq_sum", "dq_n").fill_null(0.0).cast(pl.Float32),
                              *[pl.lit(0.0, pl.Float32).alias(c) for c in ("ds_max", "ds_sum", "ds_n")]))


def unseen_matrix(d, X, feats):
    """Overwrite, in the feature matrix X (rows aligned with d), the decoy-word features of
    records whose country is absent from the training data, according to BER_UNSEEN."""
    if UNSEEN == "global":
        return X
    keys = (d.select(pl.col("cseen"), pl.col("qid").cast(pl.UInt32), pl.col("s1").cast(pl.UInt32))
             .with_row_index("row").filter(pl.col("cseen") == 0).drop("cseen"))
    if keys.height == 0:
        return X
    v = _unseen_values(keys).sort("row")
    rows = v["row"].to_numpy()
    for c in DECOY_FEATS:
        if c in feats:
            X[rows, feats.index(c)] = v[c].to_numpy()
    return X


def id_maps():
    s1 = pl.read_parquet(work_path("test_s1.parquet"), columns=["idx", "entity_id"]).rename({"idx": "s1"})
    qs = []
    for k, off in ((2, 0), (3, QOFF)):
        d = pl.read_parquet(work_path(f"test_s{k}.parquet"), columns=["idx", "entity_id"])
        qs.append(d.select((pl.col("idx") + off).alias("qid"), pl.col("entity_id").alias("qe")))
    return s1, pl.concat(qs)


def write_lists(pairs, s1, col, path, n_chunks=8):
    """pairs: DataFrame(s1, qid). Writes one row per S1 entity (empty list allowed),
    processing S1 in chunks so that 100M+ candidate pairs fit in memory."""
    frames = pairs if isinstance(pairs, list) else [pairs]
    frames = [f.select(pl.col("s1").cast(pl.UInt32), pl.col("qid").cast(pl.UInt32)) for f in frames]
    s1 = s1.with_columns(pl.col("s1").cast(pl.UInt32)).with_row_index("row")
    step = s1.height // n_chunks + 1
    nonempty = 0
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(f"source1_entity_id\t{col}\n")
        for i in range(0, s1.height, step):
            part = s1.slice(i, step)
            pp = pl.concat([f.join(part.select("s1"), on="s1", how="semi") for f in frames]).unique()
            grouped = (pp.join(QMAP, on="qid").group_by("s1")
                         .agg(pl.col("qe").unique().sort().str.join(",").alias(col)))
            out = (part.join(grouped, on="s1", how="left").sort("row")
                       .with_columns(pl.col(col).fill_null(""))
                       .select(pl.col("entity_id").alias("source1_entity_id"), col))
            nonempty += (out[col] != "").sum()
            out.write_csv(fh, separator="\t", quote_style="never", include_header=False)
    print(f"wrote {path}: {s1.height:,} rows, {nonempty:,} non-empty")



def stage1():
    """Score every candidate pair with the averaged stage-1 fold models."""
    meta = json.load(open(work_path(f"model_meta{TAG}.json")))
    models = [lgb.Booster(model_file=work_path(f"lgb_fold{f}{TAG}.txt")) for f in (0, 1)]
    preds = []
    for p in sorted(glob.glob(work_path("test_feat/part_*.parquet"))):
        d = pl.read_parquet(p)
        X = unseen_matrix(d, d.select(meta["feats"]).to_numpy().astype(np.float32), meta["feats"])
        pr = np.mean([m.predict(X) for m in models], axis=0)
        preds.append(d.select("qid", "s1").with_columns(pl.Series("p1", pr.astype(np.float32))))
    pl.concat(preds).write_parquet(work_path(f"test_pred1{TAG}{VAR}.parquet"))


def stage2():
    """Group-aware re-scoring of plausible pairs (p1 >= P_FLOOR)."""
    meta2 = json.load(open(work_path(f"model2_meta{TAG}.json")))
    pool = mp.get_context("spawn").Pool(min(N_JOBS, 10))
    P = group_feats(pl.read_parquet(work_path(f"test_pred1{TAG}{VAR}.parquet")), "test", pool)
    P = attach_stage1_feats(P, "test", pool)
    pool.close()
    X = unseen_matrix(P, P.select(meta2["feats"]).to_numpy().astype(np.float32), meta2["feats"])
    keys = P.select("qid", "s1")
    del P
    models2 = [lgb.Booster(model_file=work_path(f"lgb2_fold{f}{TAG}.txt")) for f in (0, 1)]
    p2 = np.mean([m.predict(X) for m in models2], axis=0)
    keys.with_columns(pl.Series("p", p2.astype(np.float32))).write_parquet(work_path(f"test_pred2{TAG}{VAR}.parquet"))


def main():
    global QMAP
    if not os.path.exists(work_path(f"test_pred1{TAG}{VAR}.parquet")):
        stage1()
    print("stage 1 done", flush=True)
    if not os.path.exists(work_path(f"test_pred2{TAG}{VAR}.parquet")):
        stage2()
    print("stage 2 done", flush=True)
    thr = json.load(open(work_path(f"model2_meta{TAG}.json")))["threshold"]
    s1, QMAP = id_maps()
    pred2 = pl.read_parquet(work_path(f"test_pred2{TAG}{VAR}.parquet"))
    matches = cap_links(optimize_links(pred2) if os.environ.get("BER_DECISION", "expf") == "expf" else assign(pred2, thr))
    print(f"threshold {thr}: {matches.height:,} linked S2/S3 records")
    write_lists(matches.select("s1", "qid"), s1, "matched_entity_ids", os.path.join(OUTPUT, f"matching_results{VAR}.tsv"))
    del matches
    # every pair the models scored: stage-1 candidates plus stage-2 sibling-expansion pairs
    cand = [pl.read_parquet(work_path(f"test_pred1{TAG}{VAR}.parquet"), columns=["s1", "qid"]),
            pl.read_parquet(work_path(f"test_pred2{TAG}{VAR}.parquet"), columns=["s1", "qid"])]
    write_lists(cand, s1, "candidate_entity_ids", os.path.join(OUTPUT, f"candidate_pairs{VAR}.tsv"))


QMAP = None

if __name__ == "__main__":
    main()
