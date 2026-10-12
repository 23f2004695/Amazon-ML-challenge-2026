"""Build a stage-2 training set with synthetic multi-record decoy businesses (work/aug/).

The US test holds 3-5x more near-twin decoys per S1 than train, often with 2+ records per
decoy business (which the stage-2 family features read as a true entity's consistently
drifted address). We mimic it on the labelled simulation: a share of US unmatched records
whose best S1 is a nearby-house-number near-twin get a duplicate record (an S2 original gets
an S3-coded copy -> cross-source family; an S3 original gets an S3 copy -> within-S3 family).
Duplicates carry the original's text, stage-1 features and out-of-fold stage-1 score.
Everything is written to work/aug so that `BER_WORK=work/aug` runs stage 2 on it.

Usage: BER_SIM=1 python make_decoy_aug.py [share=0.28]
"""
import glob
import os
import sys

import numpy as np
import polars as pl

from config import TRAIN_FEAT, WORK, work_path
from evaluate import load_truth

QOFF = 8_000_000
AUG = os.path.join(WORK, "aug")
S2_DUP, S3_DUP = 20_000_000, 30_000_000   # dup qid = offset + original idx (both >= QOFF -> coded S3)


def main(share):
    os.makedirs(os.path.join(AUG, TRAIN_FEAT), exist_ok=True)
    truth = load_truth().with_columns(pl.col("qid").cast(pl.UInt32))
    oof = pl.read_parquet(work_path("train_oof_sim.parquet"))
    best = oof.group_by("qid").agg(pl.col("s1").sort_by("p").last(), pl.col("p").max())
    cols = ["qid", "s1", "c_tset", "a_tset", "hn_eq", "hn_logdiff"]
    f = pl.concat([pl.read_parquet(p, columns=cols).join(best.select("qid", "s1"), on=["qid", "s1"], how="semi")
                   for p in sorted(glob.glob(work_path(f"{TRAIN_FEAT}/part_*.parquet")))])
    q = pl.concat([pl.read_parquet(work_path(f"train_s{k}.parquet"), columns=["idx", "country"])
                     .select((pl.col("idx") + off).cast(pl.UInt32).alias("qid"), "country") for k, off in ((2, 0), (3, QOFF))])
    cand = (best.join(f, on=["qid", "s1"]).join(q, on="qid").join(truth.select("qid"), on="qid", how="anti")
                .filter((pl.col("country") == "US") & (pl.col("c_tset") >= 90) & (pl.col("a_tset") >= 80)
                        & (pl.col("hn_eq") == 0) & (pl.col("hn_logdiff") <= 3.43)))
    rng = np.random.default_rng(7)
    pick = cand.filter(pl.Series(rng.random(cand.height) < share)).select("qid")
    dmap = pick.with_columns(pl.when(pl.col("qid") < QOFF).then(pl.col("qid") + S2_DUP)
                               .otherwise(pl.col("qid") - QOFF + S3_DUP).cast(pl.UInt32).alias("dup"))
    print(f"US near-twin nearby decoy singles: {cand.height:,}; duplicated: {dmap.height:,}")
    # text: dup rows appended to the S3 table (qid = idx + QOFF)
    s2 = pl.read_parquet(work_path("train_s2.parquet")); s3 = pl.read_parquet(work_path("train_s3.parquet"))
    src = pl.concat([s2.with_columns(pl.col("idx").cast(pl.UInt32).alias("qid")),
                     s3.with_columns((pl.col("idx").cast(pl.UInt32) + QOFF).alias("qid"))]).join(dmap, on="qid")
    dup_rows = src.with_columns((pl.col("dup") - QOFF).cast(pl.UInt32).alias("idx")).drop("qid", "dup").select(s3.columns)
    pl.concat([s3.with_columns(pl.col("idx").cast(pl.UInt32)), dup_rows]).write_parquet(os.path.join(AUG, "train_s3.parquet"))
    s2.write_parquet(os.path.join(AUG, "train_s2.parquet"))
    for n in ("train_s1.parquet", "train_drop_ids.parquet", "decoy_dict.parquet"):
        pl.read_parquet(work_path(n)).write_parquet(os.path.join(AUG, n))
    # stage-1 OOF scores and stage-1 feature rows for the duplicates
    d_oof = oof.join(dmap, on="qid").with_columns(pl.col("dup").alias("qid")).drop("dup").with_columns(pl.col("label") * 0)
    pl.concat([oof, d_oof.select(oof.columns)]).write_parquet(os.path.join(AUG, "train_oof_sim.parquet"))
    for p in sorted(glob.glob(work_path(f"{TRAIN_FEAT}/part_*.parquet"))):
        dst = os.path.join(AUG, TRAIN_FEAT, os.path.basename(p))
        if not os.path.exists(dst):
            os.symlink(p, dst)
    feats = pl.concat([pl.read_parquet(p).join(dmap, on="qid") for p in sorted(glob.glob(work_path(f"{TRAIN_FEAT}/part_*.parquet")))])
    feats = feats.with_columns(pl.col("dup").alias("qid")).drop("dup").with_columns(pl.col("label") * 0)
    feats.write_parquet(os.path.join(AUG, TRAIN_FEAT, "part_9999.parquet"))
    dmap.write_parquet(os.path.join(AUG, "dup_map.parquet"))
    print(f"wrote {AUG}: {feats.height:,} duplicate stage-1 rows")


if __name__ == "__main__":
    main(float(sys.argv[1]) if len(sys.argv) > 1 else 0.28)
