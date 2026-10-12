"""Evaluate / tune decisions under the test set's higher decoy density.

The test holds ~1.7x (US) / ~1.4x (India) more "decoy twins" per record than train: records
whose best S1 has a near-identical name and street but a different house number, or that
add a decoy word. Its extra records are sister-business decoys, not the orphans of
withheld S1 entities. To score decisions under test-like conditions we take the labelled
out-of-fold predictions, drop the simulation's orphans (records whose S1 was withheld),
and duplicate unmatched decoy-like records until their density matches the test
(duplicates of a record get identical predictions, so they behave like extra decoys).

Usage: BER_SIM=1 python decoy_shift.py [oof parquet]
"""
import glob
import sys

import numpy as np
import polars as pl

from config import TRAIN_FEAT, work_path
from evaluate import f05_macro, load_truth
from train import best_per_query, cap_links

QOFF = 8_000_000
DUP_OFF = 20_000_000


def decoy_like(df):
    return (((pl.col("c_tset") >= 90) & (pl.col("a_tset") >= 80) & (pl.col("hn_eq") == 0)) | (pl.col("dq_max") > 5))


def test_like_population(oof_file, w_by_country, seed=0):
    """Best-candidate predictions + truth with orphans removed and decoys duplicated."""
    drop = pl.read_parquet(work_path("train_drop_ids.parquet"))["s1"].cast(pl.UInt32)
    truth = load_truth().with_columns(pl.col("qid").cast(pl.UInt32), pl.col("s1_true").cast(pl.UInt32))
    orphans = truth.filter(pl.col("s1_true").is_in(drop.implode())).select("qid")
    truth = truth.join(orphans, on="qid", how="anti")
    best = best_per_query(pl.read_parquet(work_path(oof_file), columns=["qid", "s1", "p"]))
    best = best.join(orphans, on="qid", how="anti")
    cols = ["qid", "s1", "c_tset", "a_tset", "hn_eq", "dq_max"]
    f = pl.concat([pl.read_parquet(p, columns=cols).join(best.select("qid", "s1"), on=["qid", "s1"], how="semi")
                   for p in sorted(glob.glob(work_path(f"{TRAIN_FEAT}/part_*.parquet")))])
    q = pl.concat([pl.read_parquet(work_path(f"train_s{k}.parquet"), columns=["idx", "country"])
                     .select((pl.col("idx") + off).cast(pl.UInt32).alias("qid"), "country") for k, off in ((2, 0), (3, QOFF))])
    best = (best.join(f, on=["qid", "s1"], how="left").join(q, on="qid")
                .with_columns(decoy_like(None).fill_null(False).alias("decoy_like"),
                              pl.col("qid").is_in(truth["qid"].implode()).alias("matched")))
    rng = np.random.default_rng(seed)
    extra = []
    for c, w in w_by_country.items():
        cand = best.filter((pl.col("country") == c) & pl.col("decoy_like") & ~pl.col("matched"))
        reps = int(np.floor(w - 1)); frac = (w - 1) - reps
        for r in range(reps + (1 if frac > 0 else 0)):
            take = cand if r < reps else cand.filter(pl.Series(rng.random(cand.height) < frac))
            extra.append(take.with_columns((pl.col("qid") + DUP_OFF * (r + 1)).cast(pl.UInt32).alias("qid")))
    pop = pl.concat([best] + extra) if extra else best
    s1_ids = pl.read_parquet(work_path("train_s1.parquet"), columns=["idx"])["idx"].cast(pl.UInt32)
    s1_ids = s1_ids.filter(~s1_ids.is_in(drop.implode()))
    return pop, truth, s1_ids


def score(pop, truth, ids, thr, thr_decoy=None):
    t = pl.when(pl.col("decoy_like")).then(thr_decoy if thr_decoy is not None else thr).otherwise(thr)
    links = cap_links(pop.filter(pl.col("p") > t).select("qid", "s1", "p"))
    return f05_macro(links.select("qid", "s1"), truth, ids)[0]


if __name__ == "__main__":
    oof = sys.argv[1] if len(sys.argv) > 1 else "train_oof2_sim.parquet"
    for label, w in (("train-like (no extra decoys)", {"US": 1.0, "India": 1.0}),
                     ("test-like decoys (US x1.7, India x1.4)", {"US": 1.7, "India": 1.4})):
        pop, truth, ids = test_like_population(oof, w)
        print(f"\n== {label}: {pop.height:,} records")
        for thr in (0.66, 0.75, 0.85, 0.92):
            print(f"  global thr {thr:.2f}: {score(pop, truth, ids, thr):.5f}", flush=True)
        for td in (0.85, 0.92, 0.96, 0.98):
            print(f"  thr 0.66, decoy-like thr {td:.2f}: {score(pop, truth, ids, 0.66, td):.5f}", flush=True)
