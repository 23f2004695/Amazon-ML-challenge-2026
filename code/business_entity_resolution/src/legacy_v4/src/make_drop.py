"""Simulate the test set's sparser Source 1 on train (validation / training aid).

Test has ~5.75 S2/S3 records per S1 versus 4.68 in train, which is what train would look
like if ~18% of its S1 entities were withheld: their S2/S3 records become unmatched
"orphans" that tempt the matcher toward similar S1 records (sister businesses at the same
address, same-name clones). This script removes a deterministic DROP share of train S1
records from the candidate pairs, recomputes the query-level context features (rank, gaps)
as blocking would see them, and writes work/train_drop_feat/ plus the dropped ids.

Usage: BER_DROP=0.18 python make_drop.py
"""
import glob
import os

import polars as pl

from config import work_path
from features import query_context

DROP = float(os.environ.get("BER_DROP", "0.18"))


def dropped_ids():
    s1 = pl.read_parquet(work_path("train_s1.parquet"), columns=["idx"])
    return s1.filter((pl.col("idx").hash(seed=5) % 1000) < int(DROP * 1000))["idx"].cast(pl.UInt32)


def main():
    drop = dropped_ids()
    pl.DataFrame({"s1": drop}).write_parquet(work_path("train_drop_ids.parquet"))
    outdir = work_path("train_drop_feat")
    os.makedirs(outdir, exist_ok=True)
    n_in = n_out = 0
    for p in sorted(glob.glob(work_path("train_feat/part_*.parquet"))):
        d = pl.read_parquet(p)
        n_in += d.height
        d = d.filter(~pl.col("s1").is_in(drop.implode()))
        cols = d.columns
        d = (query_context(d.drop("q_ncand", "q_maxb", "q_2ndb", "gap_qmax", "gap_q2nd", "r_qmax"))
               .sort(["qid", "bscore"], descending=[False, True])
               .with_columns(pl.int_range(pl.len()).over("qid").cast(pl.UInt8).alias("brank")))
        d.select(cols).write_parquet(os.path.join(outdir, os.path.basename(p)))
        n_out += d.height
    print(f"dropped {drop.len():,} S1 ({DROP:.0%}); pairs {n_in:,} -> {n_out:,}")


if __name__ == "__main__":
    main()
