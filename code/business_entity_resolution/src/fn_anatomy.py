"""Diagnostic: anatomy of true pairs that reach stage 2 but are not linked (v5, simulation)."""
import glob
import json

import polars as pl

from config import TAG, TRAIN_FEAT, work_path
from evaluate import eval_population
from train import best_per_query

if __name__ == "__main__":
    thr = json.load(open(work_path(f"model2_meta{TAG}.json")))["threshold"]
    truth, ids = eval_population()
    oof = pl.read_parquet(work_path(f"train_oof2{TAG}.parquet"), columns=["qid", "s1", "p", "label"])
    best = best_per_query(oof).rename({"s1": "best_s1", "p": "best_p"})
    pos = oof.filter(pl.col("label") == 1).join(best, on="qid")
    pos = pos.with_columns((pl.col("p") > thr).alias("linked") & (pl.col("best_s1") == pl.col("s1")),
                           (pl.col("best_s1") == pl.col("s1")).alias("true_is_top"))
    cols = ["qid", "s1", "c_tset", "a_tset", "qa_empty", "sa_empty", "hn_eq", "hn_logdiff", "namefreq", "dq_max", "q_ncand"]
    f = pl.concat([pl.read_parquet(p, columns=cols).join(pos.select("qid", "s1"), on=["qid", "s1"], how="semi")
                   for p in sorted(glob.glob(work_path(f"{TRAIN_FEAT}/part_*.parquet")))])
    d = pos.join(f, on=["qid", "s1"], how="left")
    cat = (pl.when(pl.col("qa_empty") == 1).then(pl.lit("A: query has no address"))
             .when(pl.col("c_tset").is_null()).then(pl.lit("X: expansion pair"))
             .when(pl.col("c_tset") < 50).then(pl.lit("B: name replaced (<50)"))
             .when(pl.col("hn_eq") == 1).then(pl.lit("C: same house number"))
             .when(pl.col("hn_eq") == 0).then(pl.lit("D: house number differs"))
             .otherwise(pl.lit("E: house number missing")))
    d = d.with_columns(cat.alias("cat"))
    print(f"true pairs in stage 2: {d.height:,}, linked {d['linked'].sum():,}, missed {(~d['linked']).sum():,}")
    print(d.group_by("cat").agg(pl.len().alias("true_pairs"), (~pl.col("linked")).sum().alias("missed"),
                                (~pl.col("linked")).mean().round(4).alias("miss_rate"),
                                (~pl.col("true_is_top") & ~pl.col("linked")).sum().alias("missed_other_s1_on_top"),
                                pl.col("namefreq").filter(~pl.col("linked")).median().alias("namefreq_med_missed"))
           .sort("missed", descending=True))
    m = d.filter(~pl.col("linked"))
    print("missed pairs by p2:", m.select(pl.col("p").cut([0.1, 0.3, 0.5, 0.7]).value_counts()).unnest("p"))
    cols2 = ["idx", "name_raw", "addr_raw"]
    q = pl.concat([pl.read_parquet(work_path(f"train_s{k}.parquet"), columns=cols2).with_columns((pl.col("idx") + o).cast(pl.UInt32).alias("qid")).drop("idx") for k, o in ((2, 0), (3, 8_000_000))])
    s1 = pl.read_parquet(work_path("train_s1.parquet"), columns=cols2).with_columns(pl.col("idx").cast(pl.UInt32))
    for c in ("C: same house number", "D: house number differs", "A: query has no address"):
        x = m.filter(pl.col("cat") == c).sample(8, seed=4).join(q, on="qid").join(s1, left_on="s1", right_on="idx", suffix="_t").join(s1, left_on="best_s1", right_on="idx", suffix="_b")
        print(f"\n### missed, {c}")
        for r in x.iter_rows(named=True):
            print(f"p={r['p']:.2f} Q : {r['name_raw']} | {r['addr_raw']}\n       TRUE: {r['name_raw_t']} | {r['addr_raw_t']}" +
                  ("" if r["true_is_top"] else f"\n       TOP ({r['best_p']:.2f}): {r['name_raw_b']} | {r['addr_raw_b']}"))
