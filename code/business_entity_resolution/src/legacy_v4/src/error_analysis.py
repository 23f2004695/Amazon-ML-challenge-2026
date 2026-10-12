"""Diagnostic: break down OOF errors (by country, singleton status and error type)."""
import json
import sys

import polars as pl

from config import work_path
from evaluate import f05_macro, load_truth
from train import assign, best_per_query

if __name__ == "__main__":
    oof_file = sys.argv[1] if len(sys.argv) > 1 else "train_oof.parquet"
    meta_file = sys.argv[2] if len(sys.argv) > 2 else "model_meta.json"
    thr = json.load(open(work_path(meta_file)))["threshold"]
    oof = pl.read_parquet(work_path(oof_file), columns=["qid", "s1", "label", "p"])
    truth = load_truth().with_columns(pl.col("qid").cast(pl.UInt32), pl.col("s1_true").cast(pl.UInt32))
    s1 = pl.read_parquet(work_path("train_s1.parquet"), columns=["idx", "country"]).rename({"idx": "s1"})
    pred = assign(oof, thr)
    score, d = f05_macro(pred, truth, s1["s1"])
    d = d.join(s1, on="s1")
    print(f"F0.5 {score:.5f} at thr {thr}")
    print(d.group_by("country").agg(pl.col("f").mean(), pl.len()))
    print(d.group_by(pl.col("nt").clip(0, 6)).agg(pl.col("f").mean(), pl.len(),
          (pl.col("np") > 0).mean().alias("any_pred")).sort("nt"))
    # error taxonomy over queries
    best = best_per_query(oof)
    ptrue = (oof.lazy().filter(pl.col("label") == 1).select("qid", pl.col("p").alias("p_true"))
                .collect(engine="streaming"))
    oof = None
    allq = pl.concat([
        pl.read_parquet(work_path("train_s2.parquet"), columns=["idx"]).select(pl.col("idx").alias("qid")),
        pl.read_parquet(work_path("train_s3.parquet"), columns=["idx"]).select((pl.col("idx") + 8_000_000).alias("qid")),
    ]).with_columns(pl.col("qid").cast(pl.UInt32))
    e = (allq.join(best.select("qid", "s1", "p"), on="qid", how="left")
             .join(truth, on="qid", how="left")
             .join(ptrue, on="qid", how="left"))
    linked = pl.col("p") > thr
    cat = (pl.when(pl.col("s1_true").is_null() & linked).then(pl.lit("FP_unmatched_query_linked"))
             .when(pl.col("s1_true").is_null()).then(pl.lit("TN"))
             .when(linked & (pl.col("s1") == pl.col("s1_true"))).then(pl.lit("TP"))
             .when(linked).then(pl.lit("FP+FN_wrong_s1"))
             .when(pl.col("p_true").is_null()).then(pl.lit("FN_not_in_candidates"))
             .otherwise(pl.lit("FN_below_threshold")))
    print(e.with_columns(cat.alias("cat")).group_by("cat").len().sort("len", descending=True))
