"""Diagnostic: upper bounds per error class on the (simulated) OOF predictions.

Each oracle fixes one error class perfectly and re-scores, showing the most that
improving that part of the pipeline could deliver.
Usage: BER_SIM=1 python oracle_analysis.py
"""
import glob
import json

import polars as pl

from config import TAG, TRAIN_FEAT, work_path
from evaluate import eval_population, f05_macro
from train import assign, cap_links

if __name__ == "__main__":
    thr = json.load(open(work_path(f"model2_meta{TAG}.json")))["threshold"]
    truth, ids = eval_population()
    oof2 = pl.read_parquet(work_path(f"train_oof2{TAG}.parquet"), columns=["qid", "s1", "p"])
    base = cap_links(assign(oof2, thr)).select("qid", "s1")
    tp = base.join(truth, left_on=["qid", "s1"], right_on=["qid", "s1_true"], how="semi")
    # true pairs available at each stage
    in_s2 = truth.join(oof2, left_on=["qid", "s1_true"], right_on=["qid", "s1"], how="semi")
    cand = pl.concat([pl.read_parquet(p, columns=["qid", "s1", "label", "c_tset", "a_tset", "hn_eq"])
                        .filter(pl.col("label") == 1) for p in glob.glob(work_path(f"{TRAIN_FEAT}/part_*.parquet"))])
    in_cand = truth.join(cand, left_on=["qid", "s1_true"], right_on=["qid", "s1"], how="semi")
    f = lambda pred: f05_macro(pred.select("qid", "s1"), truth, ids)[0]
    r = lambda d: d.rename({"s1_true": "s1"}).select("qid", "s1")
    print(f"true pairs: {truth.height:,} | in candidates {in_cand.height/truth.height:.4f} | "
          f"in stage-2 set {in_s2.height/truth.height:.4f} | linked correctly {tp.height/truth.height:.4f}")
    print(f"base                                  {f(base):.5f}")
    print(f"O1 no false links                     {f(tp):.5f}")
    extra2 = r(in_s2).join(base, on="qid", how="anti")
    print(f"O2 + all true pairs in stage-2 set    {f(pl.concat([base, extra2])):.5f}")
    print(f"O1+O2                                 {f(pl.concat([tp, extra2])):.5f}")
    extra3 = r(in_cand).join(base, on="qid", how="anti")
    print(f"O1+O3 perfect model on candidates     {f(pl.concat([tp, extra3])):.5f}")
    print(f"O4 perfect blocking + model           1.00000")
    # what do the missed-but-in-candidates pairs look like?
    miss = cand.join(base, on=["qid", "s1"], how="anti").join(truth, left_on=["qid", "s1"], right_on=["qid", "s1_true"], how="semi")
    print("missed true pairs in candidates by name / address similarity:")
    print(miss.with_columns(
        pl.when(pl.col("c_tset") >= 80).then(pl.lit("name>=80")).when(pl.col("c_tset") >= 50).then(pl.lit("name50-80")).otherwise(pl.lit("name<50")).alias("nm"),
        pl.when(pl.col("a_tset") >= 80).then(pl.lit("addr>=80")).when(pl.col("a_tset") >= 0).then(pl.lit("addr<80")).otherwise(pl.lit("addr empty")).alias("ad"))
          .group_by("nm", "ad").len().sort("len", descending=True))
