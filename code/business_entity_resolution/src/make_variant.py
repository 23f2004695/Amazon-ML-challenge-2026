"""Compose a submission variant from saved stage-2 test predictions (no model re-run).

Used for controlled leaderboard experiments on the country without training labels:
  --fr-from FILE     take the predictions of records of unseen countries from FILE
                     (e.g. the no-decoy-vocabulary run test_pred2_sim_nd.parquet)
  --unseen-shift D   add D to the logit of every probability of unseen-country records
                     (D < 0: more conservative, D > 0: more liberal)
  --decision         expf (expected-F0.5 per S1, default) or thr (global threshold)
Writes output/matching_results_<name>.tsv and output/candidate_pairs_<name>.tsv.

Usage: BER_SIM=1 python make_variant.py --name fr_nd --fr-from test_pred2_sim_nd.parquet
"""
import argparse
import json
import os

import numpy as np
import polars as pl

from config import OUTPUT, TAG, work_path
from decision import optimize_links
from predict import id_maps, write_lists
import predict
from train import assign, cap_links

QOFF = 8_000_000


def unseen_queries():
    seen = pl.read_parquet(work_path("train_s1.parquet"), columns=["country"])["country"].unique()
    return pl.concat([pl.read_parquet(work_path(f"test_s{k}.parquet"), columns=["idx", "country"])
                        .filter(~pl.col("country").is_in(seen.implode()))
                        .select((pl.col("idx") + off).cast(pl.UInt32).alias("qid"))
                      for k, off in ((2, 0), (3, QOFF))])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--base", default=f"test_pred2{TAG}.parquet")
    ap.add_argument("--fr-from", default=None)
    ap.add_argument("--unseen-shift", type=float, default=0.0)
    ap.add_argument("--decision", default="expf", choices=["expf", "thr"])
    a = ap.parse_args()
    pred = pl.read_parquet(work_path(a.base)).with_columns(pl.col("qid").cast(pl.UInt32))
    cands = [pred.select("qid", "s1")]
    un = unseen_queries()
    if a.fr_from:
        alt = pl.read_parquet(work_path(a.fr_from)).with_columns(pl.col("qid").cast(pl.UInt32))
        pred = pl.concat([pred.join(un, on="qid", how="anti"), alt.join(un, on="qid", how="semi")])
        cands.append(alt.select("qid", "s1"))
    if a.unseen_shift:
        lg = (pl.col("p").clip(1e-6, 1 - 1e-6) / (1 - pl.col("p").clip(1e-6, 1 - 1e-6))).log() + a.unseen_shift
        pred = pred.with_columns(pl.when(pl.col("qid").is_in(un["qid"].implode()))
                                   .then(1 / (1 + (-lg).exp())).otherwise(pl.col("p")).cast(pl.Float32).alias("p"))
    thr = json.load(open(work_path(f"model2_meta{TAG}.json")))["threshold"]
    links = cap_links(optimize_links(pred) if a.decision == "expf" else assign(pred, thr))
    s1, predict.QMAP = id_maps()
    write_lists(links.select("s1", "qid"), s1, "matched_entity_ids", os.path.join(OUTPUT, f"matching_results_{a.name}.tsv"))
    base_c = [pl.read_parquet(work_path(f"test_pred1{TAG}.parquet"), columns=["qid", "s1"]).with_columns(pl.col("qid").cast(pl.UInt32))]
    write_lists(base_c + cands, s1, "candidate_entity_ids", os.path.join(OUTPUT, f"candidate_pairs_{a.name}.tsv"))
    print(f"variant {a.name}: {links.height:,} links "
          f"({links.join(un, on='qid', how='semi').height:,} for unseen-country records)")
