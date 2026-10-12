"""Score the synthetic-decoy-family data (BER_WORK=work/aug) with a stage-2 model set.
Usage: BER_WORK=.../work/aug BER_SIM=1 python aug_eval.py <model dir>"""
import json
import multiprocessing as mp
import os
import sys

import lightgbm as lgb
import numpy as np
import polars as pl

from config import N_JOBS, work_path
from evaluate import eval_population, f05_macro
from stage2 import attach_stage1_feats, group_feats
from train import assign, cap_links, with_fold

if __name__ == "__main__":
    mdir = sys.argv[1]
    pool = mp.get_context("spawn").Pool(min(N_JOBS, 10))
    P = group_feats(pl.read_parquet(work_path("train_oof_sim.parquet"), columns=["qid", "s1", "p"]).rename({"p": "p1"}), "train", pool)
    P = with_fold(attach_stage1_feats(P, "train", pool)).drop("_b")
    pool.close()
    feats = json.load(open(os.path.join(mdir, "model2_meta_sim.json")))["feats"]
    m = [lgb.Booster(model_file=os.path.join(mdir, f"lgb2_fold{f}_sim.txt")) for f in (0, 1)]
    X = P.select(feats).to_numpy().astype(np.float32)
    fold = P["fold"].to_numpy()
    p2 = np.where(fold == 0, m[1].predict(X), m[0].predict(X)).astype(np.float32)
    pred = P.select("qid", "s1").with_columns(pl.Series("p", p2))
    truth, ids = eval_population()
    dup = pred.filter(pl.col("qid") >= 20_000_000)
    for thr in (0.66, 0.75, 0.85):
        links = cap_links(assign(pred, thr))
        nd = links.filter(pl.col("qid") >= 20_000_000).height
        orig = links.filter(pl.col("qid") < 20_000_000)
        print(f"thr {thr:.2f}: F0.5 with synthetic decoy families {f05_macro(links.select('qid','s1'), truth, ids)[0]:.5f} | "
              f"duplicates linked {nd:,} of {dup['qid'].n_unique():,} in stage 2 | links on originals {orig.height:,}", flush=True)
