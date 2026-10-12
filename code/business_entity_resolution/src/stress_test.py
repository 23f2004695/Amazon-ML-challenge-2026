"""Diagnostic: score the withheld-S1 simulation (make_drop.py) with an existing model set.

Usage: BER_SIM=1 python stress_test.py <model_dir>   (e.g. work/v2 with lgb_fold*.txt,
lgb2_fold*.txt, model_meta.json, model2_meta.json). Each row is scored by the fold model
that did not train on its query, so the result is an out-of-fold estimate.
"""
import json
import multiprocessing as mp
import os
import sys

import lightgbm as lgb
import numpy as np
import polars as pl

from config import N_JOBS, SIM, work_path
from evaluate import eval_population, f05_macro
from stage2 import attach_stage1_feats, group_feats
from train import assign, parts, with_fold


def oof_predict(models, feats, frame_iter):
    out = []
    for d in frame_iter:
        d = with_fold(d)
        X = d.select(feats).to_numpy().astype(np.float32)
        fold = d["fold"].to_numpy()
        pr = np.where(fold == 0, models[1].predict(X), models[0].predict(X))
        out.append(d.select("qid", "s1").with_columns(pl.Series("p", pr.astype(np.float32))))
    return pl.concat(out)


if __name__ == "__main__":
    assert SIM, "run with BER_SIM=1"
    mdir = sys.argv[1]
    m1 = [lgb.Booster(model_file=os.path.join(mdir, f"lgb_fold{f}.txt")) for f in (0, 1)]
    f1 = json.load(open(os.path.join(mdir, "model_meta.json")))["feats"]
    p1 = oof_predict(m1, f1, (pl.read_parquet(p) for p in parts("train")))
    print("stage 1 scored", p1.height, flush=True)
    truth, ids = eval_population()
    for thr in (0.7,):
        print(f"stage1 thr {thr}: {f05_macro(assign(p1, thr), truth, ids)[0]:.5f}", flush=True)
    pool = mp.get_context("spawn").Pool(min(N_JOBS, 10))
    P = attach_stage1_feats(group_feats(p1.rename({"p": "p1"}), "train", pool), "train")
    pool.close()
    m2 = [lgb.Booster(model_file=os.path.join(mdir, f"lgb2_fold{f}.txt")) for f in (0, 1)]
    f2 = json.load(open(os.path.join(mdir, "model2_meta.json")))["feats"]
    p2 = oof_predict(m2, f2, [P])
    for thr in (0.66, 0.75, 0.85, 0.92):
        print(f"stage2 thr {thr}: {f05_macro(assign(p2, thr), truth, ids)[0]:.5f}", flush=True)
