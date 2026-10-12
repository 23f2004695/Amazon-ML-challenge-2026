"""Bagged stage 3: several LightGBM variants (seeds, leaves, feature / row sampling) trained on the
cached stage-3 rows (stage3.py writes them), averaged. Same query folds as stages 1-2.

Usage: BER_WORK=work/aug BER_SIM=1 python stage3_bag.py          (train variants, OOF report)
       BER_SIM=1 BER_UNSEEN=tx python stage3_bag.py test           (test, models from work/aug via BER_MODELS)
"""
import json
import os
import sys

import lightgbm as lgb
import numpy as np
import polars as pl

from config import SEED, TAG, work_path
from evaluate import eval_population, f05_macro
from stage3 import DUP, merge
from train import NON_FEATS, assign, cap_links

BASE = dict(objective="binary", learning_rate=0.05, num_leaves=127, min_data_in_leaf=200, feature_fraction=0.8,
            bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, num_threads=14, verbose=-1, seed=SEED)
VARIANTS = [dict(), dict(seed=7, num_leaves=63, feature_fraction=0.6), dict(seed=13, num_leaves=255, min_data_in_leaf=400, learning_rate=0.03),
            dict(seed=21, feature_fraction=0.5, bagging_fraction=0.7, lambda_l2=5.0)]
ROUNDS = [500, 700, 900, 600]
MODELS = os.environ.get("BER_MODELS", "")


def train():
    P = pl.read_parquet(work_path(f"stage3_rows{TAG}.parquet"))
    feats = [c for c in P.columns if c not in NON_FEATS]
    X = P.select(feats).to_numpy().astype(np.float32)
    y, fold = P["label"].to_numpy(), P["fold"].to_numpy()
    oof2 = pl.read_parquet(work_path(f"train_oof2{TAG}.parquet"), columns=["qid", "s1", "p"])
    truth, ids = eval_population()
    keys = P.select("qid", "s1")
    del P
    preds = []
    for v, (var, nr) in enumerate(zip(VARIANTS, ROUNDS)):
        pr = np.zeros(len(y), dtype=np.float32)
        for f in (0, 1):
            m = lgb.train({**BASE, **var}, lgb.Dataset(X[fold == f], label=y[fold == f]), num_boost_round=nr)
            m.save_model(work_path(f"lgb3b_v{v}_fold{f}{TAG}.txt"))
            pr[fold != f] = m.predict(X[fold != f])
        preds.append(pr)
        for name, p in ((f"variant {v}", pr), (f"mean of 0..{v}", np.mean(preds, axis=0))):
            o = merge(oof2, keys.with_columns(pl.Series("p", p))).filter(pl.col("qid") < DUP)
            print(f"{name}: " + "  ".join(f"thr {t}: {f05_macro(cap_links(assign(o, t)).select('qid', 's1'), truth, ids)[0]:.5f}" for t in (0.66, 0.70)), flush=True)
    merge(oof2, keys.with_columns(pl.Series("p", np.mean(preds, axis=0)))).write_parquet(work_path(f"train_oof3b{TAG}.parquet"))
    json.dump({"feats": feats, "n": len(VARIANTS)}, open(work_path(f"model3b_meta{TAG}.json"), "w"))


def predict_test():
    from predict import VAR, unseen_matrix
    meta = json.load(open(os.path.join(MODELS or os.path.dirname(work_path("x")), f"model3b_meta{TAG}.json")))
    P = pl.read_parquet(work_path(f"stage3_rows_test{TAG}{VAR}.parquet"))
    X = unseen_matrix(P, P.select(meta["feats"]).to_numpy().astype(np.float32), meta["feats"])
    keys = P.select("qid", "s1")
    del P
    ps = []
    for v in range(meta["n"]):
        ms = [lgb.Booster(model_file=os.path.join(MODELS or os.path.dirname(work_path("x")), f"lgb3b_v{v}_fold{f}{TAG}.txt")) for f in (0, 1)]
        ps.append(np.mean([m.predict(X) for m in ms], axis=0))
    p2 = pl.read_parquet(work_path(f"test_pred2{TAG}{VAR}.parquet"))
    merge(p2, keys.with_columns(pl.Series("p", np.mean(ps, axis=0).astype(np.float32)))).write_parquet(work_path(f"test_pred3b{TAG}{VAR}.parquet"))
    print(f"test bagged stage 3: {keys.height:,} region pairs")


if __name__ == "__main__":
    predict_test() if len(sys.argv) > 1 and sys.argv[1] == "test" else train()
