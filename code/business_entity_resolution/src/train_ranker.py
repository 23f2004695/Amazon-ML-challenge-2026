"""Train the blocking re-ranker on deep (top-60 by IDF) candidates of 1/40 of train queries.

It reports recall@K of true pairs on held-out queries for plain IDF ranking vs the
re-ranker, so the benefit is measured before the full blocking run.
Usage: python blocking.py train sample && python train_ranker.py
"""
import lightgbm as lgb
import numpy as np
import polars as pl

from blocking import RR_FEATS
from config import SEED, work_path
from evaluate import load_truth

if __name__ == "__main__":
    d = pl.read_parquet(work_path("train_deep_sample.parquet"))
    d = d.with_columns((pl.col("qid").hash(seed=17) % 5 == 0).alias("hold"))
    tr, ho = d.filter(~pl.col("hold")), d.filter(pl.col("hold"))
    ds = lgb.Dataset(tr.select(RR_FEATS).to_numpy().astype(np.float32), label=tr["label"].to_numpy())
    params = dict(objective="binary", learning_rate=0.1, num_leaves=63, min_data_in_leaf=200,
                  feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1, num_threads=14,
                  verbose=-1, seed=SEED)
    m = lgb.train(params, ds, num_boost_round=300)
    ho = ho.with_columns(pl.Series("rs", m.predict(ho.select(RR_FEATS).to_numpy().astype(np.float32))))
    ho = ho.with_columns(pl.col("rs").rank("ordinal", descending=True).over("qid").alias("rrank"))
    # recall denominator: all true pairs of held-out queries (including those never retrieved)
    truth = load_truth().with_columns(pl.col("qid").cast(pl.UInt32))
    n_true = truth.join(ho.select("qid").unique(), on="qid", how="semi").height
    pos = ho.filter(pl.col("label") == 1)
    for k in (1, 3, 10, 15):
        print(f"recall@{k:2d}: IDF {(pos['irank'] < k).sum() / n_true:.4f}   re-ranked {(pos['rrank'] <= k).sum() / n_true:.4f}")
    print(f"retrieved in top-{60}: {pos.height / n_true:.4f}")
    imp = sorted(zip(m.feature_importance("gain"), RR_FEATS), reverse=True)
    print("gain:", [(n, int(g)) for g, n in imp])
    m = lgb.train(params, lgb.Dataset(d.select(RR_FEATS).to_numpy().astype(np.float32), label=d["label"].to_numpy()),
                  num_boost_round=300)
    m.save_model(work_path("ranker.txt"))
