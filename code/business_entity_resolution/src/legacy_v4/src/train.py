"""Stage 4: train the pairwise LightGBM matcher with 2-fold CV grouped by query.

* fold(q) = hash(qid) % 2. Model A trains on a sample of fold-0 queries, model B on
  fold-1 queries; each predicts the other fold, giving out-of-fold (OOF) scores for
  every train pair.
* Decision rule: every S2/S3 record is linked to at most one S1 record (verified on
  the ground truth), so each query keeps only its best-scoring candidate and is linked
  when that score exceeds a threshold. The threshold is tuned on the exact macro F0.5
  over all train S1 records (singletons included).

Usage: python train.py
"""
import glob
import json
import time

import lightgbm as lgb
import numpy as np
import polars as pl

from config import SEED, TAG, TRAIN_FEAT, work_path
from evaluate import eval_population, f05_macro

SAMPLE_PCT = 12           # % of each fold's queries used for fitting
NON_FEATS = {"qid", "s1", "label", "fold"}
PARAMS = dict(objective="binary", learning_rate=0.08, num_leaves=255, min_data_in_leaf=100,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
              num_threads=14, verbose=-1, seed=SEED, max_bin=255)
N_ROUNDS = 600


def with_fold(df):
    h = pl.col("qid").hash(seed=11)
    return df.with_columns((h % 2).cast(pl.Int8).alias("fold"), (h % 100).alias("_b"))


def parts(split):
    d = TRAIN_FEAT if split == "train" else f"{split}_feat"
    return sorted(glob.glob(work_path(f"{d}/part_*.parquet")))


def load_sample():
    frames = []
    for p in parts("train"):
        d = with_fold(pl.read_parquet(p))
        frames.append(d.filter(pl.col("_b") < SAMPLE_PCT).drop("_b"))
    return pl.concat(frames)


def fit(df, feats):
    X = df.select(feats).to_numpy().astype(np.float32)
    ds = lgb.Dataset(X, label=df["label"].to_numpy(), feature_name=feats, free_raw_data=True)
    return lgb.train(PARAMS, ds, num_boost_round=N_ROUNDS)


def assign(pred, thr):
    """Best candidate per query, kept if its probability exceeds thr."""
    return best_per_query(pred).filter(pl.col("p") > thr)


MAX_PER_SOURCE = {0: 5, 1: 6}  # max S2 / S3 matches of one S1 in the train ground truth


def cap_links(matches):
    """Keep at most 5 S2 and 6 S3 links per S1 (highest probability first)."""
    src = (pl.col("qid") >= 8_000_000).cast(pl.Int8).alias("src")
    m = matches.with_columns(src).with_columns(
        pl.col("p").rank("ordinal", descending=True).over("s1", "src").alias("_r"))
    cap = pl.when(pl.col("src") == 0).then(MAX_PER_SOURCE[0]).otherwise(MAX_PER_SOURCE[1])
    return m.filter(pl.col("_r") <= cap).drop("src", "_r")


def best_per_query(pred):
    """(qid, s1, p) of the highest-probability candidate of every query."""
    return (pred.lazy().group_by("qid")
                .agg(pl.col("s1").sort_by("p").last(), pl.col("p").max())
                .collect(engine="streaming"))


def main():
    t0 = time.time()
    sample = load_sample()
    feats = [c for c in sample.columns if c not in NON_FEATS]
    print(f"sample {sample.height:,} rows, {len(feats)} feats, pos rate {sample['label'].mean():.3f} "
          f"[{time.time() - t0:.0f}s]", flush=True)
    models = {}
    for f in (0, 1):
        models[f] = fit(sample.filter(pl.col("fold") == f), feats)
        models[f].save_model(work_path(f"lgb_fold{f}{TAG}.txt"))
        print(f"model {f} trained [{time.time() - t0:.0f}s]", flush=True)
    del sample
    imp = sorted(zip(models[0].feature_importance("gain"), feats), reverse=True)
    print("top gain:", [(n, int(g)) for g, n in imp[:20]])

    oof = []
    for p in parts("train"):
        d = with_fold(pl.read_parquet(p))
        X = d.select(feats).to_numpy().astype(np.float32)
        fold = d["fold"].to_numpy()
        pr = np.where(fold == 0, models[1].predict(X), models[0].predict(X))
        oof.append(d.select("qid", "s1", "label").with_columns(pl.Series("p", pr.astype(np.float32))))
    oof = pl.concat(oof)
    oof.write_parquet(work_path(f"train_oof{TAG}.parquet"))
    print(f"oof predicted [{time.time() - t0:.0f}s]", flush=True)

    truth, n_s1 = eval_population()
    best = None
    for thr in np.arange(0.2, 0.95, 0.05):
        score, _ = f05_macro(assign(oof, thr), truth, n_s1)
        print(f"thr {thr:.2f}: F0.5 {score:.5f}")
        if best is None or score > best[1]:
            best = (float(thr), score)
    print("best", best)
    json.dump({"threshold": best[0], "oof_f05": best[1], "feats": feats}, open(work_path(f"model_meta{TAG}.json"), "w"))


if __name__ == "__main__":
    main()
