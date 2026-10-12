"""Diagnostic: model-based expected macro F0.5 per country (no labels needed).

For every S1 the linked records (and the other records whose best S1 it is) are treated as
independently true with their predicted probability; the expected F0.5 of the chosen link
set is computed exactly (Poisson-binomial). On train this is compared with the real score
to check calibration; on test it shows which country the model itself is unsure about.
Usage: [BER_SIM=1] python expected_f.py <split> <pred parquet> <threshold>
"""
import sys

import numpy as np
import polars as pl

from config import work_path
from decision import MAX_N, _pb
from train import best_per_query


def expected_f_rows(p, k):
    """p: (m x n) sorted desc probabilities, k: links chosen (same for all rows)."""
    m, n = p.shape
    tp = _pb(p[:, :k]) if k else np.ones((m, 1))
    r = _pb(p[:, k:]) if k < n else np.ones((m, 1))
    a = np.arange(tp.shape[1])[:, None]
    b = np.arange(r.shape[1])[None, :]
    if k == 0:
        f = (b == 0).astype(float) * np.ones((tp.shape[1], 1))
    else:
        f = np.where(a > 0, 1.25 * a / np.maximum(1.25 * a + 0.25 * b + (k - a), 1e-9), 0.0)
    return np.einsum("ma,mb,ab->m", tp, r, f)


if __name__ == "__main__":
    split, f, thr = sys.argv[1], sys.argv[2], float(sys.argv[3])
    best = best_per_query(pl.read_parquet(work_path(f), columns=["qid", "s1", "p"])).filter(pl.col("p") >= 0.02)
    best = best.sort(["s1", "p"], descending=[False, True]).with_columns(
        pl.int_range(pl.len()).over("s1").alias("r")).filter(pl.col("r") < MAX_N)
    best = best.with_columns(pl.len().over("s1").alias("n"), (pl.col("p") > thr).sum().over("s1").alias("k"))
    rows = []
    for (n, k), g in best.group_by("n", "k"):
        g = g.sort(["s1", "r"])
        p = np.clip(g["p"].to_numpy().reshape(-1, n).astype(np.float64), 1e-6, 1 - 1e-6)
        rows.append(pl.DataFrame({"s1": g["s1"].to_numpy().reshape(-1, n)[:, 0], "ef": expected_f_rows(p, k)}))
    ef = pl.concat(rows).with_columns(pl.col("s1").cast(pl.UInt32))
    s1 = pl.read_parquet(work_path(f"{split}_s1.parquet"), columns=["idx", "country"]).select(
        pl.col("idx").cast(pl.UInt32).alias("s1"), "country")
    if split == "train":
        from evaluate import eval_population
        _, ids = eval_population()
        s1 = s1.filter(pl.col("s1").is_in(ids.implode()))
    d = s1.join(ef, on="s1", how="left").with_columns(pl.col("ef").fill_null(1.0))  # no candidate -> assume singleton
    print(d.group_by("country").agg(pl.col("ef").mean().alias("expected_F0.5"), pl.len()).sort("country"))
    print("overall expected F0.5:", round(d["ef"].mean(), 5))
