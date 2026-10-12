"""Expected-F0.5 decision rule per Source-1 entity.

The metric is a macro average over S1 entities, so the value of a link depends on the
S1's other links: the first link of an S1 is worth far more than a fifth doubtful one.
Every query is first assigned to its best S1 (one S1 per S2/S3 record). Then, for every
S1 with candidate queries of probabilities p_1 >= p_2 >= ... >= p_n, we choose how many
of the top ones to link (k = 0..n) by maximising the expected per-S1 F0.5, assuming the
candidates are independently true with probability p_i:

    TP ~ PoissonBinomial(p_1..p_k), R ~ PoissonBinomial(p_k+1..p_n)  (missed true ones)
    F  = 1.25 TP / (1.25 TP + 0.25 (R + m0) + (k - TP)),  F = 1 if k = 0 and R + m0 = 0

m0 is the expected number of true matches the models never surface (a small constant
estimated on train); it is added as a fixed extra false-negative mass.
"""
import numpy as np
import polars as pl

from train import best_per_query

MAX_N = 12      # candidates per S1 considered (sorted by p)
P_MIN = 0.02    # queries whose best probability is below this are never linked


def _pb(p):
    """Poisson-binomial pmf for each row of p (m x n) -> (m x n+1)."""
    m, n = p.shape
    d = np.zeros((m, n + 1))
    d[:, 0] = 1.0
    for j in range(n):
        pj = p[:, j:j + 1]
        d[:, 1:j + 2] = d[:, 1:j + 2] * (1 - pj) + d[:, 0:j + 1] * pj
        d[:, 0] *= (1 - p[:, j])
    return d


def best_k(p, m0=0.0):
    """p: (m x n) probabilities sorted descending per row. Returns optimal k per row."""
    m, n = p.shape
    ef = np.zeros((m, n + 1))
    for k in range(n + 1):
        tp = _pb(p[:, :k]) if k else np.ones((m, 1))
        r = _pb(p[:, k:]) if k < n else np.ones((m, 1))
        a = np.arange(tp.shape[1])[:, None]            # TP values
        b = np.arange(r.shape[1])[None, :]             # missed-true values
        if k == 0:
            # F = 1 only when there is no true match at all (no expected unseen mass either)
            f = ((b == 0) & (m0 == 0)).astype(float) * np.ones((tp.shape[1], 1))
            if m0 > 0:
                f = np.exp(-m0) * (b == 0).astype(float) * np.ones((tp.shape[1], 1))
        else:
            fp = k - a
            denom = 1.25 * a + 0.25 * (b + m0) + fp
            f = np.where(a > 0, 1.25 * a / np.maximum(denom, 1e-9), 0.0)
        ef[:, k] = np.einsum("ma,mb,ab->m", tp, r, f)
    return ef.argmax(axis=1)


def optimize_links(pred, m0=0.0):
    """pred: DataFrame(qid, s1, p) for scored pairs. Returns DataFrame(qid, s1, p) of links."""
    best = best_per_query(pred).filter(pl.col("p") >= P_MIN)
    best = best.sort(["s1", "p"], descending=[False, True]).with_columns(
        pl.int_range(pl.len()).over("s1").alias("r"), pl.len().over("s1").alias("n"))
    head = best.filter(pl.col("r") < MAX_N).with_columns(pl.col("n").clip(upper_bound=MAX_N))
    keep = []
    for n, g in head.group_by("n"):
        n = n[0]
        g = g.sort(["s1", "r"])
        p = g["p"].to_numpy().reshape(-1, n).astype(np.float64)
        k = best_k(np.clip(p, 1e-6, 1 - 1e-6), m0)
        s1 = g["s1"].to_numpy().reshape(-1, n)[:, 0]
        keep.append(pl.DataFrame({"s1": s1, "k": k}).with_columns(pl.col("s1").cast(g["s1"].dtype)))
    ks = pl.concat(keep)
    return (head.join(ks, on="s1").filter(pl.col("r") < pl.col("k")).select("qid", "s1", "p"))
