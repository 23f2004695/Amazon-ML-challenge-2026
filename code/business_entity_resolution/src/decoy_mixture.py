"""Fits the house-number mismatch mixture (r_pos, r_neg) and lo ~ logit(pi) on train words;
writes work/mixture_params.npy used by decoy_dict_unseen.py. Usage: python decoy_mixture.py"""

import glob, numpy as np, polars as pl
from config import work_path
from evaluate import load_truth
truth = load_truth().with_columns(pl.col("qid").cast(pl.UInt32), pl.col("s1_true").cast(pl.UInt32))
q = pl.concat([pl.read_parquet(work_path(f"train_s{k}.parquet"), columns=["idx", "name_full", "addr_hn"]).select((pl.col("idx") + o).cast(pl.UInt32).alias("qid"), "name_full", pl.col("addr_hn").alias("hn")) for k, o in ((2, 0), (3, 8_000_000))])
s1 = pl.read_parquet(work_path("train_s1.parquet"), columns=["idx", "name_full", "addr_hn"]).select(pl.col("idx").cast(pl.UInt32).alias("s1"), pl.col("name_full").alias("nf1"), pl.col("addr_hn").alias("hn1"))
qt = pl.col("name_full").str.split(" ").list.unique(); st = pl.col("nf1").str.split(" ").list.unique()
W, B = [], []
for p in sorted(glob.glob(work_path("train_cand/part_*.parquet"))):
    c = pl.read_parquet(p, columns=["qid", "s1", "brank"]).filter(pl.col("brank") == 0).drop("brank")
    d = (c.join(truth, on="qid", how="left").with_columns((pl.col("s1") == pl.col("s1_true")).fill_null(False).cast(pl.Int32).alias("y"))
          .join(q, on="qid").join(s1, on="s1").filter(pl.col("hn").is_not_null() & pl.col("hn1").is_not_null() & (pl.col("hn") != "") & (pl.col("hn1") != ""))
          .select("y", (pl.col("hn") != pl.col("hn1")).cast(pl.Int32).alias("h"), qt.list.set_difference(st).alias("x")))
    B.append(d.group_by("y").agg(pl.col("h").sum().alias("k"), pl.len().alias("m")))
    W.append(d.explode("x").drop_nulls("x").group_by("x").agg(pl.len().alias("n"), pl.col("h").sum().alias("k"), (1 - pl.col("y")).sum().alias("neg")))
B = pl.concat(B).group_by("y").agg(pl.col("k").sum(), pl.col("m").sum()).with_columns((pl.col("k") / pl.col("m")).alias("r")).sort("y")
r_neg, r_pos = B["r"][0], B["r"][1]
print(f"train top-1 pairs: hn-mismatch rate | non-match r_neg = {r_neg:.3f}, true match r_pos = {r_pos:.3f}")
W = pl.concat(W).group_by("x").agg(pl.col("n").sum(), pl.col("k").sum(), pl.col("neg").sum()).filter(pl.col("n") >= 300)
W = W.with_columns((pl.col("k") / pl.col("n")).alias("rate"), (pl.col("neg") / pl.col("n")).alias("pi_true"),
                   ((pl.col("k") / pl.col("n") - r_pos) / (r_neg - r_pos)).clip(0.001, 0.999).alias("pi_est"))
print("words:", W.height, " corr(pi_est, pi_true) =", round(W.select(pl.corr("pi_est", "pi_true")).item(), 3),
      " mean abs err =", round((W["pi_est"] - W["pi_true"]).abs().mean(), 3))
for w in ["group", "holdings", "exports", "industries", "services", "center", "inc", "llc", "co", "ltd", "the", "corporation", "partners", "private", "limited", "pvt"]:
    x = W.filter(pl.col("x") == w)
    if x.height: print(f"  {w:12s} n={x['n'][0]:7d} rate={x['rate'][0]:.3f} pi_true={x['pi_true'][0]:.3f} pi_est={x['pi_est'][0]:.3f}")
dd = pl.read_parquet(work_path("decoy_dict.parquet")).filter((pl.col("dfold") == 2) & (pl.col("side") == "q")).select(pl.col("token").alias("x"), "lo")
m = W.join(dd, on="x")
m = m.with_columns((pl.col("pi_true") / (1 - pl.col("pi_true"))).clip(1e-4, 1e4).log().alias("logit_true"),
                   (pl.col("pi_est") / (1 - pl.col("pi_est"))).log().alias("logit_est"))
A = np.vstack([np.ones(m.height), m["logit_true"].to_numpy()]).T
coef = np.linalg.lstsq(A, m["lo"].to_numpy(), rcond=None)[0]
print(f"lo ~= {coef[0]:.2f} + {coef[1]:.2f} * logit(pi_true)   (fit on train words); corr(lo, logit_true) = {m.select(pl.corr('lo','logit_true')).item():.3f}")
np.save(work_path("mixture_params.npy"), np.array([r_pos, r_neg, coef[0], coef[1]]))
