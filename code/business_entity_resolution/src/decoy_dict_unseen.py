"""Decoy-word log-odds for countries without training labels (label-free transfer).

On train, a word that appears only on one side of a top-1 blocking pair is a decoy marker
almost exactly when such pairs usually have a *different* house number (decoys sit at a
nearby number; noise words do not move the address): the rank correlation between a word's
house-number-mismatch rate and its learned decoy log-odds is ~0.76, and words with a
mismatch rate above ~0.6 are decoy words 92-99% of the time.

For a word w the mismatch rate is a mixture: rate_w = pi_w * r_neg + (1 - pi_w) * r_pos,
with r_pos / r_neg the mismatch rates of true / false top-1 pairs (learned on train), so
pi_w (the share of w's pairs that are non-matches) = (rate_w - r_pos) / (r_neg - r_pos).
On train words this estimate correlates 0.92 with the true share, and the learned log-odds
follow lo = a + b * logit(pi) (corr 0.97). Pure decoy words sometimes keep the house number,
so for words with a mismatch rate >= 0.75 an isotonic map rate -> log-odds (fit on train
words) is used instead. Only query-side words are scored (the S1 side does not transfer).
No test labels are involved.

Output: work/decoy_dict_unseen.parquet (country, token, side, lo)
Usage: python decoy_dict_unseen.py
"""
import glob

import numpy as np
import polars as pl
from sklearn.isotonic import IsotonicRegression

from config import work_path

QOFF = 8_000_000
MIN_N = 300


def word_signature(split):
    """(country, side, token, n, rate) over top-1 pairs with house numbers on both sides,
    aggregated candidate part by candidate part to bound memory."""
    q = pl.concat([pl.read_parquet(work_path(f"{split}_s{k}.parquet"), columns=["idx", "name_full", "addr_hn", "country"])
                     .select((pl.col("idx") + off).cast(pl.UInt32).alias("qid"), "name_full",
                             pl.col("addr_hn").alias("hn"), "country")
                   for k, off in ((2, 0), (3, QOFF))])
    s1 = pl.read_parquet(work_path(f"{split}_s1.parquet"), columns=["idx", "name_full", "addr_hn"]).select(
        pl.col("idx").cast(pl.UInt32).alias("s1"), pl.col("name_full").alias("nf1"),
        pl.col("addr_hn").alias("hn1"))
    qt = pl.col("name_full").str.split(" ").list.unique()
    st = pl.col("nf1").str.split(" ").list.unique()
    words, bases = [], []
    for p in sorted(glob.glob(work_path(f"{split}_cand/part_*.parquet"))):
        c = pl.read_parquet(p, columns=["qid", "s1", "brank"]).filter(pl.col("brank") == 0).drop("brank")
        d = (c.join(q, on="qid").join(s1, on="s1")
              .filter(pl.col("hn").is_not_null() & pl.col("hn1").is_not_null() & (pl.col("hn") != "") & (pl.col("hn1") != ""))
              .select("country", (pl.col("hn") != pl.col("hn1")).cast(pl.Int32).alias("hn_diff"),
                      qt.list.set_difference(st).alias("q"), st.list.set_difference(qt).alias("s")))
        bases.append(d.group_by("country").agg(pl.col("hn_diff").sum().alias("k"), pl.len().alias("m")))
        for side in ("q", "s"):
            words.append(d.select("country", "hn_diff", pl.col(side).alias("token")).explode("token").drop_nulls("token")
                          .group_by("country", "token").agg(pl.len().alias("n"), pl.col("hn_diff").sum().alias("k"))
                          .with_columns(pl.lit(side).alias("side")))
    base = (pl.concat(bases).group_by("country").agg(pl.col("k").sum(), pl.col("m").sum())
              .select("country", (pl.col("k") / pl.col("m")).alias("base")))
    w = (pl.concat(words).group_by("country", "token", "side").agg(pl.col("n").sum(), pl.col("k").sum())
           .select("country", "token", "side", "n", (pl.col("k") / pl.col("n")).alias("rate")))
    return w.join(base, on="country"), base


def main():
    tr, trb = word_signature("train")
    ddict = pl.read_parquet(work_path("decoy_dict.parquet")).filter(pl.col("dfold") == 2).drop("dfold")
    train_base = float((trb["base"]).mean())
    tr = (tr.group_by("side", "token").agg(pl.col("n").sum(), ((pl.col("rate") - pl.col("base")) * pl.col("n")).sum()
                                           / pl.col("n").sum()).rename({"rate": "excess"})
            .join(ddict, on=["token", "side"]).filter(pl.col("n") >= MIN_N))
    seen = pl.read_parquet(work_path("train_s1.parquet"), columns=["country"])["country"].unique()
    te, teb = word_signature("test")
    te = te.filter(~pl.col("country").is_in(seen.implode()) & (pl.col("n") >= MIN_N)).with_columns(
        (pl.col("rate") - pl.col("base")).alias("excess"))
    r_pos, r_neg, a, b = np.load(work_path("mixture_params.npy"))
    t = tr.filter(pl.col("side") == "q")
    iso = IsotonicRegression(out_of_bounds="clip").fit(t["excess"].to_numpy(), t["lo"].to_numpy(),
                                                       sample_weight=np.sqrt(t["n"].to_numpy()))
    u = te.filter(pl.col("side") == "q")
    pi = np.clip((u["rate"].to_numpy() - r_pos) / (r_neg - r_pos), 1e-3, 1 - 1e-3)
    lo_mix = a + b * np.log(pi / (1 - pi))
    lo_iso = iso.predict(u["excess"].to_numpy())
    lo = np.where(u["rate"].to_numpy() >= 0.75, lo_iso, lo_mix).astype(np.float32)
    out = u.with_columns(pl.Series("lo", lo), pl.Series("pi", pi.astype(np.float32))).select(
        "country", "token", "side", "lo", "pi", "n", "rate")
    print(f"mixture: r_pos {r_pos:.3f} r_neg {r_neg:.3f}, lo = {a:.2f} + {b:.2f} logit(pi); {u.height} words scored")
    out.write_parquet(work_path("decoy_dict_unseen.parquet"))
    print(f"train baseline mismatch {train_base:.3f}; test baselines:", dict(zip(teb["country"], teb["base"].round(3))))
    top = out.filter(pl.col("side") == "q").sort("n", descending=True).head(40)
    print([(t, round(float(lo), 1)) for t, lo in zip(top["token"], top["lo"])])


if __name__ == "__main__":
    main()
