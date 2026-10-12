"""Residual stacker on the final US/India probabilities (v9d).

The final seen-country probability is a blend of stage 3 and two older stage-2 versions (v5, v4),
with US near-twins capped at the v8 stage-2 p. A small LightGBM re-scores the uncertain region
(0.02 < p < 0.98) from the member probabilities and the decision context they imply: the query's
number of candidates / strong candidates, whether the pair is the query's top and the other
option's p, how many other records the S1 already holds (per source) and their summed p, the
number of competing strong records for the S1, record source, empty address, country, and exact
name equality. Trained on the simulation with the stage-2 query folds.
Hardened use on test: removals everywhere, additions only where the test has no known decoy
shift (India): the US test holds 1.7-3.6x more decoy families than the simulation.

Usage: BER_WORK=work/aug BER_SIM=1 python resid_stack.py     (train on the simulation, report F)
"""
import lightgbm as lgb
import numpy as np
import polars as pl

from config import work_path
from make_final import near_twin_pairs, no_raise, rescue
from train import assign, cap_links, with_fold

QOFF, DUP = 8_000_000, 20_000_000
LO, HI = 0.02, 0.98
THR, RESCUE = 0.70, 0.55
FEATS = ["p3", "m0", "m1", "p", "p2", "q_ncand", "q_n01", "q_top", "q_pother", "s_ncomp03", "src3", "s_nl_oth",
         "s_nl3_oth", "s_psum_oth", "s_nl2_oth", "noaddr", "us", "s_noaddr", "name_eq"]
PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=200, feature_fraction=0.9,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, num_threads=3, verbose=-1, seed=1)
ROUNDS = 300
FILES = {"train": {"p3": "train_oof3_sim.parquet", "m0": "../v5/train_oof2_sim.parquet", "m1": "../v4/train_oof2_sim.parquet",
                   "p2": "train_oof2_sim.parquet"},
         "test": {"p3": "test_pred3_sim_tx.parquet", "m0": "v5/test_pred2_sim_tx.parquet", "m1": "v4/test_pred2_sim.parquet",
                  "p2": "test_pred2_sim_tx.parquet"}}


def queries(split):
    return pl.concat([pl.read_parquet(work_path(f"{split}_s{k}.parquet"), columns=["idx", "country", "addr_raw", "name_full"])
                        .select((pl.col("idx") + off).cast(pl.UInt32).alias("qid"), "country",
                                (pl.col("addr_raw").str.strip_chars() == "").cast(pl.Int8).alias("noaddr"),
                                (pl.col("country") == "US").cast(pl.Int8).alias("us"), pl.col("name_full").hash(seed=5).alias("hq"))
                      for k, off in ((2, 0), (3, QOFF))])


def members(split, seen):
    """Member probabilities of the seen-country records and the blended, near-twin-capped p."""
    def rd(f, name):
        return (pl.read_parquet(work_path(f), columns=["qid", "s1", "p"])
                  .with_columns(pl.col("qid").cast(pl.UInt32), pl.col("s1").cast(pl.UInt32), pl.col("p").cast(pl.Float32))
                  .filter(pl.col("qid") < DUP).join(seen, on="qid", how="semi").rename({"p": name}))
    f = FILES[split]
    b = (rd(f["p3"], "p3").join(rd(f["m0"], "m0"), on=["qid", "s1"], how="full", coalesce=True)
                          .join(rd(f["m1"], "m1"), on=["qid", "s1"], how="full", coalesce=True))
    b = b.with_columns((0.5 * pl.col("p3").fill_null(pl.mean_horizontal("m0", "m1")) + 0.25 * pl.col("m0").fill_null(pl.col("p3"))
                        + 0.25 * pl.col("m1").fill_null(pl.col("p3"))).cast(pl.Float32).alias("p"))
    p2 = rd(f["p2"], "p")
    us = seen.filter(pl.col("country") == "US").select("qid")
    near = near_twin_pairs(b.select("qid", "s1").join(us, on="qid", how="semi"), split=split)
    b = b.drop("p").join(no_raise(b.select("qid", "s1", "p"), p2, near), on=["qid", "s1"])
    return (b.join(p2.rename({"p": "p2"}), on=["qid", "s1"], how="left")
             .join(near.with_columns(pl.lit(1, pl.Int8).alias("near")), on=["qid", "s1"], how="left")
             .with_columns(pl.col("near").fill_null(0)))


def region_rows(P, links, q, split):
    """Feature rows of the uncertain region; links = the current decisions (qid, s1, p)."""
    qs = P.group_by("qid").agg(pl.col("p").top_k(2).alias("tk"), pl.len().alias("q_ncand"), (pl.col("p") > 0.1).sum().alias("q_n01"))
    qs = qs.select("qid", "q_ncand", "q_n01", pl.col("tk").list.get(0).alias("q_p1"),
                   pl.col("tk").list.get(1, null_on_oob=True).fill_null(0).alias("q_p2"))
    scomp = P.filter(pl.col("p") > 0.3).group_by("s1").len().rename({"len": "s_ncomp03"})
    R = P.filter((pl.col("p") > LO) & (pl.col("p") < HI)).join(qs, on="qid")
    R = R.with_columns((pl.col("p") >= pl.col("q_p1")).cast(pl.Int8).alias("q_top"))
    R = R.with_columns(pl.when(pl.col("q_top") == 1).then(pl.col("q_p2")).otherwise(pl.col("q_p1")).alias("q_pother")).drop("q_p1", "q_p2")
    L = links.select("qid", "s1", pl.col("p").alias("pl"))
    agg = (L.with_columns((pl.col("qid") >= QOFF).alias("s3")).group_by("s1")
            .agg(pl.len().alias("s_nl"), pl.col("s3").sum().alias("s_nl3"), pl.col("pl").sum().alias("s_psum")))
    R = (R.join(agg, on="s1", how="left").with_columns(pl.col("s_nl", "s_nl3", "s_psum").fill_null(0))
          .join(scomp, on="s1", how="left").with_columns(pl.col("s_ncomp03").fill_null(0))
          .join(L.select("qid", "s1", pl.lit(1).alias("is_link")), on=["qid", "s1"], how="left").with_columns(pl.col("is_link").fill_null(0))
          .with_columns((pl.col("qid") >= QOFF).cast(pl.Int8).alias("src3")))
    R = R.with_columns((pl.col("s_nl") - pl.col("is_link")).alias("s_nl_oth"),
                       (pl.col("s_nl3") - pl.col("is_link") * pl.col("src3")).alias("s_nl3_oth"),
                       (pl.col("s_psum") - pl.col("is_link") * pl.col("p")).alias("s_psum_oth")).drop("s_nl", "s_nl3", "s_psum")
    R = R.with_columns((pl.col("s_nl_oth") - pl.col("s_nl3_oth")).alias("s_nl2_oth"))
    R = R.join(q.select("qid", "noaddr", "us", "hq"), on="qid", how="left")
    s = pl.read_parquet(work_path(f"{split}_s1.parquet"), columns=["idx", "name_full", "addr_raw"]).select(
        pl.col("idx").cast(pl.UInt32).alias("s1"), pl.col("name_full").hash(seed=5).alias("hs"),
        (pl.col("addr_raw").str.strip_chars() == "").cast(pl.Int8).alias("s_noaddr"))
    return R.join(s, on="s1", how="left").with_columns((pl.col("hq") == pl.col("hs")).cast(pl.Int8).alias("name_eq")).drop("hq", "hs")


def decide(P, cap=True):
    L = rescue(P, assign(P, THR), RESCUE)
    return cap_links(L) if cap else L


def new_p(P, R, models):
    """Stacker p inside the region (US near-twins never raised), P's p elsewhere."""
    X = R.select(FEATS).to_numpy().astype(np.float32)
    pn = np.mean([m.predict(X) for m in models], axis=0).astype(np.float32)
    R = R.with_columns(pl.Series("pn", pn))
    R = R.with_columns(pl.when(pl.col("near") == 1).then(pl.min_horizontal("pn", pl.col("p2").fill_null(0), pl.col("p")))
                         .otherwise(pl.col("pn")).alias("pn"))
    return (P.select("qid", "s1", "p").join(R.select("qid", "s1", "pn"), on=["qid", "s1"], how="left")
             .with_columns(pl.coalesce("pn", "p").alias("p")).drop("pn"))


def train():
    from evaluate import eval_population, f05_macro
    q = queries("train")
    seen = q.filter(pl.col("country").is_in(["US", "India"])).select("qid", "country")
    P = members("train", seen)
    links = decide(P.select("qid", "s1", "p"))
    truth, ids = eval_population()
    base = f05_macro(links.select("qid", "s1"), truth, ids)[0]
    R = region_rows(P, links, q, "train")
    R = with_fold(R.join(truth.rename({"s1_true": "s1"}).with_columns(pl.lit(1, pl.Int8).alias("label")), on=["qid", "s1"], how="left")
                   .with_columns(pl.col("label").fill_null(0))).drop("_b")
    X, y, fo = R.select(FEATS).to_numpy().astype(np.float32), R["label"].to_numpy(), R["fold"].to_numpy()
    pr = np.zeros(R.height, dtype=np.float32)
    for f in (0, 1):
        m = lgb.train(PARAMS, lgb.Dataset(X[fo == f], y[fo == f]), ROUNDS)
        m.save_model(work_path(f"resid_fold{f}_sim.txt"))
        pr[fo != f] = m.predict(X[fo != f])
    R = R.with_columns(pl.Series("pn", pr))
    R = R.with_columns(pl.when(pl.col("near") == 1).then(pl.min_horizontal("pn", pl.col("p2").fill_null(0), pl.col("p")))
                         .otherwise(pl.col("pn")).alias("pn"))
    Pn = (P.select("qid", "s1", "p").join(R.select("qid", "s1", "pn"), on=["qid", "s1"], how="left")
            .with_columns(pl.coalesce("pn", "p").alias("p")).drop("pn"))
    new = decide(Pn)
    print(f"region rows {R.height:,}; v9c decision {base:.5f} -> residual stacker {f05_macro(new.select('qid', 's1'), truth, ids)[0]:.5f}")
    # hardened variant: no additions for US records
    add = new.join(links, on=["qid", "s1"], how="anti").join(seen, on="qid")
    rem = links.join(new, on=["qid", "s1"], how="anti")
    hard = cap_links(pl.concat([links.join(rem, on=["qid", "s1"], how="anti"), add.filter(pl.col("country") == "India").select(links.columns)]))
    print(f"hardened (removals + India additions): {f05_macro(hard.select('qid', 's1'), truth, ids)[0]:.5f}")


def test_changes(model_dir=None, us_adds=False):
    """Links to remove / add relative to the v9c seen-country decision on test (before the cap)."""
    import os
    q = queries("test")
    seen = q.filter(pl.col("country").is_in(["US", "India"])).select("qid", "country")
    P = members("test", seen)
    links = decide(P.select("qid", "s1", "p"), cap=False)
    R = region_rows(P, links, q, "test")
    d = model_dir or os.path.dirname(work_path("x"))
    models = [lgb.Booster(model_file=os.path.join(d, f"resid_fold{f}_sim.txt")) for f in (0, 1)]
    new = decide(new_p(P, R, models), cap=False)
    add = new.join(links.select("qid", "s1"), on=["qid", "s1"], how="anti").join(seen, on="qid")
    if not us_adds:
        add = add.filter(pl.col("country") != "US")
    rem = links.join(new.select("qid", "s1"), on=["qid", "s1"], how="anti")
    print(f"residual stacker on test: {links.height:,} seen-country links; remove {rem.height:,}, add {add.height:,}")
    return rem.select("qid", "s1"), add.select("qid", "s1", "p")


if __name__ == "__main__":
    train()
