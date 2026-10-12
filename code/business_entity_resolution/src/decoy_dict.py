"""Learn which "extra" name words mark a decoy (a different, sister business).

The data contains decoy records: a sister business of an S1 entity, usually at the same
street, with a word added to the name ("... Holdings", "... Exports", "... Industries")
and/or a nearby house number. Noise on true matches also adds words, but different ones
("... Services", "LLC", "Center"). For every name token that appears on only one side of
a labelled candidate pair we estimate

    lo(token, side) = log P(token extra | non-match) - log P(token extra | match)

from the top-3 blocking candidates of each train query.

To keep cross-validation honest the dictionary is built per query fold (the same
hash(qid) % 2 folds as train.py): rows of fold r are scored with the dictionary learned on
fold 1 - r (dfold = 1 - r); test rows use the dictionary learned on all of train (dfold = 2).

Output: work/decoy_dict.parquet (token, side, dfold, lo)
Usage: python decoy_dict.py
"""
import glob

import numpy as np
import polars as pl

from config import work_path
from evaluate import load_truth

QOFF = 8_000_000
MIN_COUNT = 30
TOP = 3


def extra_tokens(pairs):
    """pairs has name_full (query) and name_full_s (S1). Adds list columns qx / sx."""
    qt = pl.col("name_full").str.split(" ").list.unique()
    st = pl.col("name_full_s").str.split(" ").list.unique()
    return pairs.with_columns(qt.list.set_difference(st).alias("qx"), st.list.set_difference(qt).alias("sx"))


def main():
    truth = load_truth().with_columns(pl.col("qid").cast(pl.UInt32), pl.col("s1_true").cast(pl.UInt32))
    qn = pl.concat([pl.read_parquet(work_path(f"train_s{k}.parquet"), columns=["idx", "name_full"])
                      .select((pl.col("idx") + off).cast(pl.UInt32).alias("qid"), "name_full")
                    for k, off in ((2, 0), (3, QOFF))])
    sn = pl.read_parquet(work_path("train_s1.parquet"), columns=["idx", "name_full"]).select(
        pl.col("idx").cast(pl.UInt32).alias("s1"), pl.col("name_full").alias("name_full_s"))
    counts, totals = [], []
    for p in sorted(glob.glob(work_path("train_cand/part_*.parquet"))):
        c = (pl.read_parquet(p, columns=["qid", "s1", "brank"]).filter(pl.col("brank") < TOP)
               .join(truth, on="qid", how="left")
               .with_columns((pl.col("s1") == pl.col("s1_true")).fill_null(False).cast(pl.Int8).alias("label"),
                             (pl.col("qid").hash(seed=11) % 2).cast(pl.Int8).alias("fold"))
               .join(qn, on="qid").join(sn, on="s1"))
        c = extra_tokens(c)
        totals.append(c.group_by("fold", "label").len())
        for side, col in (("q", "qx"), ("s", "sx")):
            counts.append(c.select("fold", "label", pl.col(col).alias("token")).explode("token")
                           .drop_nulls("token").group_by("fold", "label", "token").len()
                           .with_columns(pl.lit(side).alias("side")))
    counts = pl.concat(counts).group_by("fold", "label", "token", "side").agg(pl.col("len").sum())
    totals = pl.concat(totals).group_by("fold", "label").agg(pl.col("len").sum())
    out = []
    for dfold, folds in ((0, [0]), (1, [1]), (2, [0, 1])):
        cc = counts.filter(pl.col("fold").is_in(folds)).group_by("token", "side", "label").agg(pl.col("len").sum())
        tt = totals.filter(pl.col("fold").is_in(folds)).group_by("label").agg(pl.col("len").sum())
        n_neg = tt.filter(pl.col("label") == 0)["len"].item()
        n_pos = tt.filter(pl.col("label") == 1)["len"].item()
        w = (cc.pivot(on="label", index=["token", "side"], values="len").fill_null(0)
               .rename({"0": "neg", "1": "pos"}))
        w = (w.filter(pl.col("neg") + pl.col("pos") >= MIN_COUNT)
              .with_columns((((pl.col("neg") + 1) / n_neg).log() - ((pl.col("pos") + 1) / n_pos).log())
                            .cast(pl.Float32).alias("lo"),
                            pl.lit(dfold, pl.Int8).alias("dfold"))
              .select("token", "side", "dfold", "lo"))
        out.append(w)
    out = pl.concat(out)
    out.write_parquet(work_path("decoy_dict.parquet"))
    top = out.filter((pl.col("dfold") == 2) & (pl.col("side") == "q")).sort("lo", descending=True)
    print(f"{out.height:,} entries; strongest query-side decoy words:",
          [(t, round(float(v), 1)) for t, v in zip(top["token"][:15], top["lo"][:15])])
    print("most noise-like:", [(t, round(float(v), 1)) for t, v in zip(top["token"][-10:], top["lo"][-10:])])


if __name__ == "__main__":
    main()
