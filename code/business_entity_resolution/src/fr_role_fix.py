"""Role-aware correction of the label-free decoy vocabulary for countries unseen in training.

decoy_dict_unseen.py scores a word by how often records carrying it sit at a different house
number than their best S1 (decoys move the house number, noise rarely does). In French data
'groupe', 'developpement' and 'france' play two roles:
  * decoy word  - added to an S1 name, at a nearby house number (mismatch rate ~0.95);
  * noise suffix - replacing one generic name word ('Chasse Club SARL' -> 'Chasse SARL Groupe'),
                   at the S1's own house number (mismatch rate 0.06-0.17, like the noise words
                   services / fils / associes).
The per-word vocabulary cannot tell the roles apart and scores both as decoys. For pairs in the
noise role (the word is the query's only extra token, the S1 has at most one extra token that is
not a legal form, equal non-empty house numbers, top re-ranked candidate) the probability of the
model without the unseen-country vocabulary is kept instead: p := max(p_tx, p_global).
The same signature holds for the US dual-role word 'partners' in train (swap role 99.9% true,
added role 88.6% true among equal-house-number pairs).
"""
import polars as pl

from config import work_path

QOFF = 8_000_000
WORDS = ["groupe", "developpement", "france"]
LEGAL = ["sarl", "sas", "sasu", "eurl", "sa", "snc", "ei", "sci", "cie", "scop", "selarl", "scm", "eirl"]


def rule_pairs():
    seen = pl.read_parquet(work_path("train_s1.parquet"), columns=["country"])["country"].unique().implode()
    q = pl.concat([pl.read_parquet(work_path(f"test_s{k}.parquet"), columns=["idx", "country", "name_full", "addr_hn"])
                     .filter(~pl.col("country").is_in(seen))
                     .select((pl.col("idx") + off).cast(pl.UInt32).alias("qid"), "name_full", "addr_hn")
                   for k, off in ((2, 0), (3, QOFF))])
    top = (pl.scan_parquet(work_path("test_cand.parquet")).filter(pl.col("brank") == 0).select("qid", "s1")
             .with_columns(pl.col("qid").cast(pl.UInt32), pl.col("s1").cast(pl.UInt32))
             .join(q.lazy().select("qid"), on="qid", how="semi").collect())
    s1 = pl.read_parquet(work_path("test_s1.parquet"), columns=["idx", "name_full", "addr_hn"]).select(
        pl.col("idx").cast(pl.UInt32).alias("s1"), pl.col("name_full").alias("nf_s"), pl.col("addr_hn").alias("hn_s"))
    d = top.join(q, on="qid").join(s1, on="s1")
    qt, st = pl.col("name_full").str.split(" ").list.unique(), pl.col("nf_s").str.split(" ").list.unique()
    d = d.with_columns(qt.list.set_difference(st).alias("qx"), st.list.set_difference(qt).alias("sx"))
    return d.filter((pl.col("qx").list.len() == 1) & pl.col("qx").list.first().is_in(WORDS)
                    & (pl.col("sx").list.len() <= 1) & ~pl.col("sx").list.first().is_in(LEGAL).fill_null(False)
                    & (pl.col("addr_hn") != "") & (pl.col("addr_hn") == pl.col("hn_s"))).select("qid", "s1")


def apply(pred, p_global_file="test_pred2_sim.parquet"):
    """pred: DataFrame(qid, s1, p). Returns pred with the rule pairs' p raised to the global-model p."""
    g = (pl.read_parquet(work_path(p_global_file), columns=["qid", "s1", "p"])
           .with_columns(pl.col("qid").cast(pl.UInt32), pl.col("s1").cast(pl.UInt32))
           .join(rule_pairs(), on=["qid", "s1"], how="semi").rename({"p": "p_g"}))
    out = (pred.with_columns(pl.col("qid").cast(pl.UInt32), pl.col("s1").cast(pl.UInt32))
               .join(g, on=["qid", "s1"], how="full", coalesce=True)
               .with_columns(pl.max_horizontal(pl.col("p").fill_null(0), pl.col("p_g").fill_null(0)).cast(pl.Float32).alias("p"))
               .drop("p_g"))
    print(f"France role fix: {g.height:,} pairs")
    return out
