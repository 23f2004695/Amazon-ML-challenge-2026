"""Test-time prior-shift correction for house-number "families" (label-free).

A record whose best S1 has a near-identical name and street but a nearby, different house
number is either a true match with house-number noise or a decoy (sister business). Noise
for true entities is generated per source (an entity's S3 records share one drifted address)
and occurs at the same rate per S1 in train and test; decoy businesses, however, are far more
frequent in parts of the test set (US: 3-5x more such records per S1, often with several
records each, which the stage-2 family features read as a true entity's consistent drift).

For every (country, type) cell, type in {single, cross-source family, within-S2 family,
within-S3 family}, we assume the number of TRUE such records per S1 is the same as in the
labelled simulation, so the test share of true records is
    pi_test = true_per_S1(simulation) / records_per_S1(test)
and each probability is corrected by the odds ratio  odds(pi_test) / odds(pi_sim)  (only
ever lowered). Countries without training labels use the pooled simulation rates.
"""
import polars as pl

from config import work_path

QOFF = 8_000_000
NEAR = 30


def family_types(pred, split):
    """Best pair per query with nearby-differing house number, labelled by family type."""
    best = (pred.group_by("qid").agg(pl.col("s1").sort_by("p").last(), pl.col("p").max()))
    q = pl.concat([pl.read_parquet(work_path(f"{split}_s{k}.parquet"), columns=["idx", "country", "addr_hn"])
                     .select((pl.col("idx") + off).cast(pl.UInt32).alias("qid"), "country", "addr_hn")
                   for k, off in ((2, 0), (3, QOFF))])
    s1 = pl.read_parquet(work_path(f"{split}_s1.parquet"), columns=["idx", "addr_hn"]).select(
        pl.col("idx").cast(pl.UInt32).alias("s1"), pl.col("addr_hn").alias("h1"))
    d = (best.with_columns(pl.col("qid").cast(pl.UInt32), pl.col("s1").cast(pl.UInt32))
             .join(q, on="qid").join(s1, on="s1")
             .with_columns((pl.col("qid") >= QOFF).alias("s3"),
                           (pl.col("addr_hn").cast(pl.Int64, strict=False)
                            - pl.col("h1").cast(pl.Int64, strict=False)).abs().alias("diff")))
    d = d.filter((pl.col("addr_hn") != "") & (pl.col("h1") != "") & (pl.col("addr_hn") != pl.col("h1"))
                 & (pl.col("diff") <= NEAR))
    d = d.with_columns(pl.len().over("s1", "addr_hn").alias("famsize"),
                       pl.col("s3").n_unique().over("s1", "addr_hn").alias("n_src"))
    typ = (pl.when(pl.col("famsize") == 1).then(pl.lit("single"))
             .when(pl.col("n_src") == 2).then(pl.lit("cross"))
             .when(pl.col("s3")).then(pl.lit("within_s3")).otherwise(pl.lit("within_s2")))
    return d.with_columns(typ.alias("ftype")).select("qid", "s1", "p", "country", "ftype")


def reference_rates(oof_file, truth, ids):
    """Per (country, type): true records per S1 and true share in the labelled simulation."""
    pred = pl.read_parquet(work_path(oof_file), columns=["qid", "s1", "p"])
    d = family_types(pred, "train").join(truth, on="qid", how="left").with_columns(
        (pl.col("s1_true") == pl.col("s1")).fill_null(False).alias("true"))
    n_s1 = (pl.read_parquet(work_path("train_s1.parquet"), columns=["idx", "country"])
              .filter(pl.col("idx").cast(pl.UInt32).is_in(ids.implode())).group_by("country").len()
              .rename({"len": "n_s1"}))
    r = (d.group_by("country", "ftype").agg(pl.len().alias("n"), pl.col("true").sum().alias("n_true"))
          .join(n_s1, on="country"))
    pooled = (r.group_by("ftype").agg(pl.col("n").sum(), pl.col("n_true").sum(), pl.col("n_s1").sum())
                .with_columns(pl.lit("*").alias("country")))
    r = pl.concat([r, pooled.select(r.columns)])
    return r.with_columns((pl.col("n_true") / pl.col("n_s1")).alias("true_per_s1"),
                          (pl.col("n_true") / pl.col("n")).alias("pi_sim"))


def correction_table(test_pred, ref):
    """Per (country, type) odds multiplier k <= 1 for the test set."""
    d = family_types(test_pred, "test")
    n_s1 = pl.read_parquet(work_path("test_s1.parquet"), columns=["country"]).group_by("country").len().rename({"len": "n_s1"})
    t = d.group_by("country", "ftype").agg(pl.len().alias("n_test")).join(n_s1, on="country")
    known = ref.filter(pl.col("country") != "*").select("country").unique()
    t = t.with_columns(pl.when(pl.col("country").is_in(known["country"].implode())).then(pl.col("country"))
                         .otherwise(pl.lit("*")).alias("ref_country"))
    t = t.join(ref.select(pl.col("country").alias("ref_country"), "ftype", "true_per_s1", "pi_sim"),
               on=["ref_country", "ftype"], how="left")
    t = t.with_columns((pl.col("true_per_s1") / (pl.col("n_test") / pl.col("n_s1"))).clip(0.001, 0.999).alias("pi_test"))
    odds = lambda c: pl.col(c) / (1 - pl.col(c))
    t = t.with_columns((odds("pi_test") / odds("pi_sim").clip(1e-6, None)).clip(upper_bound=1.0).alias("k"))
    return d, t


def apply(test_pred, d, t):
    """Lower p of the affected best pairs by their cell's odds multiplier."""
    adj = d.join(t.select("country", "ftype", "k"), on=["country", "ftype"]).select("qid", "s1", "k")
    out = (test_pred.with_columns(pl.col("qid").cast(pl.UInt32), pl.col("s1").cast(pl.UInt32))
                    .join(adj, on=["qid", "s1"], how="left").with_columns(pl.col("k").fill_null(1.0)))
    p = pl.col("p")
    return out.with_columns((p * pl.col("k") / (p * pl.col("k") + 1 - p)).cast(pl.Float32).alias("p")).drop("k")
