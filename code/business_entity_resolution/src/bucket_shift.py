"""Label-free prior-shift correction by pair bucket (generalizes prior_shift.py).

Every record's best pair is put in a bucket: house-number relation (equal / nearby / far /
unknown) x name similarity (>=90 / 50-90 / <50) x address similarity (>=80 / <80 / no
address), with nearby near-twins further split by family type (single / cross-source /
within-S2 / within-S3). The noise that produces TRUE records is the same in train and test,
so the number of true pairs per S1 in each bucket is assumed invariant; what changes is the
volume of decoys (more in the US test) and of orphans (fewer in the test than in the
withheld-S1 simulation). Per (country, bucket):
    pi_target = true_per_S1(reference) / records_per_S1(target)
and every probability in the bucket is moved by the odds ratio odds(pi_target)/odds(pi_ref).
Countries without training labels use pooled reference rates and are only corrected downward.

Usage (validation on labelled data): BER_SIM=1 python bucket_shift.py
"""
import glob

import polars as pl

from config import TRAIN_FEAT, work_path
from evaluate import eval_population, f05_macro, load_truth
from train import assign, cap_links

QOFF = 8_000_000
K_MIN, K_MAX = 0.02, 5.0


def bucketize(pred, split, featdir, keep_qids=None):
    """Best pair per query with its bucket label."""
    best = pred.group_by("qid").agg(pl.col("s1").sort_by("p").last(), pl.col("p").max()).with_columns(
        pl.col("qid").cast(pl.UInt32), pl.col("s1").cast(pl.UInt32))
    if keep_qids is not None:
        best = best.join(keep_qids, on="qid", how="semi")
    cols = ["qid", "s1", "c_tset", "a_tset", "hn_eq", "hn_logdiff", "qa_empty"]
    f = pl.concat([pl.read_parquet(p, columns=cols).join(best.select("qid", "s1"), on=["qid", "s1"], how="semi")
                   for p in sorted(glob.glob(work_path(f"{featdir}/part_*.parquet")))])
    q = pl.concat([pl.read_parquet(work_path(f"{split}_s{k}.parquet"), columns=["idx", "country", "addr_hn"])
                     .select((pl.col("idx") + off).cast(pl.UInt32).alias("qid"), "country", "addr_hn")
                   for k, off in ((2, 0), (3, QOFF))])
    d = best.join(f, on=["qid", "s1"], how="left").join(q, on="qid")
    hn = (pl.when(pl.col("hn_eq") == 1).then(pl.lit("hn="))
            .when(pl.col("hn_eq") == 0).then(pl.when(pl.col("hn_logdiff") <= 3.43).then(pl.lit("near")).otherwise(pl.lit("far")))
            .otherwise(pl.lit("hn?")))
    nm = pl.when(pl.col("c_tset") >= 90).then(pl.lit("N90")).when(pl.col("c_tset") >= 50).then(pl.lit("N50")).otherwise(pl.lit("N0"))
    ad = pl.when(pl.col("qa_empty") == 1).then(pl.lit("A-")).when(pl.col("a_tset") >= 80).then(pl.lit("A80")).otherwise(pl.lit("A0"))
    d = d.with_columns(pl.concat_str([hn, nm, ad], separator="|").fill_null("expansion").alias("b"),
                       (pl.col("qid") >= QOFF).alias("s3"))
    # family split of nearby near-twins (records of one S1 sharing the same differing number)
    fam = d.filter(pl.col("b") == "near|N90|A80").with_columns(
        pl.len().over("s1", "addr_hn").alias("fs"), pl.col("s3").n_unique().over("s1", "addr_hn").alias("ns"))
    fam = fam.with_columns(pl.when(pl.col("fs") == 1).then(pl.lit("|single")).when(pl.col("ns") == 2).then(pl.lit("|cross"))
                             .when(pl.col("s3")).then(pl.lit("|in_s3")).otherwise(pl.lit("|in_s2")).alias("sub")).select("qid", "sub")
    d = d.join(fam, on="qid", how="left").with_columns(pl.concat_str([pl.col("b"), pl.col("sub").fill_null("")]).alias("b"))
    return d.select("qid", "s1", "p", "country", "b")


def reference(ref_best, truth, n_s1):
    r = (ref_best.join(truth, on="qid", how="left")
                 .with_columns((pl.col("s1_true") == pl.col("s1")).fill_null(False).alias("true"))
                 .group_by("country", "b").agg(pl.len().alias("n"), pl.col("true").sum().alias("n_true"))
                 .join(n_s1, on="country"))
    pooled = (r.group_by("b").agg(pl.col("n").sum(), pl.col("n_true").sum(), pl.col("n_s1").sum())
                .with_columns(pl.lit("*").alias("country")))
    r = pl.concat([r, pooled.select(r.columns)])
    return r.with_columns((pl.col("n_true") / pl.col("n_s1")).alias("true_per_s1"),
                          (pl.col("n_true") / pl.col("n")).alias("pi_ref")).select("country", "b", "true_per_s1", "pi_ref", "n")


def corrections(tgt_best, n_s1_tgt, ref, min_n=500):
    t = tgt_best.group_by("country", "b").agg(pl.len().alias("n_tgt")).join(n_s1_tgt, on="country")
    known = ref.filter(pl.col("country") != "*")["country"].unique()
    t = t.with_columns(pl.col("country").is_in(known.implode()).alias("seen"),
                       pl.when(pl.col("country").is_in(known.implode())).then(pl.col("country")).otherwise(pl.lit("*")).alias("rc"))
    t = t.join(ref.rename({"country": "rc"}), on=["rc", "b"], how="left")
    odds = lambda c: pl.col(c) / (1 - pl.col(c))
    t = t.with_columns((pl.col("true_per_s1") / (pl.col("n_tgt") / pl.col("n_s1"))).clip(0.001, 0.999).alias("pi_tgt"),
                       pl.col("pi_ref").clip(0.001, 0.999))
    t = t.with_columns((odds("pi_tgt") / odds("pi_ref")).clip(K_MIN, K_MAX).alias("k"))
    t = t.with_columns(pl.when((pl.col("n") < min_n) | pl.col("k").is_null()).then(1.0)
                         .when(~pl.col("seen")).then(pl.min_horizontal(pl.col("k"), pl.lit(1.0)))
                         .otherwise(pl.col("k")).alias("k"))
    return t.select("country", "b", "k", "pi_ref", "pi_tgt", "n_tgt")


def apply(pred, tgt_best, t):
    adj = tgt_best.join(t.select("country", "b", "k"), on=["country", "b"], how="left").select("qid", "s1", "k")
    out = (pred.with_columns(pl.col("qid").cast(pl.UInt32), pl.col("s1").cast(pl.UInt32))
               .join(adj, on=["qid", "s1"], how="left").with_columns(pl.col("k").fill_null(1.0)))
    p = pl.col("p")
    return out.with_columns((p * pl.col("k") / (p * pl.col("k") + 1 - p)).cast(pl.Float32).alias("p")).drop("k")


if __name__ == "__main__":
    # Validation: reference = simulation (with its withheld-S1 orphans); target = the same
    # labelled data WITHOUT orphans (fewer orphans, like the test). Does the correction help?
    truth_sim, ids = eval_population()
    drop = pl.read_parquet(work_path("train_drop_ids.parquet"))["s1"].cast(pl.UInt32)
    full = load_truth().with_columns(pl.col("qid").cast(pl.UInt32), pl.col("s1_true").cast(pl.UInt32))
    orphans = full.filter(pl.col("s1_true").is_in(drop.implode())).select("qid")
    oof = pl.read_parquet(work_path("train_oof2_sim.parquet"), columns=["qid", "s1", "p"])
    n_s1 = (pl.read_parquet(work_path("train_s1.parquet"), columns=["idx", "country"])
              .filter(pl.col("idx").cast(pl.UInt32).is_in(ids.implode())).group_by("country").len().rename({"len": "n_s1"}))
    ref_best = bucketize(oof, "train", TRAIN_FEAT)
    ref = reference(ref_best, truth_sim, n_s1)
    allq = pl.concat([pl.read_parquet(work_path(f"train_s{k}.parquet"), columns=["idx"]).select((pl.col("idx") + off).cast(pl.UInt32).alias("qid"))
                      for k, off in ((2, 0), (3, QOFF))])
    keep = allq.join(orphans, on="qid", how="anti")
    tgt_pred = oof.with_columns(pl.col("qid").cast(pl.UInt32)).join(keep, on="qid", how="semi")
    tgt_best = bucketize(tgt_pred, "train", TRAIN_FEAT)
    t = corrections(tgt_best, n_s1, ref)
    print(t.filter(pl.col("n_tgt") > 20000).sort("country", "n_tgt", descending=[False, True]).head(24))
    truth_tgt = truth_sim.join(orphans, on="qid", how="anti")
    for thr in (0.62, 0.66, 0.70):
        base = f05_macro(cap_links(assign(tgt_pred, thr)), truth_tgt, ids)[0]
        corr = f05_macro(cap_links(assign(apply(tgt_pred, tgt_best, t), thr)), truth_tgt, ids)[0]
        print(f"orphan-free target, thr {thr:.2f}: uncorrected {base:.5f}  bucket-corrected {corr:.5f}", flush=True)
