"""Ground-truth loading and the official macro F0.5 metric (singletons included)."""
import polars as pl

from config import GT_PATH, SIM, work_path

QOFF = 8_000_000


def load_truth():
    """Return DataFrame(qid, s1_true) for every matched train S2/S3 record (idx space)."""
    gt = pl.read_csv(GT_PATH, separator="\t", infer_schema=False, quote_char=None)
    pairs = (gt.with_columns(pl.col("matched_entity_ids").str.split(","))
               .explode("matched_entity_ids").drop_nulls()
               .rename({"source1_entity_id": "e1", "matched_entity_ids": "entity_id"}))
    s1 = pl.read_parquet(work_path("train_s1.parquet"), columns=["idx", "entity_id"]).rename(
        {"idx": "s1_true", "entity_id": "e1"})
    out = []
    for k, off in ((2, 0), (3, QOFF)):
        sk = pl.read_parquet(work_path(f"train_s{k}.parquet"), columns=["idx", "entity_id"])
        j = pairs.join(sk, on="entity_id").join(s1, on="e1")
        out.append(j.select((pl.col("idx") + off).alias("qid"), "s1_true"))
    return pl.concat(out)


def f05_macro(pred, truth, s1_ids):
    """Macro F0.5 over the S1 records in s1_ids.

    pred:  DataFrame(qid, s1) of predicted assignments.
    truth: DataFrame(qid, s1_true).
    s1_ids: Series/list of S1 idx forming the evaluation population.
    """
    pop = pl.DataFrame({"s1": pl.Series(s1_ids, dtype=pl.UInt32)})
    pred = pred.select(pl.col("qid").cast(pl.UInt32), pl.col("s1").cast(pl.UInt32))
    truth = truth.select(pl.col("qid").cast(pl.UInt32), pl.col("s1_true").cast(pl.UInt32))
    npred = pred.group_by("s1").len().rename({"len": "np"})
    ntrue = truth.group_by("s1_true").len().rename({"s1_true": "s1", "len": "nt"})
    tp = (pred.join(truth, on="qid").filter(pl.col("s1") == pl.col("s1_true"))
              .group_by("s1").len().rename({"len": "tp"}))
    d = (pop.join(npred, on="s1", how="left").join(ntrue, on="s1", how="left")
            .join(tp, on="s1", how="left").fill_null(0))
    p = pl.col("tp") / pl.col("np")
    r = pl.col("tp") / pl.col("nt")
    f = (pl.when((pl.col("np") == 0) & (pl.col("nt") == 0)).then(1.0)
           .when((pl.col("np") == 0) | (pl.col("nt") == 0) | (pl.col("tp") == 0)).then(0.0)
           .otherwise(1.25 * p * r / (0.25 * p + r)))
    d = d.with_columns(f.alias("f"))
    return d["f"].mean(), d


def eval_population():
    """(truth, s1_ids) for OOF scoring; under BER_SIM the withheld S1 records are removed,
    so their S2/S3 records count as unmatched."""
    truth = load_truth().with_columns(pl.col("qid").cast(pl.UInt32), pl.col("s1_true").cast(pl.UInt32))
    ids = pl.read_parquet(work_path("train_s1.parquet"), columns=["idx"])["idx"].cast(pl.UInt32)
    if SIM:
        drop = pl.read_parquet(work_path("train_drop_ids.parquet"))["s1"].cast(pl.UInt32)
        truth = truth.filter(~pl.col("s1_true").is_in(drop.implode()))
        ids = ids.filter(~ids.is_in(drop))
    return truth, ids
