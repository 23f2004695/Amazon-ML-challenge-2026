"""Diagnostic: recall@K of the blocking candidates against train ground truth."""
import polars as pl

from config import work_path
from evaluate import load_truth

if __name__ == "__main__":
    truth = load_truth().with_columns(pl.col("qid").cast(pl.UInt32), pl.col("s1_true").cast(pl.UInt32))
    cand = pl.scan_parquet(work_path("train_cand.parquet")).select("qid", "s1", "brank")
    hit = (truth.lazy().join(cand, left_on=["qid", "s1_true"], right_on=["qid", "s1"], how="left")
                .select("brank").collect(engine="streaming"))
    for k in (1, 2, 3, 5, 8, 10, 15):
        print(f"recall@{k}: {(hit['brank'] < k).sum() / truth.height:.4f}")
