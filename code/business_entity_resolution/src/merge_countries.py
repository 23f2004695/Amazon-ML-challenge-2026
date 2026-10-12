"""Compose a submission from two finished ones by S1 country: rows of the listed countries come
from B, all other rows from A (matching_results and candidate_pairs alike, so every match stays
inside its own candidate list). Links never cross countries, so the rows are independent.

With BER_GUARD_NEAR=US: B's links of those countries that A does not have and whose house numbers
differ by 1-30 (the decoy-family signature; the US test holds 1.7-3.6x more such decoys than the
simulation) are dropped.

Usage: [BER_GUARD_NEAR=US] python merge_countries.py <A name> <B name> <out name> US India
       (reads/writes output/matching_results_<name>.tsv and output/candidate_pairs_<name>.tsv)
"""
import os
import sys

import polars as pl

from config import DATA, OUTPUT, work_path

QOFF = 8_000_000


def _explode(d):
    col = d.columns[1]
    return (d.with_columns(pl.col(col).str.split(",")).explode(col).drop_nulls(col).filter(pl.col(col) != "")
             .rename({col: "e"}))


def guard_near(m, fa, c, countries):
    """Drop links of `countries` absent from A whose house numbers differ by 1-30."""
    col = m.columns[1]
    q = pl.concat([pl.read_parquet(work_path(f"test_s{k}.parquet"), columns=["entity_id", "addr_hn"])
                     .select(pl.col("entity_id").alias("e"), pl.col("addr_hn").cast(pl.Int64, strict=False).alias("h"))
                   for k in (2, 3)])
    s1 = pl.read_parquet(work_path("test_s1.parquet"), columns=["entity_id", "addr_hn"]).select(
        pl.col("entity_id").alias("source1_entity_id"), pl.col("addr_hn").cast(pl.Int64, strict=False).alias("h1"))
    new = (_explode(m).join(_explode(fa), on=["source1_entity_id", "e"], how="anti")
             .join(c.filter(pl.col("country").is_in(countries)).select("source1_entity_id"), on="source1_entity_id", how="semi")
             .join(q, on="e").join(s1, on="source1_entity_id")
             .filter((pl.col("h") - pl.col("h1")).abs().is_between(1, 30)).select("source1_entity_id", "e"))
    print(f"guard: dropping {new.height:,} new near-twin links of {countries}")
    kept = (_explode(m).join(new, on=["source1_entity_id", "e"], how="anti")
              .group_by("source1_entity_id").agg(pl.col("e").sort().str.join(",").alias(col)))
    return (m.with_row_index("row").drop(col).join(kept, on="source1_entity_id", how="left")
              .with_columns(pl.col(col).fill_null("")).sort("row").select("source1_entity_id", col))


def merge(a, b, out, countries):
    c = pl.read_csv(os.path.join(DATA, "test", "test_source1.tsv"), separator="\t", infer_schema=False, quote_char=None,
                    columns=["entity_id", "country"]).rename({"entity_id": "source1_entity_id"})
    take_b = c.filter(pl.col("country").is_in(countries)).select("source1_entity_id")
    for kind in ("matching_results", "candidate_pairs"):
        fa, fb = (pl.read_csv(os.path.join(OUTPUT, f"{kind}_{n}.tsv"), separator="\t", infer_schema=False, quote_char=None)
                  for n in (a, b))
        col = fa.columns[1]
        ob = fb.join(take_b, on="source1_entity_id", how="semi").rename({col: "_b"})
        m = (fa.with_row_index("row").join(ob, on="source1_entity_id", how="left")
               .with_columns(pl.when(pl.col("source1_entity_id").is_in(take_b["source1_entity_id"].implode()))
                               .then(pl.col("_b")).otherwise(pl.col(col)).fill_null("").alias(col))
               .sort("row").select("source1_entity_id", col))
        if kind == "matching_results" and os.environ.get("BER_GUARD_NEAR"):
            m = guard_near(m, fa, c, os.environ["BER_GUARD_NEAR"].split(","))
        path = os.path.join(OUTPUT, f"{kind}_{out}.tsv")
        m.write_csv(path, separator="\t", quote_style="never")
        print(f"wrote {path}: {m.height:,} rows ({ob.height:,} from {b})")


if __name__ == "__main__":
    merge(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:])
