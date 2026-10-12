"""Stage 2: candidate generation (blocking) with an IDF-weighted inverted index.

Every record emits hashed keys, all prefixed with its country so records only meet
records from the same country label:
  n|tok        core name token (and the concatenation of all core tokens, which
               catches website-style names such as 'empirecrystalrepublic.com')
  b|t1_t2      unordered pair of core name tokens (robust to word re-ordering)
  h|num|tok    house number x address token
  a|tok        address token
  s|a1_a2      adjacent address-token pair (street / locality name)
  x|tok|atok   name token x address token (first MAX_NA_TOKS core tokens)
Keys whose Source-1 document frequency exceeds CAP are dropped (too unspecific).
Each query (S2/S3 record) is scored against Source-1 records by the sum of IDF of
shared keys, and the top-K Source-1 records per query are kept.

Output: work/{split}_cand.parquet with columns qid, s1, bscore, nkeys, brank
  qid = S2 idx, or S3 idx + QOFF.
Usage: python blocking.py train|test
"""
import os
import sys
import time

import polars as pl

from config import work_path

QOFF = 8_000_000
CAP = 400          # max S1 document frequency for name / pair / house-number keys
CAP_ADDR = 100     # stricter cap for bare address tokens (city / street words)
TOPK = 15
MAX_PAIR_TOKS = 6
MAX_NA_TOKS = 3
Q_CHUNK = int(os.environ.get("BER_QCHUNK", "100000"))


def load(split, k, cols):
    return pl.read_parquet(work_path(f"{split}_s{k}.parquet"), columns=cols)


def make_keys(df, id_col):
    """Return DataFrame(id, key:u64) with all blocking keys of each record."""
    c = pl.col("country")
    toks = (df.select(id_col, "country", pl.col("name_core").str.split(" ").alias("t"))
              .with_columns(pl.col("t").list.head(MAX_PAIR_TOKS))
              .explode("t").filter(pl.col("t").str.len_chars() >= 2)
              )
    k_name = toks.select(id_col, (c + "|n|" + pl.col("t")).alias("key"), pl.lit(0, pl.UInt8).alias("typ"))
    k_concat = (df.filter(pl.col("name_core").str.contains(" "))
                  .select(id_col, (c + "|n|" + pl.col("name_core").str.replace_all(" ", "")).alias("key"),
                          pl.lit(0, pl.UInt8).alias("typ")))
    pairs = (toks.join(toks.select(id_col, pl.col("t").alias("t2")), on=id_col)
                 .filter(pl.col("t") < pl.col("t2")))
    k_pair = pairs.select(id_col, (c + "|b|" + pl.col("t") + "_" + pl.col("t2")).alias("key"),
                          pl.lit(1, pl.UInt8).alias("typ"))
    at = (df.select(id_col, "country", pl.col("addr_toks").str.split(" ").alias("a"),
                    pl.col("addr_nums").str.split(" ").list.first().alias("hn"))
            .explode("a").filter(pl.col("a").str.len_chars() >= 3))
    k_addr = at.select(id_col, (c + "|a|" + pl.col("a")).alias("key"), pl.lit(3, pl.UInt8).alias("typ"))
    k_hn = (at.filter(pl.col("hn").is_not_null() & (pl.col("hn") != ""))
              .select(id_col, (c + "|h|" + pl.col("hn") + "|" + pl.col("a")).alias("key"),
                      pl.lit(2, pl.UInt8).alias("typ")))
    # adjacent address-token pairs: street / locality names ('buck ridge', 'water hill')
    k_street = (df.select(id_col, "country", pl.col("addr_toks").str.split(" ").alias("a"))
                  .with_columns(pl.col("a").list.slice(1).alias("a2"), pl.col("a").list.head(pl.col("a").list.len() - 1))
                  .explode("a", "a2").filter(pl.col("a2").is_not_null() & (pl.col("a").str.len_chars() >= 3)
                                             & (pl.col("a2").str.len_chars() >= 3))
                  .select(id_col, (c + "|s|" + pl.col("a") + "_" + pl.col("a2")).alias("key"),
                          pl.lit(2, pl.UInt8).alias("typ")))
    # name token x address token: generic names become specific inside a locality
    k_na = (toks.group_by(id_col, maintain_order=True).head(MAX_NA_TOKS)
                .join(at.select(id_col, "a"), on=id_col)
                .select(id_col, (c + "|x|" + pl.col("t") + "|" + pl.col("a")).alias("key"),
                        pl.lit(4, pl.UInt8).alias("typ")))
    keys = pl.concat([k_name, k_concat, k_pair, k_addr, k_hn, k_street, k_na])
    return keys.select(id_col, pl.col("key").hash(seed=7).alias("key"), "typ").unique([id_col, "key"])


def build_index(split):
    s1 = load(split, 1, ["idx", "country", "name_core", "addr_toks", "addr_nums"])
    s1 = s1.with_columns(pl.col("country").cast(pl.Categorical).to_physical().cast(pl.UInt8).alias("cc"))
    n_country = s1.group_by("cc").len().rename({"len": "N"})
    parts = []
    for i in range(0, s1.height, 200_000):  # hash keys chunk by chunk to bound memory
        ch = s1.slice(i, 200_000).rename({"idx": "s1"})
        parts.append(make_keys(ch, "s1").join(ch.select("s1", "cc"), on="s1"))
    keys = pl.concat(parts)
    del parts
    df = keys.group_by("key").agg(pl.len().alias("df"), pl.col("cc").first(), pl.col("typ").first())
    cap = pl.when(pl.col("typ") == 3).then(CAP_ADDR).otherwise(CAP)
    df = (df.filter(pl.col("df") <= cap).join(n_country, on="cc")
            .with_columns(((pl.col("N") / pl.col("df")).log()).cast(pl.Float32).alias("w"))
            .select("key", "w"))
    index = keys.select("s1", "key").join(df, on="key")
    print(f"index: {index.height:,} postings, {df.height:,} keys", flush=True)
    return index


def query_chunks(split):
    for k in (2, 3):
        q = load(split, k, ["idx", "country", "name_core", "addr_toks", "addr_nums"])
        off = 0 if k == 2 else QOFF
        q = q.with_columns((pl.col("idx") + off).alias("qid")).drop("idx")
        for i in range(0, q.height, Q_CHUNK):
            yield q.slice(i, Q_CHUNK)


def main(split):
    t0 = time.time()
    index = build_index(split)
    outdir = work_path(f"{split}_cand")
    os.makedirs(outdir, exist_ok=True)
    for ci, qc in enumerate(query_chunks(split)):
        path = os.path.join(outdir, f"part_{ci:04d}.parquet")
        if os.path.exists(path):  # resumable
            continue
        qk = make_keys(qc, "qid")
        hits = qk.join(index, on="key")  # typ comes from the query side (same key => same type)
        agg = (hits.group_by("qid", "s1")
                   .agg(pl.col("w").sum().alias("bscore"), pl.len().cast(pl.UInt16).alias("nkeys"),
                        *[pl.col("w").filter(pl.col("typ") == t).sum().alias(f"bs{t}") for t in range(5)]))
        top = (agg.sort(["qid", "bscore"], descending=[False, True])
                  .group_by("qid", maintain_order=True).head(TOPK)
                  .with_columns(pl.int_range(pl.len()).over("qid").cast(pl.UInt8).alias("brank")))
        top.write_parquet(path)
        print(f"  chunk qid {qc['qid'][0]}: hits {hits.height:,} -> {top.height:,}  "
              f"[{time.time() - t0:.0f}s]", flush=True)
        del qk, hits, agg, top
    pl.scan_parquet(os.path.join(outdir, "part_*.parquet")).sink_parquet(work_path(f"{split}_cand.parquet"))
    n = pl.scan_parquet(work_path(f"{split}_cand.parquet")).select(pl.len()).collect().item()
    print(f"candidates: {n:,} [{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main(sys.argv[1])
