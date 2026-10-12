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
shared keys (bscore). The top TOPK_DEEP (60) by bscore are then re-ranked by a small
LightGBM model (train_ranker.py) on cheap string evidence (name / address similarity,
house-number relation), and the top TOPK (15) by ranker score are kept. Many true pairs
rank 16-60 by IDF alone (typo'd rare words, common tokens); re-ranking recovers them.

Output: work/{split}_cand.parquet with columns qid, s1, bscore, nkeys, bs0..bs4,
  irank (IDF rank), rscore (ranker score), q_nhits, brank (rank by rscore)
  qid = S2 idx, or S3 idx + QOFF.
Usage: python blocking.py train|test          (needs work/ranker.txt)
       python blocking.py train sample        (ranker training data, 1/40 of queries)
"""
import multiprocessing as mp
import os
import sys
import time

import lightgbm as lgb
import numpy as np
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler

from config import N_JOBS, work_path

QOFF = 8_000_000
CAP = 400          # max S1 document frequency for name / pair / house-number keys
CAP_ADDR = 100     # stricter cap for bare address tokens (city / street words)
TOPK = 15
TOPK_DEEP = 60
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
                    pl.col("addr_hn").alias("hn"))
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
    s1 = load(split, 1, ["idx", "country", "name_core", "addr_toks", "addr_nums", "addr_hn"])
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
        q = load(split, k, ["idx", "country", "name_core", "addr_toks", "addr_nums", "addr_hn"])
        off = 0 if k == 2 else QOFF
        q = q.with_columns((pl.col("idx") + off).alias("qid")).drop("idx")
        for i in range(0, q.height, Q_CHUNK):
            yield q.slice(i, Q_CHUNK)


# ----------------------------------------------------------------------------- re-ranking
RR_TEXT = ["name_core", "addr_toks", "addr_nums", "addr_hn", "domain"]
RR_STR = ["r_name_tset", "r_name_ratio", "r_name_jw", "r_addr_tset", "r_hn_eq", "r_hn_logdiff", "r_dom",
          "r_shn_in", "r_qhn_in"]
RR_FEATS = ["bscore", "nkeys", "bs0", "bs1", "bs2", "bs3", "bs4", "irank", "q_nhits", "gap_imax"] + RR_STR


def _rr_str(args):
    """Worker: cheap string evidence for re-ranking deep candidates."""
    qn, qa, qnum, qhn, qd, sn, sa, snum, shn = args
    out = np.empty((len(qn), len(RR_STR)), dtype=np.float32)
    for i in range(len(qn)):
        a, b = qn[i], sn[i]
        qh, sh = qhn[i], shn[i]
        if qh and sh:
            hn_eq = float(qh == sh)
            hn_ld = float(np.log1p(abs(int(qh[:9]) - int(sh[:9]))))
        else:
            hn_eq = hn_ld = -1.0
        s_in = float(sh in qnum[i].split()) if sh and qnum[i] else -1.0
        q_in = float(qh in snum[i].split()) if qh and snum[i] else -1.0
        out[i] = (fuzz.token_set_ratio(a, b), fuzz.ratio(a, b),
                  JaroWinkler.similarity(a.replace(" ", ""), b.replace(" ", "")) * 100,
                  fuzz.token_set_ratio(qa[i], sa[i]) if qa[i] and sa[i] else -1.0,
                  hn_eq, hn_ld,
                  fuzz.partial_ratio(qd[i], b.replace(" ", "")) if qd[i] else -1.0,
                  s_in, q_in)
    return out


def rerank_text(split):
    q = pl.concat([load(split, k, ["idx"] + RR_TEXT).with_columns((pl.col("idx") + off).alias("qid")).drop("idx")
                   for k, off in ((2, 0), (3, QOFF))])
    s1 = load(split, 1, ["idx"] + RR_TEXT).rename({"idx": "s1"})
    return q, s1


def rerank_features(top, qtext, stext, pool):
    part = (top.join(qtext, on="qid", how="left")
               .join(stext, on="s1", how="left", suffix="_s"))
    cols = [part[c].fill_null("").to_list() for c in ("name_core", "addr_toks", "addr_nums", "addr_hn", "domain",
                                                     "name_core_s", "addr_toks_s", "addr_nums_s", "addr_hn_s")]
    step = 20000
    jobs = [tuple(c[i:i + step] for c in cols) for i in range(0, part.height, step)]
    sf = pl.DataFrame(np.vstack(pool.map(_rr_str, jobs)), schema=RR_STR)
    keep = part.drop([c for c in part.columns if c in RR_TEXT or c[:-2] in RR_TEXT])
    return pl.concat([keep, sf], how="horizontal")


def deep_candidates(qc, index):
    """Top TOPK_DEEP S1 per query by IDF score, with per-type scores and IDF rank."""
    qk = make_keys(qc, "qid")
    hits = qk.join(index, on="key")  # typ comes from the query side (same key => same type)
    agg = (hits.group_by("qid", "s1")
               .agg(pl.col("w").sum().alias("bscore"), pl.len().cast(pl.UInt16).alias("nkeys"),
                    *[pl.col("w").filter(pl.col("typ") == t).sum().alias(f"bs{t}") for t in range(5)]))
    agg = agg.with_columns(pl.len().over("qid").cast(pl.UInt32).alias("q_nhits"))
    top = (agg.sort(["qid", "bscore"], descending=[False, True])
              .group_by("qid", maintain_order=True).head(TOPK_DEEP)
              .with_columns(pl.int_range(pl.len()).over("qid").cast(pl.UInt8).alias("irank"),
                            (pl.col("bscore") - pl.col("bscore").max().over("qid")).alias("gap_imax")))
    return top, hits.height


def main(split, mode="full"):
    t0 = time.time()
    pool = mp.get_context("spawn").Pool(N_JOBS)
    index = build_index(split)
    qtext, stext = rerank_text(split)
    if mode == "sample":
        from evaluate import load_truth
        truth = load_truth().with_columns(pl.col("qid").cast(pl.UInt32), pl.col("s1_true").cast(pl.UInt32))
        parts = []
        for qc in query_chunks(split):
            qc = qc.filter((pl.col("qid").hash(seed=13) % 40) == 0)
            top, _ = deep_candidates(qc, index)
            f = rerank_features(top, qtext, stext, pool)
            parts.append(f.join(truth, on="qid", how="left")
                          .with_columns((pl.col("s1") == pl.col("s1_true")).fill_null(False).cast(pl.Int8).alias("label"))
                          .drop("s1_true"))
        out = pl.concat(parts)
        out.write_parquet(work_path(f"{split}_deep_sample.parquet"))
        print(f"deep sample: {out.height:,} pairs, {out['qid'].n_unique():,} queries [{time.time() - t0:.0f}s]")
        return
    ranker = lgb.Booster(model_file=work_path("ranker.txt"))
    outdir = work_path(f"{split}_cand")
    os.makedirs(outdir, exist_ok=True)
    for ci, qc in enumerate(query_chunks(split)):
        path = os.path.join(outdir, f"part_{ci:04d}.parquet")
        if os.path.exists(path):  # resumable
            continue
        top, nh = deep_candidates(qc, index)
        f = rerank_features(top, qtext, stext, pool)
        f = f.with_columns(pl.Series("rscore", ranker.predict(f.select(RR_FEATS).to_numpy().astype(np.float32))
                                     .astype(np.float32)))
        keep = (f.sort(["qid", "rscore"], descending=[False, True])
                 .group_by("qid", maintain_order=True).head(TOPK)
                 .with_columns(pl.int_range(pl.len()).over("qid").cast(pl.UInt8).alias("brank"))
                 .select("qid", "s1", "bscore", "nkeys", "bs0", "bs1", "bs2", "bs3", "bs4",
                         "brank", "irank", "rscore", "q_nhits"))
        keep.write_parquet(path)
        print(f"  chunk qid {qc['qid'][0]}: hits {nh:,} -> {keep.height:,}  [{time.time() - t0:.0f}s]", flush=True)
        del top, f, keep
    pool.close()
    pl.scan_parquet(os.path.join(outdir, "part_*.parquet")).sink_parquet(work_path(f"{split}_cand.parquet"))
    n = pl.scan_parquet(work_path(f"{split}_cand.parquet")).select(pl.len()).collect().item()
    print(f"candidates: {n:,} [{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "full")
