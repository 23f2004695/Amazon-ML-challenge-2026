"""Stage 1: read raw TSVs, normalize names/addresses in parallel, write parquet.

Output per (split, source): work/{split}_s{k}.parquet with columns
  idx, entity_id, country, name_raw, addr_raw, name_full, name_core, domain,
  addr_toks, addr_nums, state, addr_hn (house number, '' if none)
Token lists are stored as space-joined strings.

Usage: python prepare.py train|test
"""
import sys
from multiprocessing import Pool

import polars as pl

from config import N_JOBS, src_path, work_path
from normalize import load_indic_dict, norm_address, norm_name


def _norm_chunk(args):
    names, addrs = args
    load_indic_dict()
    out = []
    for n, a in zip(names, addrs):
        full, core, dom = norm_name(n)
        toks, nums, st, hn = norm_address(a)
        out.append((" ".join(full), " ".join(core), dom, " ".join(toks), " ".join(nums), st, hn))
    return out


def read_raw(split, k):
    df = pl.read_csv(src_path(split, k), separator="\t", infer_schema=False, quote_char=None)
    return df.with_columns(pl.col("business_address").fill_null(""),
                           pl.col("business_name").fill_null(""),
                           pl.col("country").fill_null(""))


def prepare(split, k):
    df = read_raw(split, k)
    names, addrs = df["business_name"].to_list(), df["business_address"].to_list()
    step = 20000
    chunks = [(names[i:i + step], addrs[i:i + step]) for i in range(0, len(names), step)]
    with Pool(N_JOBS) as pool:
        res = [r for part in pool.imap(_norm_chunk, chunks, chunksize=1) for r in part]
    cols = list(zip(*res))
    out = pl.DataFrame({
        "idx": pl.Series(range(df.height), dtype=pl.UInt32),
        "entity_id": df["entity_id"],
        "country": df["country"],
        "name_raw": df["business_name"],
        "addr_raw": df["business_address"],
        "name_full": cols[0], "name_core": cols[1], "domain": cols[2],
        "addr_toks": cols[3], "addr_nums": cols[4], "state": cols[5], "addr_hn": cols[6],
    })
    out.write_parquet(work_path(f"{split}_s{k}.parquet"))
    print(split, k, out.height, flush=True)


if __name__ == "__main__":
    for k in (1, 2, 3):
        prepare(sys.argv[1], k)
