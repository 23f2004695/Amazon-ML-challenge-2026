"""Learn an Indic-script token -> Latin token dictionary from training matches.

For every matched (S1, S2/S3) pair where the S2/S3 name contains Indic script and has
the same number of tokens as the S1 name, tokens are aligned position-by-position and
each alignment is a vote. A token maps to its majority Latin token. Only training data
is used (no external resources).
"""
import collections
import json
import re

import polars as pl

from config import GT_PATH, src_path, work_path
from normalize import RE_INDIC, latinize

RE_PUNCT = re.compile(r"[^\w\sऀ-෿‌‍]")


def read(path):
    return pl.read_csv(path, separator="\t", infer_schema=False, quote_char=None)


def main():
    gt = read(GT_PATH)
    pairs = (gt.with_columns(pl.col("matched_entity_ids").str.split(","))
               .explode("matched_entity_ids").drop_nulls()
               .rename({"source1_entity_id": "s1", "matched_entity_ids": "entity_id"}))
    s1 = read(src_path("train", 1)).select(pl.col("entity_id").alias("s1"),
                                            pl.col("business_name").alias("n1"))
    votes = collections.defaultdict(collections.Counter)
    for k in (2, 3):
        sk = read(src_path("train", k)).filter(pl.col("business_name").str.contains(r"[ऀ-෿]"))
        j = sk.join(pairs, on="entity_id").join(s1, on="s1")
        for n1, nk in zip(j["n1"], j["business_name"]):
            a = latinize(RE_PUNCT.sub(" ", n1)).split()
            b = RE_PUNCT.sub(" ", nk).split()
            if len(a) != len(b):
                continue
            for x, y in zip(b, a):
                if RE_INDIC.search(x):
                    votes[x][y] += 1
    out = {}
    for tok, c in votes.items():
        best, n = c.most_common(1)[0]
        tot = sum(c.values())
        if n >= 2 and n / tot >= 0.4:
            out[tok] = best
    json.dump(out, open(work_path("indic_dict.json"), "w"), ensure_ascii=False)
    print(f"learned {len(out)} indic tokens from {len(votes)} candidates")


if __name__ == "__main__":
    main()
