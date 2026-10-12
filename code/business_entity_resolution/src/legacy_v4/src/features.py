"""Stage 3: pairwise features for every (query, S1) candidate pair.

Three feature families:
  * blocking context  - IDF scores per key type, rank of the pair within the query's
                        and within the S1 record's candidate lists, gaps to the best;
  * string similarity - rapidfuzz scores on normalized names / addresses, house-number,
                        state and legal-form agreement, website-domain similarity;
  * genericness       - how many S1 records share the exact same core name.
Country is never used as a feature, so unseen labels (France) are handled the same way.

Output: work/{split}_feat/part_XXX.parquet (train files also carry `label`).
Usage: python features.py train|test
"""
import glob
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import polars as pl
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler, Levenshtein

from config import N_JOBS, work_path
from normalize import LEGAL

QOFF = 8_000_000
TOPK_MODEL = 10          # candidates per query fed to the model (= candidate_pairs.tsv)
RE_INDIC = r"[ऀ-෿]"

TEXT_COLS = ["name_full", "name_core", "domain", "addr_toks", "addr_nums", "state"]
STR_FEATS = [
    "n_ratio", "n_tset", "c_ratio", "c_tsort", "c_tset", "c_partial", "c_jw", "c_jacc",
    "c_first_eq", "c_len_q", "c_len_s", "c_sub", "dom_sim", "legal_eq",
    "a_tset", "a_ratio", "a_partial", "a_jacc", "hn_eq", "hn_in", "nums_jacc", "st_eq",
    "qa_empty", "sa_empty",
    # distinctive name part: core tokens that do not occur in either address (drops city /
    # street words inside names, e.g. 'Lille Club' vs 'Lille SAS Et Fils' -> 'club' vs 'fils')
    "d_ratio", "d_tset", "d_jacc", "d_len_q", "d_len_s",
    # name tokens with no fuzzy counterpart on the other side ('ZF Trading Holdings')
    "c_extra_q", "c_extra_s",
    # house-number relation: decoys sit at a *nearby* number (203 vs 202), while noise
    # produces digit typos (325 -> 425) or truncations (325 -> 32)
    "hn_logdiff", "hn_reldiff", "hn_edit", "hn_prefix", "nums_min_logdiff",
]


def _n_unmatched(a, b):
    """How many tokens of a have no token in b with fuzz.ratio >= 80."""
    if not b:
        return len(a)
    return sum(1 for t in a if process.extractOne(t, b, scorer=fuzz.ratio, score_cutoff=80) is None)


def _jacc(a, b):
    if not a or not b:
        return -1.0
    return len(a & b) / len(a | b)


def _distinct(qct, sct, qat, sat):
    """Similarity of the name tokens that are not address words (-1 when either is empty)."""
    addr = set(qat) | set(sat)
    qd = [t for t in qct if t not in addr]
    sd = [t for t in sct if t not in addr]
    if not qd or not sd:
        return -1.0, -1.0, -1.0, len(qd), len(sd)
    qs, ss = " ".join(qd), " ".join(sd)
    return fuzz.ratio(qs, ss), fuzz.token_set_ratio(qs, ss), _jacc(set(qd), set(sd)), len(qd), len(sd)


def _num(x):
    return int(x[:9])


def _hn_relation(qns, sns):
    """Numeric / edit relation of the first house numbers, plus the closest number pair."""
    if not qns or not sns:
        return -1.0, -1.0, -1.0, -1.0, -1.0
    x, y = qns[0], sns[0]
    a, b = _num(x), _num(y)
    diff = abs(a - b)
    mind = min(abs(_num(u) - _num(v)) for u in qns[:4] for v in sns[:4])
    return (float(np.log1p(diff)), diff / max(a, b, 1), float(Levenshtein.distance(x, y)),
            float(x != y and (x.startswith(y) or y.startswith(x))), float(np.log1p(mind)))


def _str_feats(args):
    """Compute STR_FEATS for a batch of pairs (runs in a worker process)."""
    q, s = args
    n = len(q[0])
    out = np.empty((n, len(STR_FEATS)), dtype=np.float32)
    for i in range(n):
        qn, qc, qd, qa, qnum, qst = (col[i] for col in q)
        sn, sc, sd, sa, snum, sst = (col[i] for col in s)
        qct, sct = qc.split(), sc.split()
        qcs, scs = set(qct), set(sct)
        qcat, scat = "".join(qct), "".join(sct)
        ql = {t for t in qn.split() if t in LEGAL and len(t) > 2}
        sl = {t for t in sn.split() if t in LEGAL and len(t) > 2}
        if qd:
            dom = max(fuzz.ratio(qd, scat), fuzz.partial_ratio(qd, scat) * 0.9)
        else:
            dom = -1.0
        qat, sat = qa.split(), sa.split()
        qns, sns = qnum.split(), snum.split()
        if qns and sns:
            hn_eq = float(qns[0] == sns[0])
            hn_in = float(qns[0] in sns)
        else:
            hn_eq = hn_in = -1.0
        out[i] = (
            fuzz.ratio(qn, sn), fuzz.token_set_ratio(qn, sn),
            fuzz.ratio(qc, sc), fuzz.token_sort_ratio(qc, sc), fuzz.token_set_ratio(qc, sc),
            fuzz.partial_ratio(qc, sc) if qc and sc else 0.0,
            JaroWinkler.similarity(qcat, scat) * 100,
            _jacc(qcs, scs),
            float(bool(qct) and bool(sct) and qct[0] == sct[0]),
            len(qct), len(sct),
            float(bool(qcat) and bool(scat) and (qcat in scat or scat in qcat)),
            dom,
            -1.0 if not ql or not sl else float(bool(ql & sl)),
            fuzz.token_set_ratio(qa, sa) if qa and sa else -1.0,
            fuzz.ratio(qa, sa) if qa and sa else -1.0,
            fuzz.partial_ratio(qa, sa) if qa and sa else -1.0,
            _jacc(set(qat), set(sat)),
            hn_eq, hn_in,
            _jacc(set(qns), set(sns)),
            -1.0 if not qst or not sst else float(qst == sst),
            float(not qa and not qnum), float(not sa and not snum),
            *_distinct(qct, sct, qat, sat),
            _n_unmatched(qct, sct), _n_unmatched(sct, qct),
            *_hn_relation(qns, sns),
        )
    return out


def load_text(split):
    """Return (query_text, s1_text) DataFrames keyed by qid / s1."""
    cols = ["idx", "name_raw"] + TEXT_COLS
    qs = []
    for k, off in ((2, 0), (3, QOFF)):
        d = pl.read_parquet(work_path(f"{split}_s{k}.parquet"), columns=cols + ["country"])
        qs.append(d.with_columns((pl.col("idx") + off).alias("qid")).drop("idx"))
    q = pl.concat(qs)
    s1 = pl.read_parquet(work_path(f"{split}_s1.parquet"), columns=cols + ["country"]).rename({"idx": "s1"})
    # genericness: how many S1 records share this exact core name (per country)
    freq = s1.group_by("country", "name_core").len().rename({"len": "namefreq"})
    s1 = s1.join(freq, on=["country", "name_core"], how="left")
    q = (q.join(freq.rename({"namefreq": "q_namefreq"}), on=["country", "name_core"], how="left")
          .with_columns(pl.col("q_namefreq").fill_null(0),
                        pl.col("name_raw").str.contains(RE_INDIC).cast(pl.Int8).alias("q_indic"),
                        (pl.col("qid") >= QOFF).cast(pl.Int8).alias("q_src3")))
    return q.drop("name_raw", "country"), s1.drop("name_raw", "country")


def s1_context(split):
    """S1-level context over *all* model candidates (one streaming pass)."""
    parts = sorted(glob.glob(work_path(f"{split}_cand/part_*.parquet")))
    return (pl.scan_parquet(parts).filter(pl.col("brank") < TOPK_MODEL)
              .group_by("s1").agg(pl.len().cast(pl.UInt32).alias("s_nq"), pl.col("bscore").max().alias("s_maxb"))
              .collect(engine="streaming"))


def query_context(cand):
    """Rank / gap features of each pair within its query's candidate list."""
    cand = cand.with_columns(
        pl.len().over("qid").alias("q_ncand"),
        pl.col("bscore").max().over("qid").alias("q_maxb"),
        pl.col("bscore").sort(descending=True).slice(1, 1).first().over("qid").alias("q_2ndb"),
    )
    return cand.with_columns(
        (pl.col("bscore") - pl.col("q_maxb")).alias("gap_qmax"),
        (pl.col("bscore") - pl.col("q_2ndb").fill_null(0)).alias("gap_q2nd"),
        (pl.col("bscore") / pl.col("q_maxb")).alias("r_qmax"),
    )


def context_feats(cand, sctx):
    """Query-level context plus the S1-level context."""
    cand = query_context(cand.filter(pl.col("brank") < TOPK_MODEL)).join(sctx, on="s1", how="left")
    return cand.with_columns((pl.col("bscore") - pl.col("s_maxb")).alias("gap_smax"))


DECOY_FEATS = ["dq_max", "dq_sum", "dq_n", "ds_max", "ds_sum", "ds_n"]


def decoy_feats(part, split, ddict):
    """Log-odds that the words present on only one side mark a decoy (see decoy_dict.py).
    Train rows use the dictionary learned on the other query fold; test uses all of train."""
    qt = pl.col("name_full").str.split(" ").list.unique()
    st = pl.col("name_full_s").str.split(" ").list.unique()
    dfold = ((1 - pl.col("qid").hash(seed=11) % 2) if split == "train" else pl.lit(2)).cast(pl.Int8)
    x = part.select("qid", "s1", dfold.alias("dfold"), qt.list.set_difference(st).alias("q"),
                    st.list.set_difference(qt).alias("s"))
    out = x.select("qid", "s1")
    for side in ("q", "s"):
        e = (x.select("qid", "s1", "dfold", pl.col(side).alias("token")).explode("token").drop_nulls("token")
               .join(ddict.filter(pl.col("side") == side).drop("side"), on=["token", "dfold"], how="inner")
               .group_by("qid", "s1").agg(pl.col("lo").max().alias(f"d{side}_max"), pl.col("lo").sum().alias(f"d{side}_sum"),
                                          pl.len().cast(pl.Float32).alias(f"d{side}_n")))
        out = out.join(e, on=["qid", "s1"], how="left")
    return out.with_columns(pl.col(DECOY_FEATS).fill_null(0.0))


def main(split):
    t0 = time.time()
    pool = Pool(N_JOBS)  # fork workers before loading large tables
    outdir = work_path(f"{split}_feat")
    os.makedirs(outdir, exist_ok=True)
    sctx = s1_context(split)
    truth = None
    if split == "train":
        from evaluate import load_truth
        truth = load_truth().with_columns(pl.col("qid").cast(pl.UInt32), pl.col("s1_true").cast(pl.UInt32))
    ddict = pl.read_parquet(work_path("decoy_dict.parquet"))
    qtext, stext = load_text(split)
    print(f"context + text loaded [{time.time() - t0:.0f}s]", flush=True)
    for p in sorted(glob.glob(work_path(f"{split}_cand/part_*.parquet"))):
        out = os.path.join(outdir, os.path.basename(p))
        if os.path.exists(out):  # resumable
            continue
        cand = context_feats(pl.read_parquet(p), sctx)
        if truth is not None:
            cand = (cand.join(truth, on="qid", how="left")
                        .with_columns((pl.col("s1") == pl.col("s1_true")).fill_null(False).cast(pl.Int8).alias("label"))
                        .drop("s1_true"))
        part = (cand.join(qtext, on="qid", how="left")
                    .join(stext, on="s1", how="left", suffix="_s"))
        q = [part[c].fill_null("").to_list() for c in TEXT_COLS]
        s = [part[c + "_s"].fill_null("").to_list() for c in TEXT_COLS]
        step = 10000
        jobs = [([c[i:i + step] for c in q], [c[i:i + step] for c in s]) for i in range(0, part.height, step)]
        feats = np.vstack(pool.map(_str_feats, jobs))
        sf = pl.DataFrame(feats, schema=STR_FEATS)
        dec = decoy_feats(part.select("qid", "s1", "name_full", "name_full_s"), split, ddict)
        part = part.join(dec, on=["qid", "s1"], how="left", maintain_order="left")
        keep = part.drop([c for c in part.columns if c in TEXT_COLS or c.endswith("_s")])
        pl.concat([keep, sf], how="horizontal").write_parquet(out)
        print(f"  {os.path.basename(p)}: {part.height:,} pairs [{time.time() - t0:.0f}s]", flush=True)
    pool.close()


if __name__ == "__main__":
    main(sys.argv[1])
