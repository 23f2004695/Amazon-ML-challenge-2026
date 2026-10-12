"""Chance house-number collisions in countries unseen in training (France).

French house numbers are small (86% have 1-2 digits) and French names repeat heavily (~18 S1
per core name), so a record and a same-name S1 in another street of the same city often share
the house number by chance. A model trained on US/India house numbers (3-5 digits, where an
equal number is strong evidence) links such pairs. Label-free signature: among same-name,
equal-house-number top pairs, the share whose streets differ rises from 0.06% for unique names
to 7% for names shared by 51+ S1 (flat 0.3% in US, 3.3% in India), and is 4-10x lower for
3+ digit numbers, as chance collisions predict and street noise does not.
Rule (unseen countries only): the record's top pair is linked, has the same core-name token set,
equal non-empty house numbers, different streets (street similarity < 60), and the S1 name is
shared by >= 11 S1 of the country -> unlinked. Pairs whose street string reappears elsewhere in
the other address (street extractor picked a floor / building part) are kept.
"""
import re
import unicodedata

import polars as pl
from rapidfuzz import fuzz

from config import work_path
from normalize import STREET, latinize

QOFF = 8_000_000
NAMEFREQ_MIN = 11
TYPES = set(STREET.values()) | set(STREET.keys()) | {
    "r", "q", "rpt", "sq", "pas", "passage", "lot", "lotissement", "cite", "sente", "bis", "ter", "b", "t", "a", "no", "n",
    "de", "du", "des", "la", "le", "les", "d", "l", "et", "st", "ste", "saint", "sainte"}
STREET_WORDS = ("rue", "r", "avenue", "ave", "av", "boulevard", "bd", "blvd", "allee", "all", "place", "pl", "impasse", "imp",
                "chemin", "ch", "route", "rte", "quai", "cours", "square", "street", "st", "road", "rd", "drive", "dr", "lane", "ln")
RE_W = re.compile(r"[^a-z0-9]+")
# words ignored when looking for the street elsewhere in the other address
TY_X = set("rue r av ave avenue bd boulevard pl place all allee imp impasse chem chemin rte route quai q cours sq square pas "
           "passage de du des la le les d l et bis ter b t no n ndeg nord gironde loire atlantique hauts france nouvelle "
           "aquitaine pays pas calais".split())


def street(raw, hn):
    """Street name of an address: the comma part holding the house number (else one with a street
    word), without numbers, street types and articles."""
    if not raw:
        return ""
    comps = [c for c in raw.split(",") if c.strip()]
    pick = None
    if hn:
        for c in comps:
            if hn in [x.lstrip("0") for x in re.findall(r"\d+", c)]:
                pick = c
                break
    if pick is None:
        for c in comps:
            if any(t in STREET_WORDS for t in RE_W.sub(" ", latinize(c)).split()):
                pick = c
                break
    if pick is None:
        return ""
    return " ".join(t for t in RE_W.sub(" ", latinize(pick)).split()
                    if not t.isdigit() and t not in TYPES and not re.fullmatch(r"\d+[a-z]?", t))


def _clean(s):
    return re.sub(r"\bndeg\d*\b|\bn\d+\b|\b\d+(bis|ter|b|t|a)\b", "", s).strip()


def street_sim(a, b):
    a, b = _clean(a), _clean(b)
    if not a or not b:
        return 100.0
    return max(fuzz.ratio(a, b), fuzz.token_set_ratio(a, b), fuzz.partial_ratio(a, b))


def _lat(s):
    return unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower()


def _parts(raw):
    out = []
    for c in (raw or "").split(","):
        t = [w for w in re.sub(r"[^a-z0-9]+", " ", _lat(c)).split()
             if w not in TY_X and not w.isdigit() and not re.fullmatch(r"\d+[a-z]*", w)]
        if t:
            out.append(" ".join(t))
    return out


def cross_match(qa, sa, qst, sst):
    """Best match of one side's street against any part of the other side's address."""
    b = 0.0
    if sst:
        b = max([b] + [fuzz.ratio(sst, x) for x in _parts(qa)])
    if qst:
        b = max([b] + [fuzz.ratio(qst, x) for x in _parts(sa)])
    return b


def rule_pairs(pred, thr=0.66):
    """pred: DataFrame(qid, s1, p) of unseen-country records. Returns the (qid, s1) links to drop."""
    top = (pred.group_by("qid").agg(pl.col("s1").sort_by("p").last(), pl.col("p").max()).filter(pl.col("p") > thr))
    q = pl.concat([pl.read_parquet(work_path(f"test_s{k}.parquet"), columns=["idx", "country", "name_core", "addr_raw", "addr_hn"])
                     .select((pl.col("idx") + off).cast(pl.UInt32).alias("qid"), "country", "name_core", "addr_raw", "addr_hn")
                   for k, off in ((2, 0), (3, QOFF))]).join(top.select("qid"), on="qid", how="semi")
    s1 = pl.read_parquet(work_path("test_s1.parquet"), columns=["idx", "country", "name_core", "addr_raw", "addr_hn"])
    freq = s1.group_by("country", "name_core").len().rename({"len": "namefreq", "name_core": "s_name_core"})
    s1 = s1.select(pl.col("idx").cast(pl.UInt32).alias("s1"), pl.col("name_core").alias("s_name_core"),
                   pl.col("addr_raw").alias("s_addr_raw"), pl.col("addr_hn").alias("s_addr_hn"))
    d = (top.join(q, on="qid").join(s1, on="s1").join(freq, on=["country", "s_name_core"], how="left")
            .with_columns(pl.col(pl.String).fill_null("")))
    qt, st = pl.col("name_core").str.split(" ").list.unique(), pl.col("s_name_core").str.split(" ").list.unique()
    d = d.filter((pl.col("name_core") != "") & (qt.list.set_difference(st).list.len() == 0) & (st.list.set_difference(qt).list.len() == 0)
                 & (pl.col("addr_hn") != "") & (pl.col("addr_hn") == pl.col("s_addr_hn")) & (pl.col("namefreq") >= NAMEFREQ_MIN))
    qs = [street(a, h) for a, h in zip(d["addr_raw"].to_list(), d["addr_hn"].to_list())]
    ss = [street(a, h) for a, h in zip(d["s_addr_raw"].to_list(), d["s_addr_hn"].to_list())]
    d = d.with_columns(pl.Series("qst", qs), pl.Series("sst", ss))
    d = d.filter((pl.col("qst") != "") & (pl.col("sst") != ""))
    d = d.with_columns(pl.Series("ss", [street_sim(a, b) for a, b in zip(d["qst"].to_list(), d["sst"].to_list())], dtype=pl.Float64))
    d = d.filter(pl.col("ss") < 60)
    d = d.with_columns(pl.Series("xb", [cross_match(*r) for r in zip(d["addr_raw"].to_list(), d["s_addr_raw"].to_list(),
                                                                      d["qst"].to_list(), d["sst"].to_list())], dtype=pl.Float64))
    return d.filter(pl.col("xb") < 100).select("qid", "s1")


def apply(pred, unseen_qids, thr=0.66):
    """Set p = 0 for the rule's pairs among the records of unseen countries."""
    drop = rule_pairs(pred.join(unseen_qids, on="qid", how="semi"), thr)
    print(f"France chance house-number collisions unlinked: {drop.height:,} pairs")
    return (pred.join(drop.with_columns(pl.lit(True).alias("_d")), on=["qid", "s1"], how="left")
                .with_columns(pl.when(pl.col("_d")).then(pl.lit(0.0, pl.Float32)).otherwise(pl.col("p")).alias("p")).drop("_d"))
