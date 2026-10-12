"""Stage 4c: third-stage re-scoring with group context recomputed from stage-2 probabilities.

Stage-2 group features (query / S1 competition, sibling agreement, decoy families, fuzzy
sibling similarity) are built from stage-1 probabilities. Stage-2 probabilities are much
sharper, so the same group features recomputed from p2 describe the entity's confidently
linked records more cleanly. Stage 3 = all stage-2 features + p2 + the p2-based group
features (suffix _r2), trained with the same query folds (leak-free: p2 is out-of-fold).

Usage: python stage3.py          (train on BER_WORK, OOF threshold search)
       python stage3.py test     (test: test_pred1/2{TAG}{VAR} -> test_pred3{TAG}{VAR})
"""
import json
import multiprocessing as mp
import os
import sys
import time

import lightgbm as lgb
import numpy as np
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist

from config import N_JOBS, TAG, work_path
from evaluate import eval_population, f05_macro
from stage2 import LINK_P, attach_stage1_feats, fit, group_feats, sibling_fuzzy
from train import NON_FEATS, assign, cap_links, with_fold

DUP = 20_000_000  # synthetic decoy duplicates (make_decoy_aug.py)
QOFF = 8_000_000
LO, HI = 0.01, 0.99  # uncertain region re-scored by stage 3
# BER_STAGE=4 repeats the step on top of stage 3 (group context from p3, files *4*)
ST = int(os.environ.get("BER_STAGE", "3"))


def _style(c):
    return (pl.when(c == c.str.to_uppercase()).then(0).when(c == c.str.to_lowercase()).then(1)
              .when(c == c.str.to_titlecase()).then(2).otherwise(3).cast(pl.Int8))


def _punct(c):
    return c.str.replace_all(r"[\p{L}\p{N}\s]", "").str.split("").list.unique().list.sort().list.join("")


def extra_feats(keys, split):
    """Evidence stage 1/2 do not see (all language-independent, no S1-population statistics):
      * house-number arithmetic on the leading digit runs (difference, padding, prefix, length);
      * query noise markers (the generator's noise, rarely present on unmatched records);
      * query flags: no house number, domain-style name, d/b/a;
      * raw-name formatting agreement (case, punctuation, commas) that normalization discards,
        and how many of the query's candidates share the same normalized name (twins)."""
    q = pl.concat([pl.read_parquet(work_path(f"{split}_s{k}.parquet"),
                                   columns=["idx", "name_raw", "addr_raw", "addr_hn", "domain", "name_full", "name_core"])
                     .select((pl.col("idx") + off).cast(pl.UInt32).alias("qid"), pl.all().exclude("idx").name.prefix("q_"))
                   for k, off in ((2, 0), (3, QOFF))]).join(keys.select("qid").unique(), on="qid", how="semi")
    s = pl.read_parquet(work_path(f"{split}_s1.parquet"), columns=["idx", "name_raw", "addr_hn", "name_full", "name_core"]).select(
        pl.col("idx").cast(pl.UInt32).alias("s1"), pl.all().exclude("idx").name.prefix("s_"))
    d = keys.join(q, on="qid", how="left").join(s, on="s1", how="left").with_columns(pl.col(pl.String).fill_null(""))
    n, a = pl.col("q_name_raw"), pl.col("q_addr_raw")
    lead = lambda c: pl.col(c).str.extract(r"(\d+)", 1)
    d = d.with_columns(lead("q_addr_hn").alias("qd"), lead("s_addr_hn").alias("sd"))
    d = d.with_columns(pl.col("qd").str.strip_chars_start("0").alias("qd0"), pl.col("sd").str.strip_chars_start("0").alias("sd0"))
    d = d.with_columns(
        (pl.col("qd0").cast(pl.Int64, strict=False) - pl.col("sd0").cast(pl.Int64, strict=False)).abs().alias("hn_absdiff"),
        (pl.col("qd0") == pl.col("sd0")).cast(pl.Int8).alias("hn_digits_eq"),
        (pl.col("qd") != pl.col("qd0")).cast(pl.Int8).alias("hn_q_leadzero"),
        (pl.col("qd0").str.len_chars().cast(pl.Int32) - pl.col("sd0").str.len_chars().cast(pl.Int32)).alias("hn_lendiff"),
        (pl.col("sd0").str.starts_with(pl.col("qd0")) | pl.col("qd0").str.starts_with(pl.col("sd0"))).cast(pl.Int8).alias("hn_isprefix"),
        (pl.col("q_addr_hn") == pl.col("s_addr_hn")).cast(pl.Int8).alias("hn_raw_eq"),
        n.str.contains(r"#\d{3,}").cast(pl.Int8).alias("nz_hashnum"),
        a.str.contains("##").cast(pl.Int8).alias("nz_dblhash"),
        n.str.contains(r"[A-Za-z][0-9][A-Za-z]|[0-9][A-Za-z]{2,}|[a-z][A-Z]{1}[a-z]").cast(pl.Int8).alias("nz_ocr"),
        n.str.contains(r"(?i)^(m/s|mr|mrs|dr|smt|shri|sri|the|>>|\.\.\.|co)\b|^>>|^\.\.\.").cast(pl.Int8).alias("nz_prefix"),
        n.str.contains(r"[\[\(].*[\]\)]").cast(pl.Int8).alias("nz_bracket"),
        a.str.contains(r"(?i)<NULL>|\bN/A\b|\bnull\b").cast(pl.Int8).alias("nz_nulltok"),
        a.str.contains(r"(?i)\bPMB\b|PO Box").cast(pl.Int8).alias("nz_pmb"),
        n.str.contains(r"(?i)www\.|\.com").cast(pl.Int8).alias("nz_web"),
        n.str.contains(r"\d{10}").cast(pl.Int8).alias("nz_phone"),
        (n.str.to_lowercase().str.split(" ").list.eval(pl.element().filter(pl.element() != "")).list.len()
         - n.str.to_lowercase().str.split(" ").list.eval(pl.element().filter(pl.element() != "")).list.unique().list.len()).alias("nz_duptok"),
        (pl.col("q_addr_hn") == "").cast(pl.Int8).alias("q_nohn"),
        (pl.col("q_domain") != "").cast(pl.Int8).alias("q_domain_flag"),
        n.str.contains(r"(?i)\bd/?b/?a\b|trading as").cast(pl.Int8).alias("q_dba"),
        pl.len().over("qid", "s_name_core").alias("tie_core"), pl.len().over("qid", "s_name_full").alias("tie_full"),
        (n == pl.col("s_name_raw")).cast(pl.Int8).alias("raw_eq"),
        (n.str.to_lowercase().str.strip_chars() == pl.col("s_name_raw").str.to_lowercase().str.strip_chars()).cast(pl.Int8).alias("rawci_eq"),
        (pl.col("q_name_full") == pl.col("s_name_full")).cast(pl.Int8).alias("full_eq"),
        (_style(n) == _style(pl.col("s_name_raw"))).cast(pl.Int8).alias("style_eq"), _style(n).alias("q_style"),
        (_punct(n) == _punct(pl.col("s_name_raw"))).cast(pl.Int8).alias("punct_eq"),
        (n.str.contains(",") == pl.col("s_name_raw").str.contains(",")).cast(pl.Int8).alias("comma_eq"),
    ).with_columns((pl.col("hn_absdiff") <= 1).cast(pl.Int8).alias("hn_pm1"), (pl.col("hn_absdiff") <= 10).cast(pl.Int8).alias("hn_le10"))
    qa, sa = d["q_name_raw"].to_list(), d["s_name_raw"].to_list()
    d = d.with_columns(
        pl.Series("raw_ratio", cpdist(qa, sa, scorer=fuzz.ratio, workers=-1).astype(np.float32)),
        pl.Series("raw_ratio_ci", cpdist(qa, sa, scorer=fuzz.ratio, processor=str.lower, workers=-1).astype(np.float32)))
    return d.drop([c for c in d.columns if c[:2] in ("q_", "s_") and c not in ("q_nohn", "q_domain_flag", "q_dba", "q_style")]
                  + ["qd", "sd", "qd0", "sd0"])


def _links(pred, col):
    best = pred.group_by("qid").agg(pl.col("s1").sort_by(col).last(), pl.col(col).max())
    return best.filter(pl.col(col) > LINK_P).select("qid", "s1")


def build(split, p1, p2, pool):
    """Stage-3 rows: the pairs of the uncertain region LO < p2 < HI (the others cannot change
    decisions and keep p2). Group context is computed on the full pair sets, then restricted."""
    region = p2.filter((pl.col("p") > LO) & (pl.col("p") < HI)).select("qid", "s1")
    p2 = p2.select("qid", "s1", pl.col("p").alias("p1"))
    G = group_feats(p2, split, None, expand=False).drop("p1_missing").join(region, on=["qid", "s1"], how="semi")
    G = sibling_fuzzy(G, _links(p2, "p1"), split, pool)
    G = G.rename({c: c + "_r2" for c in G.columns if c not in ("qid", "s1")})
    del p2
    links1 = _links(p1, "p1")
    P = group_feats(p1, split, None).join(region, on=["qid", "s1"], how="semi")
    del p1
    P = attach_stage1_feats(sibling_fuzzy(P, links1, split, pool), split, pool)
    P = P.join(G, on=["qid", "s1"], how="left")
    del G
    return P.join(extra_feats(P.select("qid", "s1"), split), on=["qid", "s1"], how="left")


def merge(p2, p3_region):
    """Full pair set: p3 inside the region, p2 elsewhere."""
    return (p2.select("qid", "s1", "p").join(p3_region.select("qid", "s1", pl.col("p").alias("p3")), on=["qid", "s1"], how="left")
              .with_columns(pl.coalesce("p3", "p").cast(pl.Float32).alias("p")).drop("p3"))


def main():
    t0 = time.time()
    pool = mp.get_context("spawn").Pool(min(N_JOBS, 10))
    oof2 = pl.read_parquet(work_path(f"train_oof{ST - 1}{TAG}.parquet"), columns=["qid", "s1", "p"])
    P = build("train", pl.read_parquet(work_path(f"train_oof{TAG}.parquet"), columns=["qid", "s1", "p"]).rename({"p": "p1"}),
              oof2, pool)
    pool.close()
    P = with_fold(P).drop("_b")
    P.write_parquet(work_path(f"stage{ST}_rows{TAG}.parquet"))  # reused by stage3_bag.py
    feats = [c for c in P.columns if c not in NON_FEATS]
    print(f"stage-3 rows {P.height:,}, {len(feats)} feats [{time.time() - t0:.0f}s]", flush=True)
    p3 = np.zeros(P.height, dtype=np.float32)
    for f in (0, 1):
        m = fit(P.filter(pl.col("fold") == f), feats)
        m.save_model(work_path(f"lgb{ST}_fold{f}{TAG}.txt"))
        mask = (P["fold"] != f).to_numpy()
        p3[mask] = m.predict(P.filter(pl.col("fold") != f).select(feats).to_numpy().astype(np.float32))
        print(f"stage-3 model {f} [{time.time() - t0:.0f}s]", flush=True)
    imp = sorted(zip(m.feature_importance("gain"), feats), reverse=True)
    print("top gain:", [(n, int(g)) for g, n in imp[:25]])
    oof3 = merge(oof2, P.select("qid", "s1").with_columns(pl.Series("p", p3)))
    del P
    oof3.write_parquet(work_path(f"train_oof{ST}{TAG}.parquet"))
    truth, s1_ids = eval_population()
    best = None
    for thr in np.arange(0.54, 0.80, 0.04):
        score = f05_macro(cap_links(assign(oof3, thr)).select("qid", "s1"), truth, s1_ids)[0]
        # the same decisions without the synthetic duplicates (plain withheld-S1 simulation)
        plain = f05_macro(cap_links(assign(oof3.filter(pl.col("qid") < DUP), thr)).select("qid", "s1"), truth, s1_ids)[0]
        print(f"thr {thr:.2f}: F0.5 {score:.5f}  (no duplicates {plain:.5f})", flush=True)
        if best is None or score > best[1]:
            best = (float(thr), score)
    for thr in (0.62, 0.66, 0.70):
        s2 = f05_macro(cap_links(assign(oof2, thr)).select("qid", "s1"), truth, s1_ids)[0]
        s2p = f05_macro(cap_links(assign(oof2.filter(pl.col("qid") < DUP), thr)).select("qid", "s1"), truth, s1_ids)[0]
        print(f"stage-2 reference thr {thr:.2f}: F0.5 {s2:.5f}  (no duplicates {s2p:.5f})", flush=True)
    print("best", best)
    json.dump({"threshold": best[0], "oof_f05": best[1], "feats": feats}, open(work_path(f"model{ST}_meta{TAG}.json"), "w"))


def predict_test():
    from predict import VAR, unseen_matrix
    meta = json.load(open(work_path(f"model{ST}_meta{TAG}.json")))
    pool = mp.get_context("spawn").Pool(min(N_JOBS, 10))
    p2 = pl.read_parquet(work_path(f"test_pred{ST - 1}{TAG}{VAR}.parquet"))
    P = build("test", pl.read_parquet(work_path(f"test_pred1{TAG}{VAR}.parquet")), p2, pool)
    pool.close()
    P.write_parquet(work_path(f"stage{ST}_rows_test{TAG}{VAR}.parquet"))  # reused by stage3_bag.py
    X = unseen_matrix(P, P.select(meta["feats"]).to_numpy().astype(np.float32), meta["feats"])
    keys = P.select("qid", "s1")
    del P
    models = [lgb.Booster(model_file=work_path(f"lgb{ST}_fold{f}{TAG}.txt")) for f in (0, 1)]
    p3 = np.mean([m.predict(X) for m in models], axis=0)
    merge(p2, keys.with_columns(pl.Series("p", p3.astype(np.float32)))).write_parquet(work_path(f"test_pred{ST}{TAG}{VAR}.parquet"))
    print(f"test stage 3: {keys.height:,} region pairs re-scored")


if __name__ == "__main__":
    predict_test() if len(sys.argv) > 1 and sys.argv[1] == "test" else main()
