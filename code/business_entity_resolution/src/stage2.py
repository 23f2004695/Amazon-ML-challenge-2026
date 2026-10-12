"""Stage 4b: second-stage (group-aware) re-scoring of plausible pairs.

Stage-1 scores each pair in isolation. Stage 2 re-scores every pair with stage-1
probability p1 >= P_FLOOR using all stage-1 features plus features describing the
*group* the pair would join:
  * query competition   - margin to the query's best other candidate, # strong options;
  * S1 competition      - how many other queries claim the same S1, and how strongly;
  * sibling agreement   - how many records already confidently linked to this S1 share
                          the query's exact raw address / normalized address / core name
                          / full name (records of one entity inside a source often share
                          the same address variant, even when the name was replaced), and
                          the best fuzzy address / name similarity to those records.
  * sibling expansion   - a record that shares an exact normalized address / core name /
                          full name with a record confidently linked to S1 X gets X as an
                          extra candidate (only when that text points to <= 2 linked S1s).
                          This recovers true pairs that blocking never retrieved, mostly
                          name-only records whose generic name matches many S1 entities.
                          New pairs get the text features of stage 1; their blocking-context
                          features and p1 are missing (p1_missing = 1).
Train p1 comes from out-of-fold stage-1 predictions, so stage-2 CV (same query folds)
is leak-free.

Usage: python stage2.py            (train + OOF threshold search)
"""
import glob
import json
import multiprocessing as mp
import time

import lightgbm as lgb
import numpy as np
import polars as pl
import polars.selectors as cs

from rapidfuzz import fuzz

from config import N_JOBS, SEED, TAG, TRAIN_FEAT, work_path
from evaluate import eval_population, f05_macro
from train import NON_FEATS, assign, with_fold

QOFF = 8_000_000
P_FLOOR = 0.01
LINK_P = 0.5
PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=127, min_data_in_leaf=200,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
              num_threads=14, verbose=-1, seed=SEED)
N_ROUNDS = 500
HASH_COLS = ["h_addr_raw", "h_addr_norm", "h_name_core", "h_name_full"]


def text_hashes(split):
    """Per-query hashes of address / name variants (null when the field is empty)."""
    out = []
    for k, off in ((2, 0), (3, QOFF)):
        d = pl.read_parquet(work_path(f"{split}_s{k}.parquet"),
                            columns=["idx", "country", "addr_raw", "addr_toks", "addr_nums", "name_core", "name_full"])
        norm = pl.concat_str([pl.col("addr_toks"), pl.col("addr_nums")], separator="|")

        def h(e, empty):
            return pl.when(empty).then(None).otherwise(pl.concat_str([pl.col("country"), e], separator="#").hash(seed=3))
        out.append(d.select(
            (pl.col("idx") + off).cast(pl.UInt32).alias("qid"),
            h(pl.col("addr_raw"), pl.col("addr_raw").str.strip_chars() == "").alias("h_addr_raw"),
            h(norm, norm == "|").alias("h_addr_norm"),
            h(pl.col("name_core"), pl.col("name_core") == "").alias("h_name_core"),
            h(pl.col("name_full"), pl.col("name_full") == "").alias("h_name_full"),
        ))
    return pl.concat(out)


EXP_HASHES = ["h_addr_norm", "h_name_core", "h_name_full"]
EXP_MAX_S1 = 2


def expansion_pairs(links, hashes):
    """(qid, s1) proposed because the query shares an exact normalized address / core name /
    full name with a record confidently linked to s1 (text pointing to <= EXP_MAX_S1 S1s)."""
    L = links.join(hashes, on="qid")
    out = []
    for hc in EXP_HASHES:
        m = (L.filter(pl.col(hc).is_not_null()).group_by(hc).agg(pl.col("s1").unique())
               .filter(pl.col("s1").list.len() <= EXP_MAX_S1).explode("s1"))
        out.append(hashes.filter(pl.col(hc).is_not_null()).select("qid", hc).join(m, on=hc).select("qid", "s1"))
    E = pl.concat(out).unique()
    return E.with_columns(pl.len().over("qid").cast(pl.UInt32).alias("q_n_exp_s1"))


def group_feats(pred, split, pool=None, expand=True):
    """pred: DataFrame(qid, s1, p1) for ALL scored pairs. Returns stage-2 rows: pairs with
    p1 >= P_FLOOR plus sibling-expansion pairs."""
    pred = pred.with_columns(pl.col("qid").cast(pl.UInt32), pl.col("s1").cast(pl.UInt32),
                             pl.col("p1").cast(pl.Float32))
    best = (pred.group_by("qid").agg(pl.col("s1").sort_by("p1").last().alias("best_s1"),
                                    pl.col("p1").max().alias("q_p1max")))
    links = best.filter(pl.col("q_p1max") > LINK_P).select("qid", pl.col("best_s1").alias("s1"))
    hashes = text_hashes(split)
    P = pred.filter(pl.col("p1") >= P_FLOOR)
    if expand:
        E = expansion_pairs(links, hashes)
        low = pred.filter(pl.col("p1") < P_FLOOR).join(E, on=["qid", "s1"], how="semi")
        new = (E.select("qid", "s1").join(pred.select("qid", "s1"), on=["qid", "s1"], how="anti")
                .with_columns(pl.lit(None, pl.Float32).alias("p1")))
        P = pl.concat([P, low, new], how="vertical_relaxed")
        P = (P.join(E.select("qid", "q_n_exp_s1").unique("qid"), on="qid", how="left")
              .with_columns(pl.col("q_n_exp_s1").fill_null(0)))
        print(f"  expansion: {E.height:,} proposals, {low.height:,} low-p1 pairs revived, "
              f"{new.height:,} new pairs", flush=True)
        del E, low, new
    P = P.with_columns(pl.col("p1").is_null().cast(pl.Int8).alias("p1_missing"), pl.col("p1").fill_null(0.0))
    del pred, best  # the full pair table is no longer needed
    # query competition
    P = P.with_columns(
        pl.col("p1").max().over("qid").alias("q_p1max"),
        pl.col("p1").sort(descending=True).slice(1, 1).first().over("qid").fill_null(0).alias("q_p1_2nd"),
        (pl.col("p1") > 0.3).sum().over("qid").alias("q_n_hi"),
    ).with_columns(
        (pl.col("p1") - pl.when(pl.col("p1") == pl.col("q_p1max")).then(pl.col("q_p1_2nd"))
                          .otherwise(pl.col("q_p1max"))).alias("p1_margin"))
    # S1 competition (excluding the pair itself)
    P = P.with_columns(
        (pl.col("p1").sum().over("s1") - pl.col("p1")).alias("s_p1_sum_other"),
        ((pl.col("p1") > LINK_P).sum().over("s1") - (pl.col("p1") > LINK_P).cast(pl.UInt32)).alias("s_n_hi_other"),
        pl.len().over("s1").alias("s_n_plaus"),
    )
    s_top2 = (P.group_by("s1").agg(pl.col("p1").top_k(2).alias("t")))
    P = (P.join(s_top2, on="s1", how="left")
          .with_columns(pl.when(pl.col("p1") == pl.col("t").list.get(0))
                          .then(pl.col("t").list.get(1, null_on_oob=True))
                          .otherwise(pl.col("t").list.get(0)).fill_null(0).alias("s_p1_max_other"))
          .drop("t"))
    # sibling agreement with confidently linked records of the same S1
    P = P.join(hashes, on="qid", how="left")
    L = links.join(hashes, on="qid", how="left")
    P = (P.join(links.with_columns(pl.lit(1, pl.Int8).alias("self_link")), on=["qid", "s1"], how="left")
          .with_columns(pl.col("self_link").fill_null(0)))
    for hc in HASH_COLS:
        cnt = L.filter(pl.col(hc).is_not_null()).group_by("s1", hc).len().rename({"len": "c"})
        P = (P.join(cnt, on=["s1", hc], how="left")
              .with_columns((pl.col("c").fill_null(0) - pl.when(pl.col(hc).is_not_null())
                             .then(pl.col("self_link").cast(pl.UInt32)).otherwise(0)).alias("sib_" + hc[2:]))
              .drop("c"))
    # per-source room: an S1 never has more than 5 S2 / 6 S3 records in the training truth
    src = (pl.col("qid") >= QOFF).cast(pl.Int8)
    ls = links.with_columns(src.alias("src")).group_by("s1", "src").len().rename({"len": "s_src_links"})
    P = (P.with_columns(src.alias("src")).join(ls, on=["s1", "src"], how="left")
          .with_columns((pl.col("s_src_links").fill_null(0) - pl.col("self_link").cast(pl.UInt32)).alias("s_src_links_other"))
          .with_columns((pl.when(pl.col("src") == 0).then(5).otherwise(6) - pl.col("s_src_links_other")).alias("s_src_room"))
          .drop("s_src_links", "src"))
    n_link = L.group_by("s1").len().rename({"len": "s_n_link"})
    P = (P.join(n_link, on="s1", how="left")
          .with_columns((pl.col("s_n_link").fill_null(0) - pl.col("self_link").cast(pl.UInt32)).alias("s_n_link_other"))
          .drop(HASH_COLS + ["s_n_link", "self_link"]))
    P = family_feats(P, split)
    if pool is not None:
        P = sibling_fuzzy(P, links, split, pool)
    return P


def family_feats(P, split):
    """Language-independent decoy evidence from the S1's other candidate records.
    Noise adds words / changes house numbers independently per record; a decoy (sister
    business) carries its extra word and its own house number on all of its records.
      word_fam_max: max over the query's extra words of how many other candidate records
                    of the same S1 carry the same extra word
      hn_fam:       how many other candidate records of the S1 share the query's house
                    number when it differs from the S1's (-1 equal, -2 missing)"""
    qx = []
    for k, off in ((2, 0), (3, QOFF)):
        d = pl.read_parquet(work_path(f"{split}_s{k}.parquet"), columns=["idx", "name_full", "addr_hn"])
        qx.append(d.select((pl.col("idx") + off).cast(pl.UInt32).alias("qid"), "name_full",
                           pl.col("addr_hn").alias("hn")))
    qx = pl.concat(qx).join(P.select("qid").unique(), on="qid", how="semi")
    sx = pl.read_parquet(work_path(f"{split}_s1.parquet"), columns=["idx", "name_full", "addr_hn"]).select(
        pl.col("idx").cast(pl.UInt32).alias("s1"), pl.col("name_full").alias("name_full_s"),
        pl.col("addr_hn").alias("hn_s"))
    d = P.select("qid", "s1").join(qx, on="qid", how="left").join(sx, on="s1", how="left")
    qt = pl.col("name_full").str.split(" ").list.unique()
    st = pl.col("name_full_s").str.split(" ").list.unique()
    ex = d.select("qid", "s1", qt.list.set_difference(st).alias("x")).explode("x").drop_nulls("x")
    cnt = ex.group_by("s1", "x").len().rename({"len": "fam"})
    wf = ex.join(cnt, on=["s1", "x"]).group_by("qid", "s1").agg((pl.col("fam") - 1).max().alias("word_fam_max"))
    hc = d.filter(pl.col("hn").is_not_null() & (pl.col("hn") != "")).group_by("s1", "hn").len().rename({"len": "hfam"})
    d = (d.select("qid", "s1", "hn", "hn_s").join(hc, on=["s1", "hn"], how="left")
          .with_columns(pl.when(pl.col("hn").is_null() | (pl.col("hn") == "") | pl.col("hn_s").is_null() | (pl.col("hn_s") == ""))
                          .then(pl.lit(-2)).when(pl.col("hn") == pl.col("hn_s")).then(pl.lit(-1))
                          .otherwise(pl.col("hfam") - 1).cast(pl.Int32).alias("hn_fam"))
          .select("qid", "s1", "hn_fam"))
    return (P.join(wf, on=["qid", "s1"], how="left").join(d, on=["qid", "s1"], how="left")
             .with_columns(pl.col("word_fam_max").fill_null(-1)))


def _sib_scores(args):
    """Worker: fuzzy address / name similarity of each (query, sibling) row."""
    qa, qn, sa, sn = args
    out = np.empty((len(qa), 2), dtype=np.float32)
    for i in range(len(qa)):
        out[i, 0] = fuzz.token_set_ratio(qa[i], sa[i]) if qa[i] and sa[i] else -1.0
        out[i, 1] = fuzz.token_set_ratio(qn[i], sn[i]) if qn[i] and sn[i] else -1.0
    return out


def sibling_fuzzy(P, links, split, pool, chunk=400_000):
    """Best fuzzy address / name agreement between a query and the other records
    confidently linked to the same S1 (-1 when there are none)."""
    txt = []
    for k, off in ((2, 0), (3, QOFF)):
        d = pl.read_parquet(work_path(f"{split}_s{k}.parquet"), columns=["idx", "addr_toks", "addr_nums", "name_core"])
        txt.append(d.select((pl.col("idx") + off).cast(pl.UInt32).alias("qid"),
                            pl.concat_str([pl.col("addr_toks"), pl.col("addr_nums")], separator=" ")
                              .str.strip_chars().alias("a"),
                            pl.col("name_core").alias("n")))
    txt = pl.concat(txt)
    L = links.join(txt, on="qid").rename({"qid": "sib", "a": "a_s", "n": "n_s"})
    base = P.select("qid", "s1").join(txt, on="qid")
    res = []
    for i in range(0, base.height, chunk):
        j = base.slice(i, chunk).join(L, on="s1").filter(pl.col("qid") != pl.col("sib"))
        if j.height == 0:
            continue
        cols = [j[c].to_list() for c in ("a", "n", "a_s", "n_s")]
        step = 20000
        jobs = [tuple(c[k:k + step] for c in cols) for k in range(0, j.height, step)]
        sc = np.vstack(pool.map(_sib_scores, jobs))
        j = j.select("qid", "s1").with_columns(pl.Series("fa", sc[:, 0]), pl.Series("fn", sc[:, 1]))
        res.append(j.group_by("qid", "s1").agg(
            pl.col("fa").max().alias("sibf_addr_max"), pl.col("fn").max().alias("sibf_name_max"),
            pl.min_horizontal("fa", "fn").max().alias("sibf_both_max")))
    res = pl.concat(res) if res else pl.DataFrame(schema={"qid": pl.UInt32, "s1": pl.UInt32})
    return (P.join(res, on=["qid", "s1"], how="left")
             .with_columns(pl.col("sibf_addr_max", "sibf_name_max", "sibf_both_max").fill_null(-1.0)))


def attach_stage1_feats(P, split, pool=None):
    """Join the stage-1 feature rows of the pairs in P (streaming over feature parts).
    Pairs without a stage-1 row (sibling expansion) get their text features computed here;
    their blocking-context features stay missing."""
    fdir = TRAIN_FEAT if split == "train" else f"{split}_feat"
    files = sorted(glob.glob(work_path(f"{fdir}/part_*.parquet")))
    keys = P.select("qid", "s1")
    have = pl.concat([pl.read_parquet(p, columns=["qid", "s1"]).join(keys, on=["qid", "s1"], how="semi")
                      for p in files])
    new = keys.join(have, on=["qid", "s1"], how="anti")
    del have
    nf = None
    if new.height and pool is not None:
        from features import load_text, pair_features
        ddict = pl.read_parquet(work_path("decoy_dict.parquet"))
        qtext, stext = load_text(split)
        nf = pair_features(new, qtext, stext, pool, split, ddict)
        del qtext, stext
        if split == "train":
            from evaluate import load_truth
            truth = load_truth().with_columns(pl.col("qid").cast(pl.UInt32), pl.col("s1_true").cast(pl.UInt32))
            nf = (nf.join(truth, on="qid", how="left")
                    .with_columns((pl.col("s1") == pl.col("s1_true")).fill_null(False).cast(pl.Int8).alias("label"))
                    .drop("s1_true"))
        nf = nf.with_columns(cs.float().cast(pl.Float32))
        print(f"  text features for {nf.height:,} expansion pairs", flush=True)
    parts = []
    for p in files:
        d = pl.read_parquet(p).join(keys, on=["qid", "s1"], how="semi")
        parts.append(d.with_columns(cs.float().cast(pl.Float32)))
    out = P.join(pl.concat(parts), on=["qid", "s1"], how="inner")
    if nf is not None:
        out = pl.concat([out, P.join(nf, on=["qid", "s1"], how="inner")], how="diagonal_relaxed")
    return out


def fit(df, feats):
    X = df.select(feats).to_numpy().astype(np.float32)
    ds = lgb.Dataset(X, label=df["label"].to_numpy(), feature_name=feats, free_raw_data=True)
    return lgb.train(PARAMS, ds, num_boost_round=N_ROUNDS)


def main():
    t0 = time.time()
    pool = mp.get_context("spawn").Pool(min(N_JOBS, 10))
    P = group_feats(pl.read_parquet(work_path(f"train_oof{TAG}.parquet"), columns=["qid", "s1", "p"]).rename({"p": "p1"}),
                    "train", pool)
    P = with_fold(attach_stage1_feats(P, "train", pool)).drop("_b")
    pool.close()
    feats = [c for c in P.columns if c not in NON_FEATS]
    print(f"stage-2 rows {P.height:,}, {len(feats)} feats [{time.time() - t0:.0f}s]", flush=True)
    p2 = np.zeros(P.height, dtype=np.float32)
    for f in (0, 1):
        m = fit(P.filter(pl.col("fold") == f), feats)
        m.save_model(work_path(f"lgb2_fold{f}{TAG}.txt"))
        mask = (P["fold"] != f).to_numpy()
        p2[mask] = m.predict(P.filter(pl.col("fold") != f).select(feats).to_numpy().astype(np.float32))
        print(f"stage-2 model {f} [{time.time() - t0:.0f}s]", flush=True)
    imp = sorted(zip(m.feature_importance("gain"), feats), reverse=True)
    print("top gain:", [(n, int(g)) for g, n in imp[:15]])
    oof2 = P.select("qid", "s1", "label").with_columns(pl.Series("p", p2))
    oof2.write_parquet(work_path(f"train_oof2{TAG}.parquet"))
    truth, s1_ids = eval_population()
    best = None
    for thr in np.arange(0.3, 0.92, 0.04):
        score, _ = f05_macro(assign(oof2, thr), truth, s1_ids)
        print(f"thr {thr:.2f}: F0.5 {score:.5f}", flush=True)
        if best is None or score > best[1]:
            best = (float(thr), score)
    print("best", best)
    json.dump({"threshold": best[0], "oof_f05": best[1], "feats": feats},
              open(work_path(f"model2_meta{TAG}.json"), "w"))


if __name__ == "__main__":
    main()
