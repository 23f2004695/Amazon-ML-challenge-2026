"""Final submission builder: ensemble of model versions + optional unseen-country rule.

  * Ensemble: the stage-2 probabilities of several model versions (different
    normalization, candidates and features) are averaged per (record, S1) pair; a pair
    scored by only some versions keeps the mean of those. On the withheld-S1 simulation
    the v4+v5+v6 mean scores 0.98502 vs 0.98450 for v6 alone (threshold 0.66).
  * --prior-shift: label-free test prior-shift correction for house-number "families"
    (prior_shift.py): the US test holds 3-5x more multi-record decoy businesses than train.
  * --unseen-hn-drift (leaderboard experiment, 0.971 < 0.978: rejected): for records of countries absent from the
    training data, also link the best candidate when name and street are near-identical
    (name token-set >= 90, address token-set >= 80), the house numbers differ by a small
    amount and no decoy word is present. On train such pairs are mostly decoys; the rule
    tests whether the unseen country's generator treats house-number drift as noise.
Writes output/matching_results_<name>.tsv and output/candidate_pairs_<name>.tsv.

Usage: BER_SIM=1 python make_final.py --name ens [--unseen-hn-drift]
"""
import argparse
import glob
import os

import polars as pl

import predict
from config import OUTPUT, TAG, work_path
from predict import id_maps, write_lists
from train import assign, cap_links

QOFF = 8_000_000
MEMBERS = [f"test_pred2{TAG}_tx.parquet", "v5/test_pred2_sim_tx.parquet", "v4/test_pred2_sim.parquet"]


def rescue(pred, links, lo=0.5):
    """S1 records left without any link get their best still-unlinked candidate record when its
    probability exceeds `lo` (one S1 per record, highest probability first). 5.9% of S1 end with
    no link against 5.6% that truly have no record, and an S1 without links scores 0 whenever
    it has records; below the link threshold such pairs are still right more often than not."""
    c = (pred.join(links.select("qid"), on="qid", how="anti").join(links.select("s1").unique(), on="s1", how="anti")
             .filter(pl.col("p") > lo))
    c = c.sort("p", descending=True).unique("s1", keep="first").unique("qid", keep="first")
    return pl.concat([links.select("qid", "s1", "p"), c.select("qid", "s1", "p")])


def query_country():
    return pl.concat([pl.read_parquet(work_path(f"test_s{k}.parquet"), columns=["idx", "country"])
                        .select((pl.col("idx") + off).cast(pl.UInt32).alias("qid"), "country")
                      for k, off in ((2, 0), (3, QOFF))])


def near_twin_pairs(pairs, split="test"):
    """Pairs whose house numbers differ by 1-30 (the decoy-family signature)."""
    q = pl.concat([pl.read_parquet(work_path(f"{split}_s{k}.parquet"), columns=["idx", "addr_hn"])
                     .select((pl.col("idx") + off).cast(pl.UInt32).alias("qid"), pl.col("addr_hn").cast(pl.Int64, strict=False).alias("h"))
                   for k, off in ((2, 0), (3, QOFF))])
    s = pl.read_parquet(work_path(f"{split}_s1.parquet"), columns=["idx", "addr_hn"]).select(
        pl.col("idx").cast(pl.UInt32).alias("s1"), pl.col("addr_hn").cast(pl.Int64, strict=False).alias("h1"))
    d = pairs.select("qid", "s1").join(q, on="qid").join(s, on="s1").with_columns((pl.col("h") - pl.col("h1")).abs().alias("diff"))
    return d.filter((pl.col("diff") >= 1) & (pl.col("diff") <= 30)).select("qid", "s1")


def no_raise(p_new, p_old, pairs):
    """p_new with the given pairs capped at their p_old (stage 3 may lower but not raise them);
    pairs p_old never scored are capped at 0."""
    old = pairs.select("qid", "s1").join(p_old.select("qid", "s1", pl.col("p").alias("p_old")), on=["qid", "s1"], how="left") \
               .with_columns(pl.col("p_old").fill_null(0.0))
    return (p_new.join(old, on=["qid", "s1"], how="left")
                 .with_columns(pl.min_horizontal("p", pl.col("p_old").fill_null(pl.col("p"))).cast(pl.Float32).alias("p")).drop("p_old"))


def unseen_queries():
    seen = pl.read_parquet(work_path("train_s1.parquet"), columns=["country"])["country"].unique()
    return pl.concat([pl.read_parquet(work_path(f"test_s{k}.parquet"), columns=["idx", "country"])
                        .filter(~pl.col("country").is_in(seen.implode()))
                        .select((pl.col("idx") + off).cast(pl.UInt32).alias("qid"))
                      for k, off in ((2, 0), (3, QOFF))])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--members", nargs="*", default=MEMBERS)
    ap.add_argument("--thr", type=float, default=0.66)
    ap.add_argument("--unseen-hn-drift", action="store_true")
    ap.add_argument("--prior-shift", action="store_true",
                    help="lower p of house-number 'family' pairs by the label-free test prior shift (prior_shift.py)")
    ap.add_argument("--stage3", help="stage-3 test predictions used for the records of countries seen in training")
    ap.add_argument("--fr-role-fix", action="store_true", help="role-aware unseen-country vocabulary fix (fr_role_fix.py)")
    ap.add_argument("--rescue", type=float, help="link unlinked S1s' best free record above this p (seen countries)")
    ap.add_argument("--thr-seen", type=float, help="threshold for records of countries seen in training (default --thr)")
    ap.add_argument("--blend-country", nargs="*", default=[], help="countries whose stage-3 p is blended with --blend-members")
    ap.add_argument("--blend-members", nargs="*", default=["v5/test_pred2_sim_tx.parquet", "v4/test_pred2_sim.parquet"])
    ap.add_argument("--blend-w3", type=float, default=0.5, help="stage-3 weight; the members share the rest equally")
    ap.add_argument("--fr-collision", action="store_true", help="unlink chance house-number collisions in unseen countries (fr_collision_fix.py)")
    ap.add_argument("--resid", action="store_true", help="residual stacker on the seen-country decision (resid_stack.py; additions only for India)")
    ap.add_argument("--no-raise-near", nargs="*", default=[],
                    help="countries where stage 3 may not raise p of nearby-house-number near-twins above stage 2 "
                         "(the US test holds 1.7-3.6x more such decoys than the simulation stage 3 was fit on)")
    a = ap.parse_args()
    frames = []
    for i, f in enumerate(a.members):
        frames.append(pl.read_parquet(work_path(f), columns=["qid", "s1", "p"])
                        .with_columns(pl.col("qid").cast(pl.UInt32), pl.col("s1").cast(pl.UInt32))
                        .rename({"p": f"p{i}"}))
    pred = frames[0]
    for fr in frames[1:]:
        pred = pred.join(fr, on=["qid", "s1"], how="full", coalesce=True)
    pcols = [f"p{i}" for i in range(len(frames))]
    pred = pred.select("qid", "s1", pl.mean_horizontal(pcols).cast(pl.Float32).alias("p"))
    if a.prior_shift:
        from evaluate import eval_population
        from prior_shift import apply, correction_table, reference_rates
        truth, ids = eval_population()
        ref = reference_rates(f"train_oof2{TAG}.parquet", truth, ids)
        d, t = correction_table(pred, ref)
        print(t.sort("country", "ftype").select("country", "ftype", "n_test", "pi_sim", "pi_test", "k"))
        pred = apply(pred, d, t)
    un = unseen_queries()
    if a.stage3:
        p3 = (pl.read_parquet(work_path(a.stage3), columns=["qid", "s1", "p"])
                .with_columns(pl.col("qid").cast(pl.UInt32), pl.col("s1").cast(pl.UInt32))
                .join(un, on="qid", how="anti"))
        if a.blend_country:
            # older stage-2 versions (different normalization / candidates / features); a pair
            # missing from a member takes the stage-3 p, one missing from stage 3 the members' mean
            bq = query_country().filter(pl.col("country").is_in(a.blend_country)).select("qid")
            b = p3.join(bq, on="qid", how="semi").rename({"p": "p3"})
            ms = [f"m{i}" for i in range(len(a.blend_members))]
            for m, f in zip(ms, a.blend_members):
                b = b.join(pl.read_parquet(work_path(f), columns=["qid", "s1", "p"]).rename({"p": m})
                             .with_columns(pl.col("qid").cast(pl.UInt32), pl.col("s1").cast(pl.UInt32))
                             .join(bq, on="qid", how="semi"), on=["qid", "s1"], how="full", coalesce=True)
            w = (1 - a.blend_w3) / len(ms)
            b = b.select("qid", "s1", (a.blend_w3 * pl.col("p3").fill_null(pl.mean_horizontal(ms))
                                       + sum(w * pl.col(m).fill_null(pl.col("p3")) for m in ms)).cast(pl.Float32).alias("p"))
            p3 = pl.concat([p3.join(bq, on="qid", how="anti"), b])
            print(f"stage 3 blended with {a.blend_members} (w3 {a.blend_w3}) for {a.blend_country}: {b.height:,} pairs")
        if a.no_raise_near:
            nq = query_country().filter(pl.col("country").is_in(a.no_raise_near)).select("qid")
            near = near_twin_pairs(p3.join(nq, on="qid", how="semi"))
            p3 = no_raise(p3, pred, near)
            print(f"no-raise for near-twin pairs of {a.no_raise_near}: {near.height:,} pairs")
        pred = pl.concat([pred.join(un, on="qid", how="semi"), p3])
        print(f"stage 3 for seen-country records: {p3.height:,} pairs")
    if a.fr_role_fix:
        from fr_role_fix import apply as fr_apply
        pred = fr_apply(pred)
    if a.fr_collision:
        from fr_collision_fix import apply as fc_apply
        pred = fc_apply(pred, un)
    thr_seen = a.thr_seen if a.thr_seen is not None else a.thr
    links = pl.concat([assign(pred.join(un, on="qid", how="semi"), a.thr),
                       assign(pred.join(un, on="qid", how="anti"), thr_seen)])
    n_base = links.height
    if a.rescue:
        seen_pred = pred.join(un, on="qid", how="anti")
        links = rescue(seen_pred, links, a.rescue)
        print(f"rescue > {a.rescue}: +{links.height - n_base:,} links")
    if a.resid:
        from resid_stack import test_changes
        rem, add = test_changes()
        n0 = links.height
        links = pl.concat([links.select("qid", "s1", "p").join(rem, on=["qid", "s1"], how="anti"), add.select("qid", "s1", "p")])
        print(f"residual stacker: {n0:,} -> {links.height:,} links")
    if a.unseen_hn_drift:
        un = unseen_queries()
        best = (pred.join(un, on="qid", how="semi").group_by("qid")
                    .agg(pl.col("s1").sort_by("p").last(), pl.col("p").max()))
        best = best.join(links.select("qid"), on="qid", how="anti")  # not linked yet
        cols = ["qid", "s1", "c_tset", "a_tset", "hn_eq", "hn_logdiff", "dq_max"]
        f = pl.concat([pl.read_parquet(p, columns=cols).join(best.select("qid", "s1"), on=["qid", "s1"], how="semi")
                       for p in sorted(glob.glob(work_path("test_feat/part_*.parquet")))])
        dd = pl.read_parquet(work_path("decoy_dict_unseen.parquet")).filter(pl.col("lo") > 5)["token"]
        add = (best.join(f, on=["qid", "s1"])
                   .filter((pl.col("c_tset") >= 90) & (pl.col("a_tset") >= 80) & (pl.col("hn_eq") == 0)
                           & (pl.col("hn_logdiff") <= 3.5)))  # |difference| <= ~32
        # no decoy word (label-free vocabulary) added by the record
        q = pl.concat([pl.read_parquet(work_path(f"test_s{k}.parquet"), columns=["idx", "name_full"])
                         .select((pl.col("idx") + off).cast(pl.UInt32).alias("qid"), "name_full")
                       for k, off in ((2, 0), (3, QOFF))]).join(add.select("qid"), on="qid", how="semi")
        s1n = pl.read_parquet(work_path("test_s1.parquet"), columns=["idx", "name_full"]).select(
            pl.col("idx").cast(pl.UInt32).alias("s1"), pl.col("name_full").alias("nf1"))
        add = (add.join(q, on="qid").join(s1n, on="s1")
                  .with_columns(pl.col("name_full").str.split(" ").list.unique()
                                .list.set_difference(pl.col("nf1").str.split(" ").list.unique()).alias("x"))
                  .filter(~pl.col("x").list.eval(pl.element().is_in(dd.implode())).list.any()))
        links = pl.concat([links.select("qid", "s1", "p"), add.select("qid", "s1", "p")])
        print(f"unseen-country house-number-drift rule: +{add.height:,} links")
    links = cap_links(links)
    s1, predict.QMAP = id_maps()
    write_lists(links.select("s1", "qid"), s1, "matched_entity_ids", os.path.join(OUTPUT, f"matching_results_{a.name}.tsv"))
    cand = [pl.read_parquet(work_path(f"test_pred1{TAG}_tx.parquet"), columns=["qid", "s1"])] + \
           [fr.select("qid", "s1") for fr in frames]
    if a.stage3:
        cand += [pl.read_parquet(work_path(a.stage3), columns=["qid", "s1"])]
    if a.blend_country:
        cand += [pl.read_parquet(work_path(f), columns=["qid", "s1"]).join(bq, on="qid", how="semi") for f in a.blend_members]
    write_lists(cand, s1, "candidate_entity_ids", os.path.join(OUTPUT, f"candidate_pairs_{a.name}.tsv"))
    print(f"{a.name}: ensemble of {len(frames)} ({n_base:,} links) -> {links.height:,} links after rules/cap")
