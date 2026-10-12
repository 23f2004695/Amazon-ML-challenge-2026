"""Diagnostic: why do true pairs never reach the candidate set?"""
import polars as pl
from rapidfuzz import fuzz

from blocking import CAP, CAP_ADDR, make_keys
from config import work_path
from evaluate import load_truth

QOFF = 8_000_000

if __name__ == "__main__":
    truth = load_truth().with_columns(pl.col("qid").cast(pl.UInt32), pl.col("s1_true").cast(pl.UInt32))
    cand = pl.scan_parquet(work_path("train_cand.parquet")).select("qid", "s1", "brank")
    miss = (truth.lazy().join(cand, left_on=["qid", "s1_true"], right_on=["qid", "s1"], how="anti")
                 .collect(engine="streaming"))
    has_c = pl.scan_parquet(work_path("train_cand.parquet")).select("qid").unique().collect(engine="streaming")
    miss = miss.with_columns(pl.col("qid").is_in(has_c["qid"].implode()).alias("q_has_cands"))
    print(f"missed true pairs: {miss.height:,} of {truth.height:,} ({miss.height/truth.height:.4f})")
    print(miss.group_by("q_has_cands").len())
    cols = ["idx", "country", "name_raw", "addr_raw", "name_core", "name_full", "addr_toks", "addr_nums", "addr_hn"]
    q = pl.concat([pl.read_parquet(work_path(f"train_s{k}.parquet"), columns=cols)
                     .with_columns((pl.col("idx") + off).cast(pl.UInt32).alias("qid")).drop("idx")
                   for k, off in ((2, 0), (3, QOFF))])
    s1 = pl.read_parquet(work_path("train_s1.parquet"), columns=cols).with_columns(pl.col("idx").cast(pl.UInt32))
    # --- sibling recoverability: another record of the same S1 that IS found and shares text with q
    found = (truth.lazy().join(cand.filter(pl.col("brank") < 10), left_on=["qid", "s1_true"],
                               right_on=["qid", "s1"], how="semi").collect(engine="streaming"))
    qa = q.select("qid", pl.concat_str(["addr_toks", "addr_nums"], separator="|").alias("a"), "name_core", "name_full")
    f = found.join(qa, on="qid").rename({"qid": "sib"})
    m = miss.join(qa, on="qid")
    for col, lab in (("a", "same normalized address"), ("name_core", "same core name"), ("name_full", "same full name")):
        ok = (m.filter(pl.col(col).str.len_chars() > 1)
               .join(f.select("s1_true", col), on=["s1_true", col], how="semi"))
        print(f"  recoverable via a found sibling with {lab}: {ok.height:,} ({ok.height/miss.height:.3f})")
    anyrec = pl.concat([m.filter(pl.col(c).str.len_chars() > 1).join(f.select("s1_true", c), on=["s1_true", c], how="semi").select("qid")
                        for c in ("a", "name_core", "name_full")]).unique()
    print(f"  recoverable via any of these: {anyrec.height:,} ({anyrec.height/miss.height:.3f})")
    # --- text similarity + shared keys for a sample
    smp = miss.sample(30000, seed=1).join(q, on="qid").join(s1, left_on="s1_true", right_on="idx", suffix="_1")
    smp = smp.with_columns(
        pl.struct("name_core", "name_core_1").map_elements(lambda r: fuzz.token_set_ratio(r["name_core"], r["name_core_1"]), return_dtype=pl.Float64).alias("name_sim"),
        pl.struct("addr_toks", "addr_toks_1").map_elements(lambda r: fuzz.token_set_ratio(r["addr_toks"], r["addr_toks_1"]) if r["addr_toks"] and r["addr_toks_1"] else -1.0, return_dtype=pl.Float64).alias("addr_sim"))
    b = lambda c, lo, hi: pl.when(pl.col(c) < 0).then(pl.lit("empty")).when(pl.col(c) >= hi).then(pl.lit(f">={hi}")).when(pl.col(c) >= lo).then(pl.lit(f"{lo}-{hi}")).otherwise(pl.lit(f"<{lo}"))
    print(smp.with_columns(b("name_sim", 50, 80).alias("name"), b("addr_sim", 50, 80).alias("addr"))
             .group_by("name", "addr").len().sort("len", descending=True).head(12))
    # shared keys with df info
    kq = make_keys(smp.select(pl.col("qid").alias("id"), "country", "name_core", "addr_toks", "addr_nums", "addr_hn"), "id")
    ks = make_keys(smp.select(pl.col("qid").alias("id"), "country", pl.col("name_core_1").alias("name_core"),
                               pl.col("addr_toks_1").alias("addr_toks"), pl.col("addr_nums_1").alias("addr_nums"), pl.col("addr_hn_1").alias("addr_hn")), "id")
    shared = kq.join(ks.select("id", "key"), on=["id", "key"])
    need = shared.select("key").unique()
    dfs = []
    for i in range(0, s1.height, 200_000):
        ch = s1.slice(i, 200_000).select(pl.col("idx").alias("id"), "country", "name_core", "addr_toks", "addr_nums", "addr_hn")
        dfs.append(make_keys(ch, "id").join(need, on="key", how="semi").group_by("key").len())
    df = pl.concat(dfs).group_by("key").agg(pl.col("len").sum().alias("df"))
    shared = shared.join(df, on="key", how="left").with_columns(
        pl.when(pl.col("typ") == 3).then(pl.col("df") <= CAP_ADDR).otherwise(pl.col("df") <= CAP).alias("kept"))
    per = shared.group_by("id").agg(pl.len().alias("n_shared"), pl.col("kept").sum().alias("n_kept"))
    per = smp.select(pl.col("qid").alias("id")).join(per, on="id", how="left").fill_null(0)
    print("sample: shares no key at all:", (per["n_shared"] == 0).mean().__round__(3),
          "| shares keys but all over the df cap:", ((per["n_shared"] > 0) & (per["n_kept"] == 0)).mean().__round__(3),
          "| has kept shared keys (ranked out of top-15):", (per["n_kept"] > 0).mean().__round__(3))
    print(shared.filter(~pl.col("kept")).group_by("typ").len().sort("typ"))
    ex = smp.sample(20, seed=3)
    for r in ex.iter_rows(named=True):
        print(f"Q : {r['name_raw']} | {r['addr_raw']}\nS1: {r['name_raw_1']} | {r['addr_raw_1']}   (name {r['name_sim']:.0f} addr {r['addr_sim']:.0f})")
