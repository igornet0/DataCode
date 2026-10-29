# Табличный бенчмарк — pandas-половина сравнения DC vs pandas.
# Сценарии и логика совпадают с bench.dc; данные готовит gen.py до замеров.
# Env: BENCH_DIR (каталог с fact.csv/dim.csv), BENCH_RUNS.
import os
import time

import pandas as pd


def report(name, times, check):
    times = sorted(times)
    n = len(times)
    print(
        f"RESULT lang=py scenario={name} runs={n} median={times[n // 2]:.6f} "
        f"min={times[0]:.6f} max={times[-1]:.6f} {check}"
    )


def timed(runs, body):
    times, out = [], None
    for _ in range(runs):
        t0 = time.perf_counter()
        out = body()
        times.append(time.perf_counter() - t0)
    return times, out


def main():
    d = os.environ["BENCH_DIR"]
    runs = int(os.environ.get("BENCH_RUNS", "5"))
    fact_path = os.path.join(d, "fact.csv")
    dim_path = os.path.join(d, "dim.csv")

    times, (t, dim) = timed(runs, lambda: (pd.read_csv(fact_path), pd.read_csv(dim_path)))
    kinds = {"i": "int", "f": "float", "O": "string"}
    schema = ",".join(f"{col}:{kinds.get(t[col].dtype.kind, t[col].dtype.name)}" for col in t.columns)
    nulls = int(t.amount.isna().sum())
    report(
        "load",
        times,
        f"rows={len(t)} nulls={nulls} schema={schema} amount_sum={t.amount.sum():.4f}",
    )

    times, f = timed(
        runs, lambda: t[(t.amount > 50) & ((t.region == "north") | (t.qty >= 5))]
    )
    report("filter", times, f"rows={len(f)} amount_sum={f.amount.sum():.4f}")

    times, (c, r) = timed(
        runs, lambda: (t.dropna(subset=["amount"]), t.fillna({"amount": 0.0}))
    )
    report("nulls", times, f"rows={len(c)} filled_sum={r.amount.sum():.4f}")

    times = []
    for _ in range(runs):
        n = t.dropna(subset=["amount"]).copy()
        t0 = time.perf_counter()
        n["total"] = n.amount * n.qty
        n["net"] = n.total - n.qty * 0.5
        times.append(time.perf_counter() - t0)
    report(
        "newcol",
        times,
        f"rows={len(n)} total_sum={n.total.sum():.4f} net_sum={n.net.sum():.4f}",
    )

    times, g = timed(
        runs,
        lambda: c.groupby("cat", sort=False).amount.agg(
            n="size", s="sum", m="mean", mn="min", mx="max"
        ),
    )
    report(
        "groupby",
        times,
        f"rows={len(g)} n_sum={g.n.sum()} s_sum={g.s.sum():.4f} m_sum={g.m.sum():.4f} "
        f"mn_sum={g.mn.sum():.4f} mx_sum={g.mx.sum():.4f}",
    )

    times, s = timed(runs, lambda: c.sort_values("amount", ascending=False).head(100))
    report(
        "sort_top",
        times,
        f"rows={len(s)} amount_sum={s.amount.sum():.4f} first={s.amount.iloc[0]:.4f}",
    )

    times, j = timed(runs, lambda: t.merge(dim, on="cat", how="inner"))
    report("join", times, f"rows={len(j)} weight_sum={j.weight.sum():.4f}")

    def pipeline():
        pt = pd.read_csv(fact_path)
        pdim = pd.read_csv(dim_path)
        pc = pt.dropna(subset=["amount"])
        pf = pc[(pc.amount > 20) & (pc.qty >= 3)].copy()
        pf["total"] = pf.amount * pf.qty
        pj = pf.merge(pdim, on="cat", how="inner")
        pj["wt"] = pj.total * pj.weight
        return pj.groupby("region", sort=False).wt.agg(n="size", s="sum")

    times, pg = timed(runs, pipeline)
    report("pipeline", times, f"rows={len(pg)} n_sum={pg.n.sum()} s_sum={pg.s.sum():.4f}")


if __name__ == "__main__":
    main()
