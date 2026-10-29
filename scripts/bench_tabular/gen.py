# Генератор данных для табличного бенчмарка DC vs pandas.
# Запускается до замеров; обе стороны читают одни и те же CSV.
#
#   python3 gen.py ROWS OUT_DIR [SEED]
#
# fact.csv: id,cust,cat,region,amount,qty  (amount пустой у ~5% строк)
# dim.csv:  cat,cat_name,weight            (20 строк, ключ cat)
import os
import sys

import numpy as np

REGIONS = np.array(["north", "south", "east", "west", "center"])
N_CATS = 20


def main():
    rows = int(sys.argv[1])
    out = sys.argv[2]
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 42
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(seed)

    cats = np.array([f"c{i:02d}" for i in range(N_CATS)])
    ids = np.arange(rows)
    cust = rng.integers(0, max(rows // 10, 1), rows)
    cat = cats[rng.integers(0, N_CATS, rows)]
    region = REGIONS[rng.integers(0, len(REGIONS), rows)]
    # Два знака после запятой: значения точно представимы в CSV у обеих сторон.
    amount = np.round(rng.uniform(0.0, 100.0, rows), 2)
    null_mask = rng.random(rows) < 0.05
    qty = rng.integers(1, 11, rows)

    amount_s = np.char.mod("%.2f", amount)
    amount_s[null_mask] = ""
    cols = [ids.astype(str), cust.astype(str), cat, region, amount_s, qty.astype(str)]
    with open(os.path.join(out, "fact.csv"), "w") as f:
        f.write("id,cust,cat,region,amount,qty\n")
        lines = np.char.add(cols[0], ",")
        for c in cols[1:-1]:
            lines = np.char.add(np.char.add(lines, c), ",")
        lines = np.char.add(lines, cols[-1])
        f.write("\n".join(lines.tolist()))
        f.write("\n")

    with open(os.path.join(out, "dim.csv"), "w") as f:
        f.write("cat,cat_name,weight\n")
        for i, c in enumerate(cats):
            f.write(f"{c},name_{i:02d},{(i % 7) + 1}.5\n")


if __name__ == "__main__":
    main()
