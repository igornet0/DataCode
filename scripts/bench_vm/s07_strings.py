import time
def main():
    t0 = time.perf_counter()
    parts = []; total = 0
    for i in range(300000):
        s = "item_" + str(i)
        total += len(s)
        if i % 10 == 0:
            parts.append(s)
    joined = ",".join(parts)
    t1 = time.perf_counter()
    print(f"RESULT s07 {total} {len(joined)} {t1-t0:.4f}")
main()
