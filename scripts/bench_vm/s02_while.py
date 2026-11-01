import time
def main():
    t0 = time.perf_counter()
    i = 0; s = 0
    while i < 10000000:
        s += i & 7
        i += 1
    t1 = time.perf_counter()
    print(f"RESULT s02 {s} {t1-t0:.4f}")
main()
