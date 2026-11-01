import time
def add1(a, b):
    return a + b
def main():
    t0 = time.perf_counter()
    s = 0
    for i in range(1000000):
        s = add1(s, i)
    t1 = time.perf_counter()
    print(f"RESULT s04 {s} {t1-t0:.4f}")
main()
