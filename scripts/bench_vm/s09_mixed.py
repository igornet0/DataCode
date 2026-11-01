import time
def collatz_len(n):
    steps = 0
    while n != 1:
        if n % 2 == 0: n = n // 2
        else: n = 3 * n + 1
        steps += 1
    return steps
def main():
    t0 = time.perf_counter()
    best = 0; best_n = 0; hist = {}
    for n in range(1, 150000):
        l = collatz_len(n); b = l // 10
        hist[b] = hist.get(b, 0) + 1
        if l > best: best = l; best_n = n
    t1 = time.perf_counter()
    print(f"RESULT s09 {best_n} {best} {len(hist)} {t1-t0:.4f}")
main()
