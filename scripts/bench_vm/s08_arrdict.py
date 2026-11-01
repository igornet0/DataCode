import time
def main():
    n = 500000
    t0 = time.perf_counter()
    arr = []
    for i in range(n): arr.append(i * 2)
    s = 0
    for i in range(n): s += arr[i]
    t1 = time.perf_counter()
    d = {}
    for i in range(n): d[i] = i + 1
    for i in range(n): s += d[i]
    t2 = time.perf_counter()
    print(f"RESULT s08 {s} array={t1-t0:.4f} dict={t2-t1:.4f}")
main()
