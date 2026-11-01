import time
g_counter = 0; g_step = 3
t0 = time.perf_counter()
for i in range(2000000):
    g_counter = (g_counter + g_step) % 1000003
t1 = time.perf_counter()
def local_version():
    c = 0; st = 3
    for i in range(2000000):
        c = (c + st) % 1000003
    return c
t2 = time.perf_counter()
r = local_version()
t3 = time.perf_counter()
print(f"RESULT s06 {g_counter} {r} global={t1-t0:.4f} local={t3-t2:.4f}")
