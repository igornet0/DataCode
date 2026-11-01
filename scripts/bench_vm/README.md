# VM micro-benchmarks (Stage 24/25)

Сценарии изначально написаны во время «Этап 24 — VM Performance Audit»
(см. `OPTIMIZATION_PROGRESS.md`), перенесены сюда без изменений на этапе 25.
Каждый `.dc` имеет эквивалент `.py`; строка `RESULT sNN <checksum> <inner_time>`
должна совпадать у обеих сторон (кроме времени).

| файл | что нагружает |
|---|---|
| s01_arith | int/float арифметика, Mod/Div в цикле |
| s02_while | while + bitwise `&` (утечка `bitwise::push_int`) |
| s03_nested | вложенные for-range |
| s04_calls | вызовы пользовательских функций |
| s05_argkinds | default / keyword / *args / **kwargs |
| s06_globals | LoadGlobal/StoreGlobal |
| s07_strings | конкатенация/интерполяция строк |
| s08_arrdict | массивы и словари |
| s09_mixed | смесь всего (сравнения, арифметика, вызовы) |
| s10_astar | A* (см. также `scripts/bench_astar`) |
| big_s04 / big_s09 / rs | увеличенные варианты для sample/malloc_history |

`baseline_stage24/` — исходные сырые измерения этапа 24 (Apple M4, macOS,
release build HEAD=918fe4d): `raw.tsv` (run, scenario, lang, real, user, sys,
max RSS bytes, instructions | RESULT), `prof_*.txt` — счётчики опкодов
(`--features profile`, fast path отключён), `tm.txt` — служебный тайминг.

Медианы этапа 24 (DC wall / RSS / Ginstr; Python wall):

| сценарий | DataCode | Python |
|---|---|---|
| s01 | 0.87 s / 1008 MB / 19.7 | 0.49 s |
| s02 | 0.74 s / 1112 MB / 18.95 | 0.83 s |
| s03 | 0.56 s / 488 MB / 12.3 | 0.29 s |
| s04 | 1.02 s / 369 MB / 21.5 | 0.12 s |
| s05 | 1.79 s / 946 MB / 37.9 | 0.23 s |
| s06 | 0.37 s / 70 MB | 0.32 s |
| s07 | 0.36 s / 312 MB | 0.12 s |
| s08 | 0.49 s / 477 MB | 0.19 s |
| s09 | 4.60 s / 3485 MB / 73.3 | 1.41 s |
| s10 | 0.17 s / 110 MB | 0.31 s |

Запуск: `BASE=/path/to/old/datacode ./scripts/bench_vm/compare.sh 3`.
