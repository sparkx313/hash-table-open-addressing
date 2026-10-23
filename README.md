# hash-table-open-addressing

Four open-addressing hash maps in pure Python — linear probing, quadratic
(triangular) probing, double hashing, and Robin Hood — instrumented so the probe
cost is a number you can read off the table instead of a paragraph about clustering.

No dependencies. Standard library only, Python 3.9+ (measured on CPython 3.11.9,
Windows x64).

```python
from open_hashing import OpenAddressMap, DoubleProbe, RobinHoodMap

m = OpenAddressMap(DoubleProbe, capacity=1024, max_load=0.66)
m["ada"] = 1
"ada" in m                    # True
del m["ada"]                  # leaves a tombstone, not a hole
m.stats()                     # {'size': 0, 'capacity': 1024, 'tombstones': 1, ...}

rh = RobinHoodMap(capacity=1024)
```

The dict-like surface is real, not a demo: `[]`, `get`, `in`, `del`, `pop(key,
default)`, `setdefault`, `update`, `len`, `iter`, `keys`, `values`, `items`,
`repr`.

## Why the probe sequence is the whole design

Open addressing keeps entries in the slot array itself — no per-bucket lists, good
cache shape, terrible deletion story. A slot cannot simply be emptied on delete,
because *the empty slot is what terminates a probe walk*: clearing the middle of a
cluster strands everything behind it. So a delete writes a tombstone, probes walk
over tombstones, and an insert may recycle one.

That single constraint drives every other decision here:

- `used` (ever-occupied slots) and `size` (live entries) are tracked separately.
- Two different rebuilds, not one: **grow** when live load is the problem,
  **sweep** (same-capacity rehash) when dead slots are the problem. Growing to fix
  tombstones only postpones the same walk.
  ```python
  def _should_grow(self):   return (self.size + 1) / self.capacity > self.max_load
  def _should_sweep(self):  return self.used - self.size > self.capacity // 4
  ```
- Capacity must be a power of two, because the probe guarantees below depend on it.

## Measured behaviour

`python benchmarks/run_benchmarks.py` reproduces every number below.

### Probe length as the table fills

65,536 slots, uniform random `int` keys, growth suppressed (`max_load=0.98`) so the
measured load is the one asked about and not the policy's choice. The benchmark
asserts the capacity afterwards — a row measured on a resized table is a lie.

```
 strategy  load  hit mean  hit max  hit sd  miss mean
    linear  0.25      1.17        7    0.50       1.38
    linear  0.50      1.50       21    1.23       2.57
    linear  0.75      2.53      234    4.71       9.09
    linear  0.90      5.69      429  17.70      50.21
 quadratic  0.25      1.16        6    0.47       1.38
 quadratic  0.50      1.44       16    0.98       2.12
 quadratic  0.75      1.99       32    1.98       4.59
 quadratic  0.90      2.88       75    4.06      12.80
    double  0.25      1.15        6    0.44       1.33
    double  0.50      1.38       12    0.83       2.02
    double  0.75      1.84       28    1.63       4.08
    double  0.90      2.58       70    3.36      10.13
robin-hood  0.25      1.17        5    0.43       1.41
robin-hood  0.50      1.50        8    0.81       2.55
robin-hood  0.75      2.49       16    1.83       8.66
robin-hood  0.90      5.49       34    4.60      47.63
```

The classic closed forms for linear probing under uniform hashing are
`hit = ½(1 + 1/(1−α))` and `miss = ½(1 + 1/(1−α)²)`. Against the `linear` rows:

| load | hit predicted | hit measured | miss predicted | miss measured |
|---|---|---|---|---|
| 0.25 | 1.17 | 1.17 | 1.38 | 1.38 |
| 0.50 | 1.50 | 1.50 | 2.50 | 2.57 |
| 0.75 | 2.50 | 2.53 | 8.50 | 9.09 |
| 0.90 | 5.50 | 5.69 | 50.50 | 50.21 |

So the instrumentation is measuring what it says it measures. Given that, the
comparison between strategies is meaningful:

- **Quadratic** beats linear on the *tail* (max 75 vs 429 at 0.90) because it breaks
  primary clustering, but it is not a free win — mean 2.88 vs 5.69 is a real gap and
  secondary clustering remains.
- **Double hashing** is the best column in the table at every load, which is what the
  theory promises when the step is coprime with the size.
- **Robin Hood** at 0.90 has essentially linear's mean (5.49 vs 5.69) with a *thirteenth*
  of the worst case (max 34 vs 429) and a quarter of the standard deviation. That is
  the entire point of the algorithm, and it is the most interesting row here: it
  doesn't improve the average, it caps the tail.

### The adversarial case

Same table, but every one of 2,000 keys is built to hash to the *same* home slot
(`i * 65536`, and `int` hashes to itself, so all homes collide at 0).

```
 strategy    mean    max    ops
    linear 1000.50   2000   2000
 quadratic 1000.50   2000   2000
    double    2.80      4   2000
robin-hood 1000.50   2000   2000
```

Linear, quadratic and Robin Hood all degrade to "walk the whole cluster": 1000.5
probes per lookup is exactly the average length of a 2,000-long run. Double hashing
is unaffected (2.80 probes) because its step is derived from the key, so colliding
homes do not imply colliding steps. Robin Hood — whose whole selling point above is
tail control — gets nothing here, because the invariant it maintains is about keys
that *share* a home slot; it can reorder a cluster, it cannot shrink one.

One implementation note that is a real bug I hit: `DoubleProbe`'s step must be odd.
With an even step and a power-of-two size the walk closes a cycle over half the
table and never reaches an empty slot, which looked like an infinite loop on a
near-full table. `self.step = step | 1` in `open_hashing/probes.py:54` and
`test_every_probe_is_a_full_cycle` are the fix and the guard.

### Throughput against the built-in dict

200,000 `str` keys, best of 3 runs.

```
          strategy    insert       hit      miss   hit vs dict
  dict (reference)     0.071     0.009     0.010     reference
            linear     2.000     0.070     0.067          7.5x
         quadratic     1.141     0.065     0.077          7.0x
            double     1.477     0.080     0.100          8.6x
        robin-hood     1.325     0.064     0.066          6.8x
```

CPython's dict is C with a tuned hash seed and a specialised lookup; this is Python
doing a `while` loop per probe. Lookups land 7–9× behind and inserts ~16–28× behind.
Use it to understand open addressing, not to replace `dict`.

### Resize cost

50,000 keys, best of 3 runs per starting capacity:

```
  start capacity     8: 0.279s, 14 resizes, final capacity 131072, mean probes 2.569
  start capacity  1024: 0.316s, 7 resizes, final capacity 131072, mean probes 2.557
```

Halving the resize count buys almost nothing, because doubling reinserts
*everything* on the last resize: the total reinsert work is dominated by the final
capacity in both cases (≈86,000 entries either way), so the 14-resize run is within
noise of the 7-resize one. Pre-sizing helps mean probe length slightly (2.557 vs
2.569); it does not meaningfully help time.

### Deletion churn

60,000 interleaved inserts and deletes over a 10⁶ key space so homes spread over the
whole table, starting capacity 4,096:

```
    linear: 0.36s, size 56468, capacity 131072, tombstones 658, resizes 5, mean probes 2.387, max probe 71
 quadratic: 0.33s, size 56654, capacity 131072, tombstones 662, resizes 5, mean probes 2.021, max probe 26
    double: 0.39s, size 56484, capacity 131072, tombstones 719, resizes 5, mean probes 1.905, max probe 24
robin-hood: 0.39s, size 56476, capacity 131072, tombstones 684, resizes 5, mean probes 2.355, max probe 60
```

Tombstones stay at ~1% of capacity in all four, which is the sweep policy doing its
job: without the used-vs-size quota the same workload drifts into tables where most
of a probe walk is dead slots.

## Tests

```
python -m unittest discover -s tests -p "test_*.py" -t tests

Ran 21 tests in 0.2s

OK
```

Coverage worth naming, because these are the tests that found the bugs:

- `test_every_probe_is_a_full_cycle` — every strategy must reach all `capacity`
  slots from a full walk (catches the even-step double-hashing hang).
- `test_tombstone_at_home_is_reused` — keys 0, 64 and 128 all hash to home slot 0 at
  capacity 64, so deleting the middle must let the next one land there.
- `test_tombstone_quota_sweeps_without_growing` — the rebuild must be same-capacity.
- `test_robin_hood_keeps_no_key_poorer_than_its_neighbour` — the distance invariant,
  asserted on the final table rather than trusted from the insert code.
- `test_linear_probes_longer_than_double_on_blocks` — 200 keys spaced one capacity
  apart give linear a longest probe of exactly 200 and double hashing a short one.
- `test_holds_every_key_type` — ints, `str`, `bytes`, tuples, `frozenset`, `None`,
  `bool` and `float` keys. `1` and `True` are the same key in Python, so a reference
  `dict` decides that rather than the table inventing an answer.
- `test_random_operations_mirror_dict` — 4,000 interleaved set/get/delete/member
  ops per strategy over a 400-key space, compared against a real `dict` at every
  step and key-for-key at the end.

## Layout

```
open_hashing/probes.py        LinearProbe, QuadProbe, DoubleProbe
open_hashing/table.py         OpenAddressMap, RobinHoodMap, sizing policy, stats
tests/test_table.py           21 tests, including a dict-mirroring random op fuzz
benchmarks/run_benchmarks.py  everything in the measured section
```

## Limits, stated

- Insert order is *not* preserved on iteration; iteration is slot order, tombstones
  skipped.
- The table never shrinks. A map that peaked at 131,072 slots keeps them.
- Not thread-safe, and no plan to be.
- `hash()` is per-process randomised for `str`, so the wall-clock rows move between
  runs; the probe-length sweep uses `int` keys so it does not.
- `max_load` is honoured on insert only; deleting everything does not release capacity.
