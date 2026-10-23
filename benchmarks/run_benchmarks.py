"""Measure what an open-addressing table actually costs, next to CPython's dict.

Two numbers matter for this design:

1. probe length as the table fills - that is the algorithmic cost open
   addressing pays for not having buckets, and it is what the strategies differ
   on
2. wall clock against the built-in dict - the dict is C with a tuned SipHash
   key hash, so a Python implementation loses by a wide margin and the honest
   number is the ratio, not a claim of competitiveness
"""

import os
import random
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from open_hashing import PROBE_STRATEGIES, LinearProbe, OpenAddressMap, RobinHoodMap

CAPACITY = 2**16
# The tables in the fill-level sweep must not resize, or the measured load
# factor is the policy's choice and not the one being asked about.
NO_RESIZE = 0.98


def hit_distances(table, keys):
    """Probe counts a lookup pays, measured straight off the search routine."""
    return [table._find(key)[2] for key in keys]


def probe_table():
    print("Probe length by fill level (uniform random int keys, capacity 65536)")
    print("  hits = keys that are in the table, misses = keys that never were")
    header = (
        f"{'strategy':>11} {'load':>5} {'hit mean':>9} {'hit max':>8} "
        f"{'hit sd':>7} {'miss mean':>10}"
    )
    print(header)
    for name, factory in PROBE_STRATEGIES.items():
        for fraction in (0.25, 0.5, 0.75, 0.9):
            n = int(CAPACITY * fraction)
            keys = [random.randrange(10**9) for _ in range(n)]
            table = factory(capacity=CAPACITY, max_load=NO_RESIZE)
            for key in keys:
                table[key] = key
            # The row only means what it says if the table is still the size it
            # was built at; a resize would move the measurement to another load.
            if table.capacity != CAPACITY or table.resizes:
                raise AssertionError(
                    f"{name} at {fraction}: capacity {table.capacity}, "
                    f"{table.resizes} resizes"
                )
            hit = hit_distances(table, keys)
            misses = [key for key in (random.randrange(10**9) for _ in range(4000)) if key not in table]
            miss = hit_distances(table, misses[:2000])
            print(
                f"{name:>11} {table.load_factor:>5.2f} {statistics.fmean(hit):>9.2f} "
                f"{max(hit):>8} {statistics.pstdev(hit):>7.2f} {statistics.fmean(miss):>10.2f}"
            )


def adversarial_table():
    print("Probe length when every key hashes to the same home slot")
    keys = [i * CAPACITY for i in range(2000)]
    print(f"{'strategy':>11} {'mean':>7} {'max':>6} {'ops':>6}")
    for name, factory in PROBE_STRATEGIES.items():
        table = factory(capacity=CAPACITY)
        for key in keys:
            table[key] = key
        table.probes = table.ops = table.longest_probe = 0
        for key in keys:
            key in table
        print(
            f"{name:>11} {table.probes / table.ops:>7.2f} "
            f"{table.longest_probe:>6} {table.ops:>6}"
        )


def throughput(repeat=3):
    print(f"Wall clock, 200k string keys, best of {repeat} runs "
          f"(CPython dict is C, this is Python)")
    keys = [f"key-{i}" for i in range(200_000)]
    hits = keys[:50_000]
    misses = [f"absent-{i}" for i in range(50_000)]

    def measure(build, lookups):
        best = None
        for _ in range(repeat):
            t0 = time.perf_counter()
            container = build()
            build_time = time.perf_counter() - t0
            t0 = time.perf_counter()
            for key in lookups:
                key in container
            look_time = time.perf_counter() - t0
            if best is None or look_time < best[1]:
                best = (build_time, look_time)
        return best

    build_ref, hit_ref = measure(lambda: dict.fromkeys(keys, keys), hits)
    _, miss_ref = measure(lambda: dict.fromkeys(keys, keys), misses)
    rows = [("dict (reference)", build_ref, hit_ref, miss_ref)]
    for name, factory in PROBE_STRATEGIES.items():
        def build(f=factory):
            table = f(capacity=2**16)
            for key in keys:
                table[key] = key
            return table
        build_time, hit_time = measure(build, hits)
        _, miss_time = measure(build, misses)
        rows.append((name, build_time, hit_time, miss_time))

    print(f"{'strategy':>18} {'insert':>9} {'hit':>9} {'miss':>9} {'hit vs dict':>13}")
    for name, build, hit, miss in rows:
        ratio = "reference" if name.startswith("dict") else f"{hit / hit_ref:.1f}x"
        print(f"{name:>18} {build:>9.3f} {hit:>9.3f} {miss:>9.3f} {ratio:>13}")


def resize_cost(repeat=3):
    print(f"Resize behaviour from two starting capacities (50k keys, best of {repeat})")
    print("  doubling reinserts almost everything on the last resize, so the two")
    print("  starts differ by far less than their resize counts suggest")
    rng = random.Random(5)
    keys = [rng.randrange(10**7) for _ in range(50_000)]
    for start in (8, 1024):
        best, chosen = None, None
        for _ in range(repeat):
            table = OpenAddressMap(LinearProbe, capacity=max(8, start))
            t0 = time.perf_counter()
            for key in keys:
                table[key] = key
            elapsed = time.perf_counter() - t0
            if best is None or elapsed < best:
                best, chosen = elapsed, table.stats()
        print(
            f"  start capacity {start:>5}: {best:.3f}s, {chosen['resizes']} resizes, "
            f"final capacity {chosen['capacity']}, mean probes {chosen['mean_probes']}"
        )


def churn():
    print("Delete/insert churn: tombstone pressure against the used-slot rehash")
    print("  60k operations, key space 1e6 so homes spread over the whole table")
    rng = random.Random(9)
    for strategy in ("linear", "quadratic", "double", "robin-hood"):
        table = PROBE_STRATEGIES[strategy](capacity=4096)
        live = set()
        t0 = time.perf_counter()
        for _ in range(60_000):
            key = rng.randrange(10**6)
            if key in live:
                live.discard(key)
                table.pop(key, None)
            else:
                live.add(key)
                table[key] = True
        elapsed = time.perf_counter() - t0
        stats = table.stats()
        assert stats["size"] == len(live), (stats, len(live))
        print(
            f"  {strategy:>10}: {elapsed:.2f}s, size {stats['size']}, "
            f"capacity {stats['capacity']}, tombstones {stats['tombstones']}, "
            f"resizes {stats['resizes']}, mean probes {stats['mean_probes']}, "
            f"max probe {stats['longest_probe']}"
        )


def main():
    random.seed(1)
    probe_table()
    print()
    adversarial_table()
    print()
    throughput()
    print()
    resize_cost()
    print()
    churn()


if __name__ == "__main__":
    main()
