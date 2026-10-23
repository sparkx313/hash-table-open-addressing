import os
import random
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from open_hashing import (
    DoubleProbe,
    LinearProbe,
    OpenAddressMap,
    PROBE_STRATEGIES,
    QuadProbe,
    RobinHoodMap,
)


def all_strategies():
    return [(name, factory()) for name, factory in PROBE_STRATEGIES.items()]


class TestBasics(unittest.TestCase):
    def test_roundtrip_and_miss(self):
        for name, table in all_strategies():
            with self.subTest(strategy=name):
                table["alpha"] = 1
                table["beta"] = 2
                self.assertEqual(table["alpha"], 1)
                self.assertEqual(table.get("gamma"), None)
                self.assertEqual(table.get("gamma", "sentinel"), "sentinel")
                self.assertIn("beta", table)
                self.assertNotIn("gamma", table)
                self.assertEqual(len(table), 2)

    def test_overwrite_does_not_duplicate(self):
        for name, table in all_strategies():
            with self.subTest(strategy=name):
                for value in range(5):
                    table["k"] = value
                self.assertEqual(len(table), 1)
                self.assertEqual(table["k"], 4)

    def test_delete_then_lookup(self):
        for name, table in all_strategies():
            with self.subTest(strategy=name):
                for i in range(20):
                    table[i] = i * i
                for i in range(0, 20, 2):
                    del table[i]
                for i in range(20):
                    if i % 2 == 0:
                        self.assertNotIn(i, table)
                    else:
                        self.assertEqual(table[i], i * i)
                self.assertEqual(len(table), 10)

    def test_missing_key_raises(self):
        for name, table in all_strategies():
            with self.subTest(strategy=name):
                with self.assertRaises(KeyError):
                    del table["nope"]
                with self.assertRaises(KeyError):
                    table["nope"]
                with self.assertRaises(KeyError):
                    table.pop("nope")
                self.assertEqual(table.pop("nope", 7), 7)

    def test_pop_and_setdefault_and_update(self):
        for name, table in all_strategies():
            with self.subTest(strategy=name):
                table.update({"a": 1, "b": 2})
                self.assertEqual(table.pop("a"), 1)
                self.assertEqual(len(table), 1)
                self.assertEqual(table.setdefault("c", 3), 3)
                self.assertEqual(table.setdefault("c", 99), 3)
                self.assertEqual(sorted(table.keys()), ["b", "c"])
                self.assertEqual(sorted(table.values()), [2, 3])
                self.assertEqual(sorted(table.items()), [("b", 2), ("c", 3)])

    def test_holds_every_key_type(self):
        # 1 and True are the same key in Python, so the reference dict has to
        # decide that too rather than the table being asked to invent an answer.
        keys = [0, -1, 1, 2, 3, "a", b"bytes", (1, 2), frozenset({1}), None, True, 3.5]
        ref = {}
        for i, key in enumerate(keys):
            ref[key] = i
        for name, table in all_strategies():
            with self.subTest(strategy=name):
                for i, key in enumerate(keys):
                    table[key] = i
                for key, value in ref.items():
                    self.assertEqual(table[key], value)
                self.assertEqual(len(table), len(ref))


class TestAgainstDict(unittest.TestCase):
    def test_random_operations_mirror_dict(self):
        rng = random.Random(7)
        for name, table in all_strategies():
            with self.subTest(strategy=name):
                ref = {}
                for _ in range(4000):
                    key = rng.randint(0, 400)
                    action = rng.random()
                    if action < 0.55:
                        value = rng.randint(0, 999)
                        table[key] = value
                        ref[key] = value
                    elif action < 0.75:
                        if key in ref:
                            del table[key]
                            del ref[key]
                    elif action < 0.9:
                        self.assertEqual(table.get(key), ref.get(key))
                    else:
                        self.assertEqual((key in table), (key in ref))
                self.assertEqual(sorted(table.keys()), sorted(ref.keys()))
                self.assertEqual(
                    sorted(table.items(), key=lambda kv: str(kv[0])),
                    sorted(ref.items(), key=lambda kv: str(kv[0])),
                )


class TestSizing(unittest.TestCase):
    def test_capacity_stays_power_of_two_and_bounded(self):
        table = OpenAddressMap(LinearProbe)
        for i in range(5000):
            table[i] = i
            self.assertEqual(table.capacity & (table.capacity - 1), 0)
            self.assertLessEqual(table.load_factor, 0.67)

    def test_tombstone_quota_sweeps_without_growing(self):
        table = OpenAddressMap(LinearProbe, capacity=64)
        for i in range(42):
            table[i] = i
        self.assertEqual(table.capacity, 64)
        for i in range(32):
            del table[i]
        self.assertEqual(table.stats()["tombstones"], 32)
        table["trigger"] = 1
        self.assertEqual(table.capacity, 64)
        self.assertEqual(table.sweeps, 1)
        self.assertEqual(table.stats()["tombstones"], 0)
        self.assertEqual(len(table), 11)
        for i in range(32, 42):
            self.assertEqual(table[i], i)

    def test_tombstone_at_home_is_reused(self):
        table = OpenAddressMap(LinearProbe, capacity=64)
        # 0, 64 and 128 all hash to home slot 0.
        table[0] = "a"
        table[64] = "b"
        del table[0]
        table[128] = "c"
        self.assertEqual(table.capacity, 64)
        self.assertEqual(table.used, 2)
        self.assertEqual(table.slots[0], (128, "c"))
        self.assertEqual(table[64], "b")
        self.assertEqual(table[128], "c")

    def test_rejects_bad_construction(self):
        for capacity in (4, 10, 12):
            with self.assertRaises(ValueError):
                OpenAddressMap(LinearProbe, capacity=capacity)
        with self.assertRaises(ValueError):
            OpenAddressMap(LinearProbe, max_load=1.0)

    def test_rehash_preserves_content(self):
        table = OpenAddressMap(DoubleProbe, capacity=8)
        for i in range(1000):
            table[i] = str(i)
        self.assertEqual(table.resizes, 8)
        self.assertEqual([table[i] for i in range(1000)], [str(i) for i in range(1000)])


class TestProbeBehaviour(unittest.TestCase):
    def test_every_probe_is_a_full_cycle(self):
        """An insert walks until it finds an empty slot, so a probe sequence
        that skips positions can hang instead of failing."""
        for probe_class in (LinearProbe, QuadProbe, DoubleProbe):
            for size in (8, 16, 64, 512):
                for home in (0, 1, size - 1):
                    with self.subTest(probe=probe_class.__name__, size=size, home=home):
                        probe = probe_class(size)
                        index = probe.reset(home, home * 97 + 13)
                        seen = {index}
                        for _ in range(size - 1):
                            index = probe.next()
                            seen.add(index)
                        self.assertEqual(len(seen), size)

    def test_adversarial_keys_stay_correct(self):
        """Small ints hash to themselves, so every key in a block lands on the
        same home slot when capacity divides the block stride."""
        for name, table in all_strategies():
            with self.subTest(strategy=name):
                for i in range(0, 4096, 8):
                    table[i] = i
                self.assertEqual(len(table), 512)
                self.assertEqual([table[i] for i in range(0, 4096, 8)], list(range(0, 4096, 8)))

    def test_robin_hood_keeps_no_key_poorer_than_its_neighbour(self):
        table = RobinHoodMap(capacity=256)
        for i in range(120):
            table[i] = i
        occupied = [idx for idx, slot in enumerate(table.slots) if isinstance(slot, tuple)]
        runs = []
        for idx in occupied:
            if runs and idx == runs[-1][-1] + 1:
                runs[-1].append(idx)
            else:
                runs.append([idx])
        for run in runs:
            for a in range(len(run)):
                for b in range(a + 1, len(run)):
                    i, j = run[a], run[b]
                    di = table._distance(i, table.slots[i][0])
                    dj = table._distance(j, table.slots[j][0])
                    self.assertLessEqual(di - dj, j - i)

    def test_robin_hood_shortens_clusters_vs_plain_linear(self):
        rng = random.Random(11)
        keys = [rng.randint(0, 10_000) for _ in range(600)]
        plain = OpenAddressMap(LinearProbe, capacity=1024)
        hood = RobinHoodMap(capacity=1024)
        for table in (plain, hood):
            for key in keys:
                table[key] = key
        self.assertLessEqual(hood.longest_probe, plain.longest_probe)
        self.assertLess(hood.probes / hood.ops, plain.probes / plain.ops)

    def test_probe_counts_match_rebuilt_tables(self):
        """Every strategy must find every key it stored, and a table that has
        grown still answers lookups for keys inserted before the growth."""
        rng = random.Random(3)
        for name, factory in PROBE_STRATEGIES.items():
            with self.subTest(strategy=name):
                table = factory(capacity=8)
                ref = {}
                for _ in range(300):
                    key = rng.choice([rng.randint(0, 50), f"s{rng.randint(0, 50)}"])
                    value = rng.randint(0, 1000)
                    table[key] = value
                    ref[key] = value
                self.assertEqual(len(table), len(ref))
                for key, value in ref.items():
                    self.assertEqual(table[key], value)
                self.assertGreater(table.longest_probe, 0)
                self.assertGreater(table.probes, 0)

    def test_linear_probes_longer_than_double_on_blocks(self):
        # Every key hashes to slot 0: this is primary clustering in its purest
        # form, and double hashing escapes it because each key gets its own step.
        keys = [i * 1024 for i in range(200)]
        linear = OpenAddressMap(LinearProbe, capacity=1024)
        double = OpenAddressMap(DoubleProbe, capacity=1024)
        for table in (linear, double):
            for key in keys:
                table[key] = key
        self.assertEqual(linear.longest_probe, 200)
        self.assertLess(double.longest_probe, linear.longest_probe)
        self.assertLess(double.probes / double.ops, linear.probes / linear.ops)


class TestIteration(unittest.TestCase):
    def test_iteration_visits_each_key_once(self):
        for name, table in all_strategies():
            with self.subTest(strategy=name):
                for i in range(50):
                    table[i] = i
                for i in range(0, 50, 3):
                    del table[i]
                keys = list(table)
                self.assertEqual(len(keys), len(set(keys)))
                self.assertEqual(len(keys), sum(1 for i in range(50) if i % 3))

    def test_iteration_survives_deletion_by_key(self):
        table = OpenAddressMap(QuadProbe)
        for i in range(30):
            table[i] = i
        for key in list(table):
            del table[key]
        self.assertEqual(list(table), [])
        self.assertEqual(table.used, 30)
        self.assertEqual(table.stats()["tombstones"], 30)

    def test_repr_mentions_sizes(self):
        table = OpenAddressMap(LinearProbe, capacity=8)
        table["a"] = 1
        self.assertIn("size=1", repr(table))
        self.assertIn("capacity=8", repr(table))


if __name__ == "__main__":
    unittest.main()
