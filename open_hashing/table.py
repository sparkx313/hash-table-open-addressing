"""Hash maps that store entries in the slot array itself.

Chaining keeps a list per bucket; open addressing keeps `capacity` slots and
resolves collisions by probing. The tradeoff is deletion: a slot cannot simply
be emptied, because the empty slot is what terminates a probe sequence. Deleting
the middle of a cluster would strand everything behind it, so a delete leaves a
tombstone that probes walk through but an insert may recycle.
"""

from .probes import DEFAULT_PROBE, DoubleProbe, LinearProbe, QuadProbe, _EMPTY, _TOMB

_MISSING = object()


class OpenAddressMap:
    """Probe sequence chosen by `probe_class`; sizing policy is shared."""

    def __init__(self, probe_class=DEFAULT_PROBE, capacity=8, max_load=0.66):
        if capacity < 8 or capacity & (capacity - 1):
            raise ValueError("capacity must be a power of two, at least 8")
        if not 0 < max_load < 1:
            raise ValueError("max_load must be between 0 and 1")
        self.probe_class = probe_class
        self.max_load = max_load
        self.slots = [_EMPTY] * capacity
        self.size = 0
        self.used = 0
        self.resizes = 0
        self.sweeps = 0
        self.probes = 0
        self.ops = 0
        self.longest_probe = 0
        self._probe = probe_class(capacity)

    @property
    def capacity(self):
        return len(self.slots)

    @property
    def load_factor(self):
        return self.size / self.capacity

    @staticmethod
    def _home(key, capacity):
        return hash(key) % capacity

    def _note_probe(self, count):
        self.probes += count
        self.ops += 1
        if count > self.longest_probe:
            self.longest_probe = count

    def _find(self, key):
        """Return (slot_with_key, first_tombstone, probe_count).

        The probe stops at the first truly empty slot: everything that was ever
        stored for this key lies before it. A table that has been emptied only
        by deletes has no such slot, so the walk is bounded by the number of
        distinct positions the probe sequence can visit, which is the capacity.
        """
        capacity = self.capacity
        probe = self._probe
        index = probe.reset(self._home(key, capacity), key)
        tomb = None
        for count in range(1, capacity + 1):
            slot = self.slots[index]
            if slot is _EMPTY:
                return None, tomb, count
            if slot is _TOMB:
                if tomb is None:
                    tomb = index
            elif slot[0] == key:
                return index, tomb, count
            index = probe.next()
        return None, tomb, capacity

    def _should_grow(self):
        return (self.size + 1) / self.capacity > self.max_load

    def _should_sweep(self):
        """Dead slots still have to be walked over, so a table that has been
        emptied mostly by deletes is slow even when its live load factor looks
        healthy. Give tombstones a quarter of the capacity, then rebuild at the
        same size; growing would only postpone the same work."""
        return self.used - self.size > self.capacity // 4

    def _grow(self):
        self._rehash(self.capacity * 2)

    def _rehash(self, capacity):
        old = self.slots
        self.slots = [_EMPTY] * capacity
        self._probe = self.probe_class(capacity)
        self.used = 0
        self.resizes += 1
        for slot in old:
            if slot is _EMPTY or slot is _TOMB:
                continue
            self._insert_slot(slot[0], slot[1])

    def _insert_slot(self, key, value):
        probe = self._probe
        index = probe.reset(self._home(key, self.capacity), key)
        for _ in range(self.capacity):
            if self.slots[index] is _EMPTY:
                self.slots[index] = (key, value)
                self.used += 1
                return index
            index = probe.next()
        raise RuntimeError("no empty slot in reach; the probe sequence is not a full cycle")

    def __setitem__(self, key, value):
        index, tomb, count = self._find(key)
        self._note_probe(count)
        if index is not None:
            self.slots[index] = (key, value)
            return
        if self._should_grow():
            self._grow()
            self._insert_slot(key, value)
            self.size += 1
            return
        if self._should_sweep():
            self.sweeps += 1
            self._rehash(self.capacity)
            tomb = None
        if tomb is not None:
            self.slots[tomb] = (key, value)
        else:
            self._insert_slot(key, value)
        self.size += 1

    def __getitem__(self, key):
        index, _, count = self._find(key)
        self._note_probe(count)
        if index is None:
            raise KeyError(key)
        return self.slots[index][1]

    def get(self, key, default=None):
        index, _, count = self._find(key)
        self._note_probe(count)
        return default if index is None else self.slots[index][1]

    def __contains__(self, key):
        index, _, count = self._find(key)
        self._note_probe(count)
        return index is not None

    def __delitem__(self, key):
        index, _, count = self._find(key)
        self._note_probe(count)
        if index is None:
            raise KeyError(key)
        self.slots[index] = _TOMB
        self.size -= 1

    def pop(self, key, default=_MISSING):
        index, _, count = self._find(key)
        self._note_probe(count)
        if index is None:
            if default is _MISSING:
                raise KeyError(key)
            return default
        value = self.slots[index][1]
        self.slots[index] = _TOMB
        self.size -= 1
        return value

    def setdefault(self, key, default=None):
        index, _, count = self._find(key)
        self._note_probe(count)
        if index is not None:
            return self.slots[index][1]
        self[key] = default
        return default

    def update(self, other):
        for key, value in dict(other).items():
            self[key] = value

    def __iter__(self):
        for slot in self.slots:
            if slot is not _EMPTY and slot is not _TOMB:
                yield slot[0]

    def items(self):
        for slot in self.slots:
            if slot is not _EMPTY and slot is not _TOMB:
                yield slot

    def keys(self):
        return list(self)

    def values(self):
        return [value for _, value in self.items()]

    def __len__(self):
        return self.size

    def __repr__(self):
        return f"{type(self).__name__}(size={self.size}, capacity={self.capacity})"

    def stats(self):
        return {
            "size": self.size,
            "capacity": self.capacity,
            "load_factor": round(self.load_factor, 4),
            "tombstones": self.used - self.size,
            "resizes": self.resizes,
            "tombstone_sweeps": self.sweeps,
            "mean_probes": round(self.probes / max(1, self.ops), 3),
            "longest_probe": self.longest_probe,
        }


class RobinHoodMap(OpenAddressMap):
    """Linear probing that keeps every key within one slot of the best case.

    On insert the newcomer walks its probe sequence; whenever it meets an
    occupant that is closer to its home slot than the newcomer is, it steals the
    slot and becomes the displaced entry. The invariant is that all keys sharing
    a home slot differ in probe distance by at most one, which caps the worst
    case a lookup can pay.
    """

    def __init__(self, capacity=8, max_load=0.66):
        super().__init__(LinearProbe, capacity, max_load)

    def _distance(self, index, key):
        return (index - self._home(key, self.capacity)) % self.capacity

    def __setitem__(self, key, value):
        index, _, count = self._find(key)
        self._note_probe(count)
        if index is not None:
            self.slots[index] = (key, value)
            return
        if self._should_grow():
            self._grow()
        elif self._should_sweep():
            self.sweeps += 1
            self._rehash(self.capacity)
        self._steal_insert(key, value)
        self.size += 1

    def _steal_insert(self, key, value):
        capacity = self.capacity
        index = self._home(key, capacity)
        distance = 0
        for _ in range(capacity + 1):
            slot = self.slots[index]
            if slot is _EMPTY or slot is _TOMB:
                self.slots[index] = (key, value)
                if slot is _EMPTY:
                    self.used += 1
                return
            if self._distance(index, slot[0]) < distance:
                self.slots[index], key, value = (key, value), slot[0], slot[1]
            index = (index + 1) % capacity
            distance += 1
        raise RuntimeError("table is full")


PROBE_STRATEGIES = {
    "linear": lambda **kw: OpenAddressMap(LinearProbe, **kw),
    "quadratic": lambda **kw: OpenAddressMap(QuadProbe, **kw),
    "double": lambda **kw: OpenAddressMap(DoubleProbe, **kw),
    "robin-hood": lambda **kw: RobinHoodMap(**kw),
}
