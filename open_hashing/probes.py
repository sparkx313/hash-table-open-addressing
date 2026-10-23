"""Probe sequences for the open-addressing table.

    LinearProbe  - i + k          : fast, terrible primary clustering
    QuadProbe    - i + k(k+1)/2   : breaks primary clustering only
    DoubleProbe  - i + k*step     : no clustering, step must stay coprime

Each is an iterator over one key's walk: `reset(home, key)` returns the first
slot, `next()` advances. The `step` used by DoubleProbe is derived from the key
and forced odd, so it is always coprime with a power-of-two table size and every
slot is reachable.
"""


class Probe:
    def __init__(self, size):
        self.size = size

    def reset(self, index, key):
        raise NotImplementedError

    def next(self):
        raise NotImplementedError


class LinearProbe(Probe):
    def reset(self, index, key):
        self.index = index
        return index

    def next(self):
        self.index = (self.index + 1) % self.size
        return self.index


class QuadProbe(Probe):
    def reset(self, index, key):
        self.index = index
        self.k = 0
        return index

    def next(self):
        self.k += 1
        self.index = (self.index + self.k) % self.size
        return self.index


class DoubleProbe(Probe):
    def reset(self, index, key):
        self.index = index
        # The step must be odd: with an odd step and a power-of-two size the
        # greatest common divisor is 1, so the walk visits every slot before it
        # repeats. An even step strands half the table and can never terminate.
        step = ((hash(key) >> 7) % (self.size - 1)) + 1
        self.step = step | 1
        return index

    def next(self):
        self.index = (self.index + self.step) % self.size
        return self.index


_EMPTY = object()
_TOMB = object()

DEFAULT_PROBE = LinearProbe
