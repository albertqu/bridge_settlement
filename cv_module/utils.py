import numpy as np
from scipy.optimize import curve_fit

""" ===================================
========== MATRIX-ARRAY UTIL ==========
======================================= """


class PseudoLL(list):
    def push(self, obj):
        self.append(obj)

    def __getitem__(self, item):
        return super(PseudoLL, self).__getitem__(len(self) - 1 - item)

    def __repr__(self):
        rep = '['
        ls = len(self)
        for i in range(ls):
            if i > 0:
                rep += ', '
            val = self[i]
            rep += str(val)
        rep += ']'
        return rep


class FastDataMatrix2D:

    HOR = 1
    VER = 0

    def __init__(self, data, ax, index):
        self._data = data
        self._ax = ax
        assert 0 <= index < self.irange(), "start: {0}, end: {1}, ax: {2}, index: {3}".format(self.start, self.end,
                                                                                              self._ax, index)
        self._index = index
        self.itr = 0
        self.initialize()

    def segmentize(self, start, end):
        assert 0 <= start < end <= self._data.shape[self._ax], \
            "start: {0}, end: {1}, ax: {2}, index: {3}".format(start, end, self._ax, self._index)
        self.start = start
        self.end = end

    def initialize(self):
        self.segmentize(0, self._data.shape[self._ax])

    def irange(self):
        # index range of the fast matrix array
        return self._data.shape[1 - self._ax]

    def set_axis(self, ax):
        """USE copy when trying to switch axis and index"""
        self._ax = ax
        if self._index >= self.irange():
            raise IndexError("Old index {0} is too large for the new axis".format(self._index))
        self.initialize()

    def set_index(self, index):
        assert 0 <= index < self.irange(), "start: {0}, end: {1}, ax: {2}, index: {3}".format(self.start, self.end,
                                                                                              self._ax, index)
        self._index = index

    def extract_array(self):
        """Optimize later for better performance"""
        arr = self._data[self._index, self.start:self.end] if self._ax == FastDataMatrix2D.HOR \
            else self._data[self.start:self.end, self._index]
        return np.array(arr)

    def copy(self, ax=None, index=None):
        if ax is not None and index is not None:
            return FastDataMatrix2D(self._data, ax, index)
        else:
            return FastDataMatrix2D(self._data, self._ax, self._index)

    def __iter__(self):
        raise RuntimeError("You need the ITER method!")

    def __next__(self):
        raise RuntimeError("You need the NEXT method!")

    def __getitem__(self, item):
        return self._data[self._index, item + self.start] if self._ax == FastDataMatrix2D.HOR else self._data[item + self.start, self._index]

    def __setitem__(self, key, value):
        if self._ax == FastDataMatrix2D.HOR:
            self._data[self._index, key + self.start] = value
        else:
            self._data[key + self.start, self._index] = value

    def __len__(self):
        return self.end - self.start

FM = FastDataMatrix2D

""" ===================================
=========== REGRESSION UTILS ==========
======================================= """


def gauss_reg(x, y, p0):
    """Given a set of x and y training points, this function
    calculates the gaussian approximation function"""
    param, vm = curve_fit(gauss_hat, x, y, p0=p0)
    return param


def gauss_hat(x, a, b, c_s):
    return a * np.exp(- (x - b) ** 2 / (2 * c_s))


""" ===================================
============= GENERAL UTILS ===========
======================================= """


def root_finding(x1, x2, y1, y2):
    """Given two points on a line, finds its zero crossing root."""
    return - y1 * (x2 - x1) / (y2 - y1) + x1


def check_crossing(data, i):
    return i + 1 < len(data) and check_cross(data[i], data[i+1])


def check_cross(a, b):
    # Checks whether two points are of opposite signs
    return a * b < 0
