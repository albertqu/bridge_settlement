import numpy as np
from scipy.optimize import curve_fit
import os, cv2, matplotlib
import matplotlib.pyplot as plt

""" ===================================
========== MATRIX-ARRAY UTIL ==========
======================================= """


class PseudoLL(list):
    def push(self, obj):
        self.append(obj)

    def __getitem__(self, item):
        return super().__getitem__(len(self) - 1 - item)

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


"""def std_dev(data):
    std_dev = 0
    len_data = len(data)
    mean = sum(data) / len_data
    for i in range(len_data):
        std_dev += (data[i] - mean) ** 2
    return np.sqrt(std_dev / len_data)"""


def reg_pre_debias(ind, data):
    sd = np.std(data)
    miu = np.mean(data)
    debiased_data = []
    debiased_ind = []
    for i in range(len(data)):
        dev = abs((data[i] - miu) / sd)
        if dev < 2:
            debiased_data.append(data[i])
            debiased_ind.append(ind[i])
    return debiased_ind, debiased_data


def gauss_reg(x, y, p0):
    """Given a set of x and y training points, this function
    calculates the gaussian approximation function"""
    param, vm = curve_fit(gauss_hat, x, y, p0=p0)
    return param


def r2_sqe(y, yerr2):
    ym = np.mean(y)
    s_tot = np.sum(np.square(y-ym))
    s_reg = np.sum(yerr2)
    return 1 - s_reg/s_tot


def poly_curve(params, x_input):
    # params contains the coefficients that multiply the polynomial terms, in degree of lowest degree to highest degree
    degree = len(params) - 1
    x_range = [x_input[1], x_input[-1]]
    x = np.linspace(x_range[0], x_range[1], 1000)
    y = x * 0

    for k in range(0, degree + 1):
        coeff = params[k]
        y = y + list(map(lambda z: coeff * z ** k, x))
    return x, y


def gauss_2d(x, y, a, b1, c_s1, b2, c_s2):
    return a * np.exp(- ((x - b1) ** 2 / (2 * c_s1) + (y - b2) ** 2) / (2 * c_s2))


def gauss_hat(x, a, b, c_s):
    return a * np.exp(- (x - b) ** 2 / (2 * c_s))


def gaussian_curve(x_input, a, b, c_s):
    x_range = [x_input[1], x_input[-1]]
    x = np.linspace(x_range[0], x_range[1], 1000)
    y = a * np.exp(- (x - b) ** 2 / (2 * c_s))
    return x, y


def hough_line(x, theta, rho):
    return (rho - x * np.cos(theta)) / np.sin(theta)


""" ===================================
============= GENERAL UTILS ===========
======================================= """


# Generic Helper
def max_min(data):
    # Returns the maximum and minimum for the data
    maxi = data[0]
    max_ind = 0
    mini = data[0]
    min_ind = 0
    for i in range(1, len(data)):
        target = data[i]
        if target > maxi:
            maxi = target
            max_ind = i
        if target < mini:
            mini = target
            min_ind = i
    return max_ind, min_ind


def min_max(data, max, min):
    """ Converts the data to min max space. """
    g_diff = max - min
    return [(d - min) / g_diff for d in data]


def root_finding(x1, x2, y1, y2):
    """Given two points on a line, finds its zero crossing root."""
    return - y1 * (x2 - x1) / (y2 - y1) + x1


def check_crossing(data, i):
    return i + 1 < len(data) and check_cross(data[i], data[i+1])


def check_cross(a, b):
    # Checks whether two points are of opposite signs
    return a * b < 0


def round_up(num):
    down = int(num)
    if num - down > 0:
        return num + 1
    else:
        return num


def folder_to_imgs(img_name_scheme, num_sample):
    """This function takes img files and return cv imgs"""
    return [cv2.imread(img_name_scheme.format(i)) for i in range(1, num_sample + 1)]


def path_prefix_free(path, symbol='/'):
    if path[-len(symbol):] == symbol:
        return path[path.rfind(symbol,0, -len(symbol))+len(symbol):-len(symbol)]
    else:
        return path[path.rfind(symbol)+len(symbol):]


def fname_suffix_free(fname, symbol='.'):
    return fname[:fname.rfind(symbol)]



"""======================================
========== VISUALIZATION UTIL ===========
========================================= """


def visualize_centers(img, centers_v, centers_h, ax):
    xsv = [c[1] for c in centers_v]
    ysv = [c[0] for c in centers_v]
    xsh = [c[1] for c in centers_h]
    ysh = [c[0] for c in centers_h]
    ax.imshow(img, cmap='gray')
    ax.scatter(xsv, ysv, facecolor='blue', linewidths=1)
    ax.scatter(xsh, ysh, facecolor='red', linewidths=1)


def model_contrast_plot(titres, variations, mregx, mregy, path, saveopt):
    font = {'family': 'normal',
            'weight': 'bold',
            'size': 22}
    ls = np.arange(titres)
    matplotlib.rc('font', **font)
    fig = plt.figure(figsize=(200, 100))
    plt.style.use(['seaborn-dark', 'seaborn-paper'])
    plt.subplot(311)
    plt.plot(ls, variations, color='#FA5B3D', marker='o', markeredgewidth=1, linestyle='-', linewidth=2)
    plt.xticks(ls, titres)
    plt.ylabel('deviations(+/-px)')
    plt.xlabel('loss functions')
    plt.title('Convergence By {}'.format(saveopt))
    plt.subplot(312)
    plt.plot(ls, mregx, color='#FA5B3D', marker='o', markeredgewidth=1, linestyle='-', linewidth=2)
    plt.xticks(ls, titres)
    plt.ylabel('Max Reg Error X')
    plt.xlabel('loss functions')
    plt.title('Max Reg Error X')
    plt.subplot(313)
    plt.plot(ls, mregy, color='#FA5B3D', marker='o', markeredgewidth=1, linestyle='-', linewidth=2)
    plt.xticks(ls, titres)
    plt.ylabel('Max Reg Error Y')
    plt.xlabel('loss functions')
    plt.title('Max Reg Error Y')
    plt.show()
    fig.savefig(os.path.join(path, 'Convergence By {}.png'.format(saveopt)))
    print(titres[np.argmin(variations)])
    plt.close('all')

