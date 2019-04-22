import cv2
import numpy as np
import csv
from scipy.optimize import curve_fit
from scipy.optimize import least_squares
import heapq
from decimal import Decimal
import matplotlib
from math import sqrt, acos, asin, degrees, sin, cos
from fitting import leastSquares, improvedCost
import matplotlib.pyplot as plt


""" ===================================
========== ANALYTIC GEOMETRY ==========
======================================= """


class Line:
    inf = float('inf')

    def __init__(self, a=inf, b=inf, p1=None, p2=None, data=None):
        if p1 and p2:
            self.point(p1, p2)
        elif data:
            self.reg(data)
        else:
            self.a = a
            self.b = b

    def reg(self, data):
        D = np.array([[d[0], 1] for d in data])
        y = np.array([d[1] for d in data])
        self.a, self.b = leastSquares(D, y)

    def point(self, p1, p2):
        self.a = 1

    @staticmethod
    def intersect(l1, l2):
        x = (l2.b - l1.b) / (l1.a - l2.a)
        y = l1.a * x + l1.b
        return x, y


class HoughLine:
    """ Houghline class that fits a analytical line to the data, r = xcos(\theta)+ysin(\theta)

    loss: loss kernel during least square
    theta, rho: theta angle and distance to origin for the houghline

    """
    def __init__(self, theta=0, rho=0, x=None, data=None, loss='soft_l1'):
        if data is not None and x is not None:
            self.x = x
            self.data = data
            self.loss = loss
            self.reg(x, data)
        else:
            self._r = rho
            self._t = theta
            self._s = sin(theta)
            self._c = cos(theta)
        self.debias = self.debias_old
        self.pred = None

    def reg(self, x, data):
        """ Given sample points x and the labels data, find a best fit Houghline with self.loss and leastSquares
        """
        x1, x2 = np.mean(x[:int(len(x) / 2)]), np.mean(x[int(len(x) / 2):])
        y1, y2 = np.mean(data[:int(len(data) / 2)]), np.mean(data[int(len(data) / 2):])
        theta0 = theta_pred(x1, y1, x2, y2)
        p0 = [theta0, np.mean(x) * np.cos(theta0) + np.mean(data) * np.sin(theta0)]
        res = least_squares(HoughLine.get_err, p0, loss=self.loss, f_scale=3, args=(x, data))
        self.opti = res.optimality
        angle = normalize_angle(res.x[0])
        self._t = angle
        self._r = res.x[1]
        self._s = sin(angle)
        self._c = cos(angle)

    @staticmethod
    def get_err(vars, xs, ys):
        return xs * np.cos(vars[0]) + ys * np.sin(vars[0]) - vars[1]

    def fit_x(self, x):
        # print(self._r, self._t)
        fits = (self._r - x * self._c) / self._s if self._s != 0 else np.nan
        self.pred = fits
        return fits

    def debias_old(self, thres=1.0):
        """ @:returns tuple with a) zerr_before: regression error before debias
         b) zerr_after: regression error after debias"""
        x, y = self.x, self.data
        zero_hat_before = x * self._c + y * self._s - self._r
        zerr_before = np.square(zero_hat_before)
        conds = (zerr_before - np.mean(zerr_before)) / np.std(zerr_before) <= thres
        new_x, new_y = x[conds], y[conds]
        self.x, self.data = new_x, new_y
        self.reg(new_x, new_y)
        zero_hat_after = new_x * self._c + new_y * self._s - self._r
        zerr_after = np.square(zero_hat_after)
        return zerr_before, zerr_after

    def debias_z(self, thres=1.0):
        """ @:returns tuple with a) zerr_before: regression error before debias
         b) zerr_after: regression error after debias"""
        x, y = self.x, self.data
        zero_hat_before = x * self._c + y * self._s - self._r
        zerr_before = np.square(zero_hat_before)
        conds = np.abs(zero_hat_before) / np.sqrt(np.sum(zerr_before)/len(zerr_before)) <= thres
        new_x, new_y = x[conds], y[conds]
        self.x, self.data = new_x, new_y
        self.reg(new_x, new_y)
        zero_hat_after = new_x * self._c + new_y * self._s - self._r
        zerr_after = np.square(zero_hat_after)
        return zerr_before, zerr_after

    def debias_y(self, thres=1.0):
        """ @:returns tuple with a) zerr_before: regression error before debias
         b) zerr_after: regression error after debias
         Uses Standard estimate error to filter out bad regressions.
         """
        x, y = self.x, self.data
        y_hat_before = self.fit_x(x)
        yerr_before = y_hat_before - y
        yerrb2 = yerr_before ** 2
        se_reg = np.sqrt(np.sum(yerrb2) / len(yerr_before))
        conds = np.abs(yerr_before) / se_reg  <= thres
        new_x, new_y = x[conds], y[conds]
        self.x, self.data = new_x, new_y
        self.reg(new_x, new_y)
        y_hat_after = self.fit_x(new_x)
        yerr_after = np.square(y_hat_after - new_y)
        return yerrb2, yerr_after

    def point_gen(self):
        x0 = self._c * self._r
        y0 = self._s * self._r
        x1 = int(x0 + 10000 * (-self._s))
        y1 = int(y0 + 10000 * (self._c))
        x2 = int(x0 - 10000 * (-self._s))
        y2 = int(y0 - 10000 * (self._c))
        return (x1, y1), (x2, y2)

    def __str__(self):
        return 'hough line with cos:{0}, sin:{1}, rho:{2}, theta:{3}'.format(self._c, self._s,
                                                                             self._r, degrees(self._t))

    @staticmethod
    def intersect(l1, l2):
        if l1._s == 0 and l2._s == 0:
            return float('inf'), float('inf')
        if l1._s == 0:
            return l1._r, l2.fit_x(l1._r)
        elif l2._s == 0:
            return l2._r, l1.fit_x(l2._r)
        x = (l2._r / l2._s - l1._r / l1._s) / (l2._c / l2._s - l1._c / l1._s)
        y = l1.fit_x(x)
        return x, y


# QC: QUARTER_CYCLE, HC: HALF_CYCLE, TQC: THIRD_QUARTER_CYCLE, FC: FULL_CYCLE
QC = np.pi / 2
HC = np.pi
TQC = 3 * np.pi / 2
FC = 2 * np.pi


def angle_interp(s, c):
    if c == 1.0:
        return 0
    elif s == 1.0:
        return np.pi / 2
    elif c == -1.0:
        return np.pi
    elif s == -1.0:
        return np.pi * 3 / 2
    elif c < 0 < s:
        return acos(c)
    elif s < 0 and c < 0:
        return 2 * np.pi - acos(c)
    elif s < 0 < c:
        return asin(s) + 2 * np.pi
    else:
        return asin(s)


def normalize_angle(angle):
    res = angle - (angle // FC) * FC
    return res


def theta_pred(x1, y1, x2, y2):
    dx = x2 - x1
    dy = y2 - y1
    angle = sin_angle_from_points(dx, dy)
    if dx * dy <= 0:
        return np.pi / 2 - angle
    else:
        root = root_finding(x1, x2, y1, y2)
        if root < 0:
            return np.pi / 2 + angle
        else:
            return angle + 3 * np.pi / 2


def sin_angle_from_points(dx, dy):
    return asin(abs(dy) / sqrt(dx ** 2 + dy ** 2))


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


""" ===================================
=========== REGRESSION UTILS ==========
======================================= """


def std_dev(data):
    std_dev = 0
    len_data = len(data)
    mean = sum(data) / len_data
    for i in range(len_data):
        std_dev += (data[i] - mean) ** 2
    return sqrt(std_dev / len_data)


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


""" =======================================
========= EDGE DETECTION UTILS ============
=========================================== """


def gather_all(imgr, sample_int=30, gk=9, ks=-1):
    """ @:returns centers_v, centers_h, list of possible centers gathered by algorithm, represented as (row, col)
    """
    dimr = imgr.shape[0]
    dimc = imgr.shape[1]
    # Image Processing
    gksize = (gk, gk)
    sigmaX = 0
    img = cv2.GaussianBlur(imgr, gksize, sigmaX)
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=ks)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=ks)
    # Parameter Setting
    nr = sample_int
    nc = sample_int
    # Gathering Data
    centers_v = []
    centers_h = []
    while nr < dimr:
        data_x = sobelx[nr, :]
        am = gather_centers(data_x, img[nr, :], centers_v, 0, nr, gaussian_center)
        nr += sample_int
    while nc < dimc:
        data_y = sobely[:, nc]
        raw_y = img[:, nc]
        am = gather_centers(data_y, img[:, nc], centers_h, 1, nc, gaussian_center)
        nc += sample_int
    return centers_v, centers_h


def gather_centers(grad, raw_data, reserve, ax, ax_n, center_method):
    # Given grad and raw_data, insert the possible beam centers to reserves
    max_grad, min_grad = get_maxi_mini(grad)
    max_q, min_q, locs = bi_order_sort(max_grad, min_grad)
    miu, sig = np.mean(raw_data), np.std(raw_data)
    avant = lambda locs, i: locs[i - 1] if i - 1 >= 0 else None
    apres = lambda locs, i: locs[i + 1] if i + 1 < len(locs) else None
    i = 0
    peaked = False
    while (i < 2 or peaked) and max_q and min_q:
        if peaked:
            top = min_q.pop()
            av = avant(locs, top[1])
            if av in max_q and beam_bound(raw_data, miu, sig, av, top):
                mid = center_method(grad, raw_data, av[0], top[0])
                reserve.append((ax_n, mid) if ax == 0 else (mid, ax_n))
                max_q.remove(av)
                i += 1
            peaked = False
        else:
            max_top = max_q.pop()
            min_top = min_q[0]
            if beam_bound(raw_data, miu, sig, max_top, min_top):
                # print("MaxMin: ", max_top, min_top)
                mid = center_method(grad, raw_data, max_top[0], min_top[0])
                reserve.append((ax_n, mid) if ax == 0 else (mid, ax_n))
                min_q.remove(min_top)
                i += 1
            else:
                peaked = True
                ap = apres(locs, max_top[1])
                # QUICK FIX HERE, THINK MORE
                if ap in min_q and beam_bound(raw_data, miu, sig, max_top, ap):
                    # print("Max_apres: ", max_top, ap)
                    mid = center_method(grad, raw_data, max_top[0], ap[0])
                    reserve.append((ax_n, mid) if ax == 0 else (mid, ax_n))
                    min_q.remove(ap)
                    i += 1
    return i


def sup_tout(max_q, min_q, d, guess):
    if guess == 0:
        prem, deux = max_q, min_q
    else:
        prem, deux = min_q, max_q
    if d in prem:
        prem.remove(d)
    elif d in prem:
        deux.remove(d)
    else:
        raise RuntimeError("Tried to remove already removed term! {0}, {1}".format(d, guess))


def get_maxi_mini(data, ceil=3):
    # Given data, pluck the maxis and minis and store them in minpq and maxpq respectively
    max_grad = []
    min_grad = []
    nat_select = lambda a, b: a if a[0] >= b[0] else b
    maxer = lambda d: len(max_grad) < ceil or d > max_grad[0][0]
    miner = lambda d: len(min_grad) < ceil or d > min_grad[0][0]
    active_max, active_min = None, None

    for i, d in enumerate(data):
        if not active_max:
            if maxer(d):
                active_max = (d, i)
        else:
            curr = (d, i)
            active_max = nat_select(curr, active_max)
        if active_max and (check_crossing(data, i) or i == len(data) - 1):
            heapq.heappush(max_grad, active_max)
            if len(max_grad) > ceil:
                heapq.heappop(max_grad)
            active_max = None

        if not active_min:
            if miner(-d):
                active_min = (-d, i)
        else:
            curr = (-d, i)
            active_min = nat_select(curr, active_min)
        if active_min and (check_crossing(data, i) or i == len(data) - 1):
            heapq.heappush(min_grad, active_min)
            if len(min_grad) > ceil:
                heapq.heappop(min_grad)
            active_min = None

    return max_grad, min_grad


def bi_order_sort(max_grad, min_grad):
    # Takes in a max_grad and min_grad (heapq), returns a locational ordered array and a magnitude queue.
    locs = []
    max_q, min_q = PseudoLL(), PseudoLL()
    while len(max_grad) and len(min_grad):
        mat = heapq.heappop(max_grad)
        mit = heapq.heappop(min_grad)
        maxt, mint = [mat[1], None], [mit[1], None]
        locs.append(maxt)
        locs.append(mint)
        max_q.push(maxt)
        min_q.push(mint)
    locs.sort(key=lambda pair: pair[0])
    for i in range(len(locs)):
        locs[i][1] = i
    return max_q, min_q, locs


def beam_bound(raw, miu, sig, a1, a2, thres=2):
    # given raw data, determine if a1, a2 are beam bound entries, where a1, a2 are tuples [index, *(loc)]
    return a1 and a2 and a2[1] == a1[1] + 1 and (raw[(a1[0] + a2[0]) // 2] - miu) / sig >= thres


def edge_max_min(data):
    # Returns a *safe* edge maxi, mini for the data
    # TODO: OPTIMIZE THE WIDTH AND VALUE THRESHOLD
    width_thres = 90
    value_thres = 20
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
    maxi, mini = max_ind, min_ind
    assert data[maxi] >= value_thres and mini - maxi <= width_thres and maxi < mini
    return maxi, mini


def smart_interval(start, end, data):
    start = 0 if start < 0 else start
    end = len(data) if end > len(data) else end
    return start, end


def edge_preprocess(data, padding):
    maxi, mini = edge_max_min(data)
    start, end = smart_interval(maxi - padding, mini + padding + 1, data)
    return start, end


# ZERO CROSSING (PEAK FINDING)
def zero_crossing(data):
    # Yields the center with zero-crossing method
    maxi, mini = max_min(data)
    cross = -1
    cross_count = 0
    for j in range(maxi, mini):
        y1 = data[j]
        x2 = j + 1
        y2 = data[x2]
        if check_cross(y1, y2):
            cross = root_finding(j, x2, y1, y2)  # change to function later
            cross_count += 1
    return -1 if cross_count > 1 else cross


# AVERAGE METHOD FOR EDGE CENTER FINDING
def edge_converge_base(data):
    maxi, mini = max_min(data)
    width_thres = 70
    if mini - maxi > width_thres:
        return -1
    else:
        return (maxi + mini) / 2


# PSEUDO EDGE (closest noise gradient change) CENTER FINDING
def edge_converge_extreme(data):
    maxi, mini = max_min(data)
    width_thres = 70
    if mini - maxi > width_thres:
        return -1
    emax = -1
    emin = -1
    maxflag = True
    minflag = True
    cmax = maxi
    cmin = mini
    while cmax > 0 and cmin < len(data) - 1 and (maxflag or minflag):
        max2 = cmax - 1
        maxy1, maxy2 = data[cmax], data[max2]
        min2 = cmin + 1
        miny1, miny2 = data[cmin], data[min2]
        if check_cross(maxy1, maxy2) and maxflag:
            emax = root_finding(cmax, max2, maxy1, maxy2)
            maxflag = False
        if check_cross(miny1, miny2) and minflag:
            emin = root_finding(cmin, min2, miny1, miny2)
            minflag = False
        if maxflag:
            cmax -= 1
        if minflag:
            cmin += 1
    return -1 if (emax == -1 or emin == -1) else (emax + emin) / 2


def extract_extrema(data):
    len_data = len(data)
    mean = sum(data) / len_data
    std_dev = 0
    for x in data:
        std_dev += (x - mean) ** 2
    std_dev = sqrt(std_dev / len_data)
    signifmax = []
    signifmin = []
    coeff = 3
    for i, x in enumerate(data):
        diff = (x - mean) / std_dev
        if diff >= coeff:
            signifmax.append(i)
        elif diff <= -coeff:
            signifmin.append(i)
    return signifmax, signifmin


# CENTROID METHOD FOR EDGE CENTER FINDING
def edge_centroid(data, img_data, padding=20):
    # With Gaussian Blur might achieve the best performance
    try:
        start, end = edge_preprocess(data, padding)
    except AssertionError:
        return -1
    isums = 0
    total = 0
    for i in range(start, end):
        isums += img_data[i] * i
        total += img_data[i]
    return isums / total if total else -1


def centroid_seg(data, start, end):
    """ Given a segment (start, end) of the data,
    find the centroid. """
    isums = 0
    total = 0
    start = 0 if start < 0 else start
    end = len(data) if end > len(data) else end
    for i in range(start, end):
        isums += data[i] * i
        total += data[i]
    return isums / total if total else -1


# POLYNOMIAL FITTING
def poly_fitting(data, img_data, padding=20):
    # TODO: OPTIMIZE THE AWKWARD TYPE CHECKING AND THE EXTRACT_ARRAY
    try:
        start, end = edge_preprocess(data, padding)
    except AssertionError:
        return -1
    x = np.array(range(start, end))
    if type(img_data) == FastDataMatrix2D:
        img_data.segmentize(start, end)
        y = img_data.extract_array()
    else:
        #y = np.array(img_data[start:end])
        idata = np.zeros(end - start)
        for i in range(start, end):
            idata[i - start] = img_data[i]
    degrees, params, cost = improvedCost(x, y, x, y, 1, 7)
    ind = np.argmin(cost)
    degree = degrees[ind]
    param = params[ind]
    degree_register(degree)
    curve_x, curve_y = poly_curve(param, x)
    center_id = np.argmax(curve_y)
    return curve_x[center_id]


def poly_fitting_params(data, img_data, padding=10):
    # TODO: OPTIMIZE THE AWKWARD TYPE CHECKING AND THE EXTRACT_ARRAY
    maxi, mini = max_min(data)
    width_thres = 90
    value_thres = 20
    if data[maxi] < value_thres or mini - maxi > width_thres:
        print(data[maxi], mini, maxi)
        raise AssertionError("Bad column or row")
    start = maxi - padding
    end = mini + padding + 1
    start, end = smart_interval(start, end, data)
    if start > end:
        try:
            print(maxi, mini, img_data._ax, start, end)
        except:
            print("else", maxi, mini, start, end)
    x = np.array(range(start, end))
    if type(img_data) == FastDataMatrix2D:
        img_data.segmentize(start, end)
        y = img_data.extract_array()
    else:
        idata = np.zeros(end - start)
        for i in range(start, end):
            idata[i - start] = img_data[i]
    degrees, params, cost = improvedCost(x, y, x, y, 1, 7)
    ind = np.argmin(cost)
    degree = degrees[ind]
    param = params[ind]
    degree_register(degree)
    return param, x


DEGREES = {}


def degree_register(elem):
    if elem in DEGREES:
        DEGREES[elem] += 1
    else:
        DEGREES[elem] = 1


def gaussian_center(data, img_data, maxi, mini, padding=20):
    #padding = (mini - maxi) // 2
    start, end = smart_interval(maxi - padding, mini + padding + 1, data)
    x = np.array(range(start, end))
    if type(img_data) == FastDataMatrix2D:
        img_data.segmentize(start, end)
        idata = img_data.extract_array()
    else:
        idata = np.zeros(end - start)
        for i in range(start, end):
            idata[i - start] = img_data[i]
    try:
        param = gauss_reg(x, idata, p0=[10, (maxi + mini) / 2, std_dev(idata)])
    except RuntimeError:
        return -1
    return param[1]


# GAUSSIAN FITTING
def gaussian_fitting(data, img_data, padding=20):
    # TODO: OPTIMIZE THE AWKWARD TYPE CHECKING, ALONG WITH THE WIDTH THRESHOLD
    try:
        maxi, mini = edge_max_min(data)
        print((mini - maxi) // 2)
    except AssertionError:
        return -1
    start, end = smart_interval(maxi - padding, mini + padding + 1, data)
    x = np.array(range(start, end))
    if type(img_data) == FastDataMatrix2D:
        img_data.segmentize(start, end)
        idata = img_data.extract_array()
    else:
        idata = np.zeros(end-start)
        for i in range(start, end):
            idata[i-start] = img_data[i]
    try:
        param = gauss_reg(x, idata, p0=[10, (maxi + mini) / 2, std_dev(idata)])
    except RuntimeError:
        return -1
    return param[1]


def gaussian_fitting_params(data, img_data, padding=10):
    # TODO: OPTIMIZE THE AWKWARD TYPE CHECKING, ALONG WITH THE WIDTH THRESHOLD
    maxi, mini = max_min(data)
    width_thres = 90
    value_thres = 20  # TODO: CONSOLIDATE THIS VALUE
    if data[maxi] < value_thres or mini - maxi > width_thres:
        print(data[maxi], mini, maxi)
        raise AssertionError("Bad column or row")
    start = maxi - padding
    end = mini + padding
    start, end = smart_interval(start, end, data)
    x = np.array(range(start, end))
    if type(img_data) == FastDataMatrix2D:
        img_data.segmentize(start, end)
        idata = img_data.extract_array()
    else:
        idata = np.zeros(end - start)
        for i in range(start, end):
            idata[i-start] = img_data[i]
    return gauss_reg(x, idata, p0=[10, (maxi + mini) / 2, std_dev(idata)]), x


"""======================================
======== IMAGE PROCESSING UTIL ==========
========================================= """


def image_diff(img1, img2):
    # Calculate the difference in images
    i1 = img1.astype(np.int16)
    i2 = img2.astype(np.int16)
    res = i1 - i2
    dst = np.zeros_like(res, dtype=np.uint8)
    return cv2.normalize(res, dst, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=8)


def gauss_bg_deduce(x, img_data):
    # TODO: OPTIMIZE PERFORMANCE
    idata = img_data.extract_array()
    p0 = [1, len(img_data) / 2, std_dev(idata)]
    a, b, c_s = gauss_reg(x, idata, p0=p0)
    rem_gauss = gauss_hat(x, a, b, c_s)
    new_y = idata - rem_gauss
    return x, rem_gauss, new_y


def significance_test(data, val):
    """Outputs whether the maximum or minimum value is a significant value."""
    return val in extract_extrema(data)[0] or val in extract_extrema(data)[1]


def get_center_val(img):
    return img.item(img.shape[0] // 2, img.shape[1] // 2)


def test_noise_reduce(iname, numIMG=5):
    imgs = iname
    imgr = None
    original = None
    for i in range(1, numIMG + 1):
        target = cv2.imread(imgs.format(i), 0)
        if target is None:
            raise AttributeError("File {0} not found".format(iname))
        if i == 1:
            imgr = np.uint16(target)
            original = target
        else:
            imgr = imgr + target
    return imgr / numIMG, original


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


FM = FastDataMatrix2D


def folder_to_imgs(img_name_scheme, num_sample):
    """This function takes img files and return cv imgs"""
    return [cv2.imread(img_name_scheme.format(i)) for i in range(1, num_sample + 1)]


def center_detect_hough(img):
    centers_v, centers_h = gather_all(img)
    centpointsv = np.zeros_like(img, dtype=np.uint8)
    centpointsh = np.zeros_like(img, dtype=np.uint8)
    #print(centers_v+centers_h)
    for r, c in centers_v:
        try:
            centpointsv[int(r)][int(c)] = 255
        except:
            continue
    for r, c in centers_h:
        try:
            centpointsh[int(r)][int(c)] = 255
        except:
            continue
    linesv = cv2.HoughLines(centpointsv, 0.5, 0.005, 1)
    linesh = cv2.HoughLines(centpointsh, 0.5, 0.005, 1)
    all_lines = []
    try:
        for rho, theta in linesv[0]:
            #print(rho, theta)
            all_lines.append(HoughLine(theta, rho))
    except:
        print(linesv)
    for rho, theta in linesh[0]:
        all_lines.append(HoughLine(theta, rho))
    ct = HoughLine.intersect(all_lines[0], all_lines[-1])
    return ct


def center_detect(folder_path, img_name, sample_int=30, gk=9, ks=-1, l='soft_l1', debias="z", norm=1, visual=False,
                  catch=None, stat=False,
                  saveopt=""):
    """This function takes in a list of images and output x, y [pixel] coordinates of the center of the cross hair
    hs: HORIZONTAL SLICE!  vs: VERTICAL SLICE!"""
    # TODO: CHANGE DEBIAS AND CHANGE VISUALIZATION PART TO HAVE:
    # TODO: 1. CENTER_VISUALIZATION WITH LINE DRAWN
    # TODO: 2. ERROR BEFORE AND AFTER
    imgr = cv2.imread(os.path.join(folder_path, img_name), 0)
    if norm:
        aut = np.empty_like(imgr, dtype=np.uint8)
        if norm == 1:
            imgr = cv2.normalize(imgr, imgr, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=8)
        else:
            imgr = cv2.normalize(imgr, aut, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=8)
    return center_detect_base(imgr, folder_path, img_name, sample_int, gk, ks, l, debias, visual, catch, stat, saveopt)


def center_detect_invar(imgr, sample_int=30, gk=9, ks=-1, l='soft_l1', debias="z", visual=False,
                        folder_path="../", img_name='.png', saveopt="", stat=False):
    """This function takes in a list of images and output x, y [pixel] coordinates of the center of the cross hair
    hs: HORIZONTAL SLICE!  vs: VERTICAL SLICE!"""
    centers_v, centers_h = gather_all(imgr, sample_int, gk, ks)
    centers_v, centers_h = np.array(centers_v), np.array(centers_h)
    centers_vx, centers_vy = centers_v[:, 1], centers_v[:, 0]
    centers_hx, centers_hy = centers_h[:, 1], centers_h[:, 0]

    line_v, line_h = HoughLine(x=centers_vx, data=centers_vy), HoughLine(x=centers_hx, data=centers_hy, loss=l)
    if visual:
        namespace = 'robust_reg/{0}{1}_{2}'.format(folder_path[3:], l, img_name)
        print(namespace)
        vp1_av, vp2_av = line_v.point_gen()
        hp1_av, hp2_av = line_h.point_gen()
    if debias == 'z':
        zv_err_av, zv_err_ap = line_v.debias()
        zh_err_av, zh_err_ap = line_h.debias()
    else:
        zv_err_av, zv_err_ap = line_v.debias_y()
        zh_err_av, zh_err_ap = line_h.debias_y()

    if visual:
        xv_av, xv_ap = np.arange(len(zv_err_av)), np.arange(len(zv_err_ap))
        xh_av, xh_ap = np.arange(len(zh_err_av)), np.arange(len(zh_err_ap))
        hp1_ap, hp2_ap = line_h.point_gen()
        vp1_ap, vp2_ap = line_v.point_gen()
        cenhs = list(zip(line_h.data, line_h.x))
        cenvs = list(zip(line_v.data, line_v.x))
        cv2.line(imgr, hp1_ap, hp2_ap, (0, 0, 255), 1, lineType=cv2.LINE_AA)
        cv2.line(imgr, vp1_ap, vp2_ap, (0, 0, 255), 1, lineType=cv2.LINE_AA)

        fig1 = plt.figure(figsize=(20, 10))
        ax1 = fig1.add_subplot(121)
        visualize_centers(imgr, centers_v, centers_h, ax1)
        ax1.set_title('Before')
        ax2 = fig1.add_subplot(122)
        visualize_centers(imgr, cenvs, cenhs, ax2)
        ax2.set_title('After')
        fig2 = plt.figure(figsize=(20, 10))
        ax3 = fig2.add_subplot(211)
        ax3.plot(xh_av, zh_err_av, 'mo-', xh_ap, zh_err_ap, 'bo-')
        ax3.set_title('HOR Loss Before and After')
        ax3.legend(['Before', 'After'])
        ax4 = fig2.add_subplot(212)
        ax4.plot(xv_av, zv_err_av, 'mo-', xv_ap, zv_err_ap, 'bo-')
        ax4.set_title('VER Loss Before and After')
        ax4.legend(['Before', 'After'])
        fpath1 = os.path.join(ROOT_DIR, namespace[:-4]+'_lines'+saveopt+namespace[-4:])
        fig1.savefig(fpath1)
        print("Saved {}".format(fpath1))
        fig2.savefig(os.path.join(ROOT_DIR, namespace[:-4] + '_errs' + saveopt+ namespace[-4:]))
        plt.close("all")
    return HoughLine.intersect(line_h, line_v)


def center_detect_base(imgr, folder_path, img_name, sample_int=30, gk=9, ks=-1, l='soft_l1', debias="z",
                       visual=False, catch=None, stat=False,saveopt=""):
    """ Takes in preprocessed (ambient invariant or normalization) and output x, y [pixel]
    coordinates of the center of the cross hair
    centers_v: vertical line!  hs: horizontal slice!
    So far it seems y (horizontal line) resolution is better than that of x (vertical line).
    """
    ylim, xlim = imgr.shape
    centers_v, centers_h = gather_all(imgr, sample_int, gk, ks)
    centers_v, centers_h = np.array(centers_v), np.array(centers_h)
    try:
        centers_vx, centers_vy = centers_v[:, 1], centers_v[:, 0]
        centers_hx, centers_hy = centers_h[:, 1], centers_h[:, 0]
    except IndexError:
        return -1, -1, None

    line_v, line_h = HoughLine(x=centers_vx, data=centers_vy), HoughLine(x=centers_hx, data=centers_hy, loss=l)
    if visual:
        fn = lambda p, n: p[p.find(n)+len(n)+1:]
        namespace = os.path.join(ROOT_DIR, 'robust_reg/{0}{1}_{2}'.format(fn(folder_path, ROOT_DIR), l, img_name))
        vp1_av, vp2_av = line_v.point_gen()
        hp1_av, hp2_av = line_h.point_gen()
    if debias == 'z':
        zv_err_av, zv_err_ap = line_v.debias_z()
        zh_err_av, zh_err_ap = line_h.debias_z()
    elif debias == "y":
        zv_err_av, zv_err_ap = line_v.debias_y()
        zh_err_av, zh_err_ap = line_h.debias_y()
    else:
        zv_err_av, zv_err_ap = line_v.debias_old()
        zh_err_av, zh_err_ap = line_h.debias_old()
    if catch is not None: # TODO: remove catch
        catch[1].append(sqrt(np.mean(zh_err_ap)))
        catch[0].append(sqrt(np.mean(zv_err_ap)))

    if debias == 'y':
        C1, C2 = np.corrcoef(line_h.data, line_h.pred)[1, 0], np.corrcoef(line_v.data, line_v.pred)[1, 0]

        metrics = r2_sqe(line_h.data, zh_err_ap), r2_sqe(line_v.data, zv_err_ap), C1, C2, C1 ** 2, C2 ** 2

        show_metrics = "Hor R2: {} Ver R2: {}, C_Hor, {}, C_Ver: {}, Hor_C2: {}, Ver_C2: {}".format(*metrics)
        print(show_metrics)

    if stat:
        lh_y, lh_pred = line_h.data, (line_h.pred if line_h.pred else line_h.fit_x(line_h.x))
        lv_y, lv_pred = line_v.data, (line_v.pred if line_v.pred else line_v.fit_x(line_v.x))
        tot_y, tot_pred = np.concatenate((lh_y, lv_y)), np.concatenate((lh_pred, lv_pred))
        from sklearn.metrics import explained_variance_score, r2_score, mean_absolute_error, mean_squared_error,\
            median_absolute_error
        mfs = explained_variance_score, r2_score, mean_absolute_error, mean_squared_error, median_absolute_error
        hseq, vseq = tuple(map(lambda f: f(lh_y, lh_pred), mfs)), tuple(map(lambda f: f(lv_y, lv_pred), mfs))
        vseq2= (np.sqrt(np.sum(zv_err_av) /len(zv_err_av)), np.sqrt(np.sum(zv_err_ap) /len(zv_err_ap)),
                np.sqrt(np.max(zv_err_av)), np.sqrt(np.max(zv_err_ap)))
        hseq2 = (np.sqrt(np.sum(zh_err_av) / len(zh_err_av)), np.sqrt(np.sum(zh_err_ap) / len(zh_err_ap)),
                np.sqrt(np.max(zh_err_av)), np.sqrt(np.max(zh_err_ap)))
        vseq = vseq
        hseq = hseq
        totseq = tuple(map(lambda f: f(tot_y, tot_pred), mfs))
        tot_err_av, tot_err_ap = np.concatenate((zv_err_av, zh_err_av)), np.concatenate((zv_err_ap, zh_err_ap))
        totseq2 = (np.sqrt(np.sum(tot_err_av) / len(tot_err_av)), np.sqrt(np.sum(tot_err_ap) / len(tot_err_ap)),
                                                    np.sqrt(np.max(tot_err_av)), np.sqrt(np.max(tot_err_ap)))
    x, y = HoughLine.intersect(line_h, line_v)
    if x >= xlim or x < 0 or y < 0 or y >= ylim:
        return -1, -1, None

    if visual:
        xv_av, xv_ap = np.arange(len(zv_err_av)), np.arange(len(zv_err_ap))
        xh_av, xh_ap = np.arange(len(zh_err_av)), np.arange(len(zh_err_ap))
        hp1_ap, hp2_ap = line_h.point_gen()
        vp1_ap, vp2_ap = line_v.point_gen()
        cenhs = list(zip(line_h.data, line_h.x))
        cenvs = list(zip(line_v.data, line_v.x))
        cv2.line(imgr, hp1_ap, hp2_ap, (0, 0, 255), 1, lineType=cv2.LINE_AA)
        cv2.line(imgr, vp1_ap, vp2_ap, (0, 0, 255), 1, lineType=cv2.LINE_AA)
        fig1 = plt.figure(figsize=(20, 10))
        ax1 = fig1.add_subplot(121)
        visualize_centers(imgr, centers_v, centers_h, ax1)
        ax1.set_title('Before')
        ax2 = fig1.add_subplot(122)
        visualize_centers(imgr, cenvs, cenhs, ax2)
        ax2.scatter([x], [y], facecolor='green', linewidths=1)
        ax2.set_title('After')
        fig2 = plt.figure(figsize=(20, 10))
        ax3 = fig2.add_subplot(211)
        ax3.plot(xh_av, zh_err_av, 'mo-', xh_ap, zh_err_ap, 'bo-')
        ax3.set_title('HOR Loss Before and After')
        ax3.legend(['Before', 'After'])
        ax4 = fig2.add_subplot(212)
        ax4.plot(xv_av, zv_err_av, 'mo-', xv_ap, zv_err_ap, 'bo-')
        ax4.set_title('VER Loss Before and After')
        ax4.legend(['Before', 'After'])
        if debias == 'y':
            fig2.suptitle(show_metrics)
            fig1.suptitle(show_metrics)
        fig1.savefig(namespace[:-4] + '_lines' + saveopt + namespace[-4:])
        fig2.savefig(namespace[:-4] + '_errs' + saveopt + namespace[-4:])
        plt.close('all')
    if stat:
        return x, y, (line_h.opti, line_v.opti) + hseq2 + vseq2 + totseq2 + hseq + vseq + totseq
    return x, y


def expr2(series, dossier, ns='img_{0}.png'):
    folder = os.path.join(ROOT_DIR, dossier)
    ins = os.path.join(folder, ns)
    repertoire = []
    i = 0
    while i < len(series):
        try:
            ambi, laser = cv2.imread(ins.format(series[i]), 0), cv2.imread(ins.format(series[i+1]), 0)
            res = image_diff(laser, ambi)
        except:
            print(ins.format(series[i]))
        repertoire.append(center_detect_base(res))
        i+= 2
    for i, ct in enumerate(repertoire):
        print((series[2*i], series[2*i+1]), ct)
    CALIB_VAL = 118.6
    print(repertoire)
    i = 0
    while i < len(repertoire):
        dx, dy = (np.array(repertoire[i+1]) - np.array(repertoire[i])) / CALIB_VAL
        print((series[2*i], series[2*i+1], series[2*i+2], series[2*i+3]), dx, dy)
        i+=2


def expr2_visual(series, dossier, ns='img_{0}.png', visual=True, saveopt=""):
    folder = os.path.join(ROOT_DIR, dossier)
    ins = os.path.join(folder, ns)
    repertoire = []
    i = 0
    while i < len(series):
        try:
            ambi, laser = cv2.imread(ins.format(series[i]), 0), cv2.imread(ins.format(series[i+1]), 0)
            res = image_diff(laser, ambi)
        except:
            print(ins.format(series[i]))
        repertoire.append(center_detect_invar(res, folder_path="../{}/".format(dossier),
                                              img_name=ns.format(series[i]), visual=visual, saveopt=saveopt))
        i+= 2
    for i, ct in enumerate(repertoire):
        print((series[2*i], series[2*i+1]), ct)
    CALIB_VAL = 118.6
    print(repertoire)
    i = 0
    while i < len(repertoire):
        dx, dy = (np.array(repertoire[i+1]) - np.array(repertoire[i])) / CALIB_VAL
        print((series[2*i], series[2*i+1], series[2*i+2], series[2*i+3]), dx, dy)
        i+=2


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


def convergence_test_final(folder, ns, visual=False, tt='d', saveopt=""):
    # tt: a, l, k, g
    convergence = {}
    titres = []
    variations = []
    startNP = 59
    startP = 80
    endP = 193
    meas = os.path.join(ROOT_DIR, 'meas/')
    fwrite = open(os.path.join(meas, 'convergence_{}{}.csv'.format(tt, saveopt)), 'w')
    cwriter = csv.writer(fwrite)
    cwriter.writerow(['Image Number', 'Center X', 'Center Y', 'StdDev Horizontal', 'Std Dev Vertical'])
    loss_types = ['huber', 'soft_l1', 'arctan', 'linear', 'cauchy']
    krange = np.arange(-1, 8, 2)
    grange = np.arange(1, 20, 2)
    sl, sk, sg = tt == 'a' or tt == 'l', tt == 'a' or tt == 'k', tt == 'a' or tt == 'g'
    swt = lambda ran, cond: ran if cond else range(0)
    ls = range(np.prod([len(loss_types) if sl else 1, len(krange) if sk else 1, len(grange) if sg else 1]))
    print(ls)
    catchx, catchy = [], []
    if tt == 'a':
        for l in loss_types:
            for ks in krange:
                for gs in grange:
                    catch = [[], []]
                    com = [l, 'Sobel:{}'.format(ks), 'Gauss:{}'.format(gs)]
                    cwriter.writerow(com)
                    # Consistency Cycled
                    pv = 0
                    pvs = 0
                    rcount = 0
                    cvx = np.zeros(4)
                    cvy = np.zeros(4)
                    cond = visual and l == 'soft_l1' and ks == -1 and gs == 7
                    # print(g)
                    for i in range(startNP, startP):
                        img_name = ns.format(i)
                        fpath = os.path.join(ROOT_DIR, folder)
                        imgfile = "%s_{0}.png" % img_name
                        # FOR NULL ROW OR COLUMN, DO NOT COUNT THE STDDEV
                        try:
                            #x, y = center_detect(fpath + imgfile, 5, m=m) TODO: RESOLVE THE CHANGE
                            x, y = center_detect(fpath, imgfile.format(1), l=l, gk=gs, ks=ks, visual=cond, catch=catch)
                            #                             # PUT IN CSV
                            cwriter.writerow([str(i), str(x), str(y)])
                            # CONVERGENCE
                        except AttributeError:
                            print('No {0}'.format(fpath + imgfile))
                            pass
                    for i in range(startP, endP):
                        img_name = ns.format(i)
                        fpath = os.path.join(ROOT_DIR, folder)
                        imgfile = "%s_{0}.png" % img_name
                        # FOR NULL ROW OR COLUMN, DO NOT COUNT THE STDDEV
                        try:
                            #x, y = center_detect(fpath + imgfile, 5, m=m) TODO: RESOLVE THE CHANGE
                            x, y = center_detect(fpath, imgfile.format(1), l=l, gk=gs, ks=ks, visual=cond, catch=catch)
                            # PUT IN CSV
                            cwriter.writerow([str(i), str(x), str(y)])
                            # CONVERGENCE
                            # Record x, y, check rcount, refresh CONSISTENCY
                            cvx[rcount] = x
                            cvy[rcount] = y
                            rcount += 1
                            if rcount == 4:
                                pv += np.var(cvx) + np.var(cvy)
                                pvs += 1
                                rcount = 0
                                cvx = np.zeros(4)
                                cvy = np.zeros(4)
                        except AttributeError:
                            print('No {0}'.format(fpath + imgfile))
                            pass
                            # print(str(i), val)
                    msepv = sqrt(pv / pvs)
                    catchx.append(max(catch[0]))
                    catchy.append(max(catch[1]))
                    cvg = {'PicConsistency': msepv, 'Max_Reg_ErrX': catchx[-1], 'Max_Reg_ErrY': catchy[-1]}
                    cwriter.writerow(cvg.items())
                    convergence[com] = cvg
                    variations.append(msepv)
                    titres.append(com)
        print(convergence)
        fwrite.close()
        model_contrast_plot(titres, variations, catchx, catchy, meas, saveopt)

    elif tt == 'l':
        for k, l in enumerate(loss_types):
            catch = [[],[]]
            com = [l]
            cwriter.writerow(com)
            # Consistency Cycled
            pv = 0
            pvs = 0
            rcount = 0
            cvx = np.zeros(4)
            cvy = np.zeros(4)
            kl = int(not visual) + k
            # print(kl, k)
            # print(g)
            for i in range(startNP, startP):
                img_name = ns.format(i)
                fpath = os.path.join(ROOT_DIR, folder)
                imgfile = "%s_{0}.png" % img_name
                # FOR NULL ROW OR COLUMN, DO NOT COUNT THE STDDEV
                try:
                    # x, y = center_detect(fpath + imgfile, 5, m=m) TODO: RESOLVE THE CHANGE
                    x, y = center_detect(fpath, imgfile.format(1), l=l, visual=True if kl == 0 else False, catch=catch)
                    # PUT IN CSV
                    cwriter.writerow([str(i), str(x), str(y)])
                    # CONVERGENCE
                except AttributeError:
                    print('No {0}'.format(fpath + imgfile))
                    pass
            for i in range(startP, endP):
                img_name = ns.format(i)
                fpath = os.path.join(ROOT_DIR, folder)
                imgfile = "%s_{0}.png" % img_name
                # FOR NULL ROW OR COLUMN, DO NOT COUNT THE STDDEV
                try:
                    # x, y = center_detect(fpath + imgfile, 5, m=m) TODO: RESOLVE THE CHANGE
                    x, y = center_detect(fpath, imgfile.format(1), l=l, visual=True if kl == 0 else False, catch=catch)
                    # PUT IN CSV
                    cwriter.writerow([str(i), str(x), str(y)])
                    # CONVERGENCE
                    # Record x, y, check rcount, refresh CONSISTENCY
                    cvx[rcount] = x
                    cvy[rcount] = y
                    rcount += 1
                    if rcount == 4:
                        pv += np.var(cvx) + np.var(cvy)
                        pvs += 1
                        rcount = 0
                        cvx = np.zeros(4)
                        cvy = np.zeros(4)
                except AttributeError:
                    print('No {0}'.format(fpath + imgfile))
                    pass
                    # print(str(i), val)
            msepv = sqrt(pv / pvs)
            catchx.append(max(catch[0]))
            catchy.append(max(catch[1]))
            cvg = {'PicConsistency': msepv, 'Max_Reg_ErrX': catchx[-1], 'Max_Reg_ErrY': catchy[-1]}
            cwriter.writerow(cvg.items())
            convergence[l] = cvg
            variations.append(msepv)
            titres.append(com)
        print(convergence)
        fwrite.close()
        model_contrast_plot(titres, variations, catchx, catchy, meas, saveopt)

    elif tt == 'd':
        norms = ['NO', 'INPLACE', 'OUTPLACE']
        for d in ['y', 'z', 'old']:
            for n in [0, 1, 2]:
                catch = [[], []]
                com = ["debias_with_{}".format(d), norms[n]]
                cwriter.writerow(com)
                # Consistency Cycled
                pv = 0
                pvs = 0
                rcount = 0
                cvx = np.zeros(4)
                cvy = np.zeros(4)
                cond = visual
                # print(g)
                for i in range(startNP, startP):
                    img_name = ns.format(i)
                    fpath = os.path.join(ROOT_DIR, folder)
                    imgfile = "%s_{0}.png" % img_name
                    # FOR NULL ROW OR COLUMN, DO NOT COUNT THE STDDEV
                    try:
                        #x, y = center_detect(fpath + imgfile, 5, m=m) TODO: RESOLVE THE CHANGE
                        print(imgfile.format(1))
                        x, y = center_detect(fpath, imgfile.format(1), debias=d, norm=n, visual=cond, catch=catch,
                                             saveopt="{}_{}_{}".format(saveopt, d, n))
                        #                             # PUT IN CSV
                        cwriter.writerow([str(i), str(x), str(y)])
                        # CONVERGENCE
                    except AttributeError:
                        print('No {0}'.format(fpath + imgfile))
                        pass
                for i in range(startP, endP):
                    img_name = ns.format(i)
                    fpath = os.path.join(ROOT_DIR, folder)
                    imgfile = "%s_{0}.png" % img_name
                    # FOR NULL ROW OR COLUMN, DO NOT COUNT THE STDDEV
                    try:
                        #x, y = center_detect(fpath + imgfile, 5, m=m) TODO: RESOLVE THE CHANGE
                        x, y = center_detect(fpath, imgfile.format(1), debias=d, norm=n, visual=cond,
                                             catch=catch, saveopt="{}_{}".format(saveopt,d))
                        # PUT IN CSV
                        cwriter.writerow([str(i), str(x), str(y)])
                        # CONVERGENCE
                        # Record x, y, check rcount, refresh CONSISTENCY
                        cvx[rcount] = x
                        cvy[rcount] = y
                        rcount += 1
                        if rcount == 4:
                            pv += np.var(cvx) + np.var(cvy)
                            pvs += 1
                            rcount = 0
                            cvx = np.zeros(4)
                            cvy = np.zeros(4)
                    except AttributeError:
                        print('No {0}'.format(fpath + imgfile))
                        pass
                        # print(str(i), val)
                msepv = sqrt(pv / pvs)
                catchx.append(max(catch[0]))
                catchy.append(max(catch[1]))
                cvg = {'PicConsistency': msepv, 'Max_Reg_ErrX': catchx[-1], 'Max_Reg_ErrY': catchy[-1]}
                cwriter.writerow(cvg.items())
                convergence[tuple(com)] = cvg
                variations.append(msepv)
                titres.append(com)
        print(convergence)
        fwrite.close()
        model_contrast_plot(titres, variations, catchx, catchy, meas, saveopt)


def path_prefix_free(path, symbol='/'):
    if path[-len(symbol):] == symbol:
        return path[path.rfind(symbol,0, -len(symbol))+len(symbol):-len(symbol)]
    else:
        return path[path.rfind(symbol)+len(symbol):]


def log_output(inpath, outpath, ns, series, invar=False, visual=False, saveopt=""):
    """ Runs center_detect with debug mode for all images in inpath+ns.format(irange), and output the test results
    in outpath/meas
    ns: str, *{}*.png, e.g. img_{}_1.png
    """
    imgseq = os.path.join(inpath, ns)
    fn = lambda p, n: p[p.find(n) + len(n) + 1:]
    fn2 = lambda segs: segs[-1] if segs[-1] else segs[-2]
    fwrite = open(os.path.join(outpath, 'meas/centerLog_{}_seriesLen{}_{}.csv'.format(path_prefix_free(inpath),
                                                                                      len(series), saveopt)), 'w+')
    cwriter = csv.writer(fwrite)
    cwriter.writerow(['img', 'x', 'y', 'HOPT', 'VOPT', 'SLD_H_0', 'SLD_H_1', 'MLD_H_0', 'MLD_H_1',
                      'SLD_V_0', 'SLD_V_1', 'MLD_V_0', 'MLD_V_1',
                      'SLD_A_0', 'SLD_A_1', 'MLD_A_0', 'MLD_A_1'
                      'EVS_h', 'r2_h', 'MAE_h', 'MSE_h', 'MEDAE_h',
                      'EVS_v', 'r2_v', 'MAE_v', 'MSE_v', 'MEDAE_v',
                      'EVS_t', 'r2_t', 'MAE_t', 'MSE_t', 'MEDAE_t'])
    i = 0
    while i < len(series):
        try:
            if invar:
                ambi, laser = cv2.imread(imgseq.format(series[i]), 0), cv2.imread(imgseq.format(series[i + 1]), 0)
                res_img = image_diff(laser, ambi)
                id = "{}_{}".format(series[i], series[i + 1])
                res = center_detect_base(res_img, inpath, ns.format(id), stat=True, visual=visual)
                i+=2
            else:
                res = center_detect(inpath, ns.format(series[i]), stat=True, visual=visual)
                id = str(series[i])
                i+=1
            cwriter.writerow([id, res[0], res[1]] + (list(res[2]) if res[2] else []))
        except AttributeError:
            print('No {0}'.format(imgseq.format(series[i])))
            i += 2 if invar else 1
    fwrite.close()


if __name__ == '__main__':
    #print(center_detect('../calib4/img_59_{0}.png', 5))
    #convergence_test_final('calib4/', 'img_{0}')
    #repertoire = expr2([10, 11, 15, 14, 18, 19, 22, 23])
    import os
    ROOT_DIR = '/Users/albertqu/Documents/7.Research/PEER Research/data'
    #repertoire = expr2_visual([14, 11, 13, 12], 'camera_tests', ns='{0}.png', visual=True, saveopt="_imgdiff_change")
    imgn = 'img_{0}_1.png'
    imgn_mul = 'img_{0}'
    test = os.path.join(ROOT_DIR, 'test1115/')
    calib4 = os.path.join(ROOT_DIR, 'calib4/')
    lab_series2 = os.path.join(ROOT_DIR, 'lab_series2/')
    metal_enclos = os.path.join(ROOT_DIR, 'metal_enclos/')
    #convergence_test_final(calib4, imgn_mul, visual=True, tt='d', saveopt="_debias")
    #print(calib4)
    #log_output(calib4, ROOT_DIR, imgn, np.arange(59, 193), saveopt='METRICS_z')
    log_output(lab_series2, ROOT_DIR, "{}.png", np.arange(1, 11), invar=True, visual=True, saveopt='METRICS_z')
    #log_output(metal_enclos, ROOT_DIR, "{}_2.png", np.arange(1, 11), invar=True, visual=True, saveopt='METRICS_z')
    # skewed = os.path.join(ROOT_DIR, 'skewed/')
    # bright = os.path.join(ROOT_DIR, 'bright/')
    #center_detect('../calib4/', 'img_113_1.png', visual=True)

    """
    for i in range(1, 13):
        ns = 'img_{0}.png'.format(i)
        try:
            print(center_detect(test, ns))
        except:
            print('NON IDEE')"""
    """
    for i in range(90, 96):
        ns = imgn.format(i)
        print(skewed + ns)
        print(center_detect(skewed, ns, visual=True))

    ns = imgn.format(81)
    print(bright + ns)
    try:
        print(center_detect(bright, ns, visual=True))
    except:
        print('NON IDEE')

    for i in range(193, 195):
        ns = imgn.format(i)
        print(bright + ns)
        try:
            print(center_detect(bright, ns, visual=True))
        except:
            print('NON IDEE')"""





