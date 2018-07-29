import cv2
import numpy as np
from scipy.optimize import curve_fit
import os
import csv
from datetime import datetime
from math import atan, sqrt, tan, radians, acos, asin, degrees, sin, cos

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
    def __init__(self, rho_rad=0, theta=0, x=None, data=None):
        if data is not None and x is not None:
            self.reg(x, data)
        else:
            self._r = degrees(rho_rad)
            self._t = theta
            self._s = sin(rho_rad)
            self._c = cos(rho_rad)

    def reg(self, x, data):
        """D = hough_data_matrix(x, data)
        print(D)
        y = np.zeros(len(data))
        cosine, sine, self._r = leastSquares(D, y)
        self._t = angle_interp(sine, cosine)
        self._s = sine
        self._c = cosine"""
        x1, x2 = x[0], x[-1]
        y1, y2 = data[0], data[-1]
        theta0 = theta_pred(x1, y1, x2, y2)
        p0 = [theta0, x1 * cos(theta0) + y1 * sin(theta0)]
        pm, vm = curve_fit(hough_line, x, data, p0=p0)
        if pm[1] < 0:
            pm[1] = -pm[1]
            pm[0] -= HC
            # pm[0] = 2 * np.pi
        angle = normalize_angle(pm[0])
        self._t = angle
        self._r = pm[1]
        self._s = sin(angle)
        self._c = cos(angle)

    def extract_points(self, x_input):
        x1 = int(x_input[0])
        x2 = int(x_input[-1])
        y1 = int(self.fit_x(x1))
        y2 = int(self.fit_x(x2))
        return (x1, y1), (x2, y2)

    def fit_x(self, x):
        # print(self._r, self._t)
        return (self._r - x * self._c) / self._s

    def __str__(self):
        return 'hough line with cos:{0}, sin:{1}, rho:{2}, theta:{3}'.format(self._c, self._s,
                                                                             self._r, degrees(self._t))

    @staticmethod
    def intersect(l1, l2):
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
    """
    if QC < res < HC:
        return res + HC
    elif HC <= res <= TQC:
        return res - HC
    else:
        return res"""


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
    return (rho - x * cos(theta)) / sin(theta)


""" ===================================
========== MATRIX-ARRAY UTIL ==========
======================================= """


class FastDataMatrix2D:
    # TODO: HANDLE THE CASE WHEN IT WRAPS 1D

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
        assert 0 <= start < end <= self._data.shape[self._ax], "start: {0}, end: {1}, ax: {2}, index: {3}".format(start,
                                                                                                                  end,
                                                                                                                  self._ax,
                                                                                                                  self._index)
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
        return self._data[self._index, item + self.start] if self._ax == FastDataMatrix2D.HOR \
            else self._data[item + self.start, self._index]

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


# Generic Regression Tools 128-133
def leastSquares(D, y):
    return np.linalg.lstsq(D, y, rcond=None)[0]


def MSE(y, y_hat, N):
    return np.linalg.norm(y - y_hat) ** 2 / N


def std_dev(data):
    std_dev = 0
    len_data = len(data)
    mean = sum(data) / len_data
    for i in range(len_data):
        std_dev += (data[i] - mean) ** 2
    return sqrt(std_dev / len_data)


# Polynomial Regression Tools 137-171
def poly_data_matrix(input_data, degree):
    # degree is the degree of the polynomial you plan to fit the data with
    Data = np.zeros((len(input_data), degree + 1))

    for k in range(0, degree + 1):
        Data[:, k] = (list(map(lambda x: x ** k, input_data)))

    return Data


def poly_error(params, D_a, y_a):
    '''degree=len(params)-1
    y=x_a*0

    for k in range(0,degree+1):
        coeff=params[k]
        y=y+list(map(lambda z:coeff*z**k,x_a))'''
    y = np.dot(D_a, params)
    return np.linalg.norm(y - y_a) ** 2


def improvedCost(x, y, x_test, y_test, start, end):
    """Given a set of x and y points training points,
    this function calculates polynomial approximations of varying
    degrees from start to end. Then it returns the cost, with
    the polynomial tested on test points of each degree in an array"""
    c = []
    degrees = range(start, end)
    ps = []
    for degree in degrees:
        # YOUR CODE HERE
        D = poly_data_matrix(x, degree)
        p = leastSquares(D, y)
        ps.append(p)
        D_t = poly_data_matrix(x_test, degree)
        y_hat = np.dot(D_t, p)
        c.append(MSE(y_test, y_hat, len(x_test)))
    return np.array(degrees), ps, np.array(c)


# Gaussian Regression Tools 184-210
def gauss_data_matrix(data):
    return np.array([[1, x ** 2, x, 1] for x in data])


def gauss_reg(x, y, p0):
    """Given a set of x and y training points, this function
    calculates the gaussian approximation function"""
    """skip_x = np.array([x[i] for i in range(len(y)) if y[i] != 0])
    y_prime = np.array([y[i] for i in range(len(y)) if y[i] != 0])
    logy = np.log(y_prime)

    sol = leastSquares(gauss_data_matrix(skip_x), logy)
    alp, bet, gam, lam = sol
    a = np.e ** alp
    c_s = - 1 / (2 * bet)
    b = gam * c_s
    maxi, mini = max_min(y)
    pred = gauss_hat(maxi, a, b, c_s)
    real = y[maxi]
    a = real / pred // 1000000000 * 10"""
    param, vm = curve_fit(gauss_hat, x, y, p0=p0)
    return param


def gauss_mat(shape, a, b1, c_s1, b2, c_s2):
    mat = np.zeros(shape)
    for r in range(mat.shape[0]):
        for c in range(mat.shape[1]):
            mat[r, c] = gauss_2d(c, r, a, b1, c_s1, b2, c_s2)

    return mat


# Hough Transform Regression Tools
def hough_data_matrix(x, y):
    dim = len(x)
    mat = np.full((dim, 3), -1.0)
    for i in range(dim):
        mat[i][0] = x[i]
        mat[i][1] = y[i]
    return mat


""" =======================================
========= EDGE DETECTION UTILS ============
=========================================== """


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


def edge_max_min(data):
    # Returns a *safe* edge maxi, mini for the data
    # TODO: OPTIMIZE THE WIDTH AND VALUE THRESHOLD
    width_thres = 120
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


def min_max(data, max, min):
    """ Converts the data to min max space. """
    g_diff = max - min
    return [(d - min) / g_diff for d in data]


def root_finding(x1, x2, y1, y2):
    """Given two points on a line, finds its zero crossing root."""
    return - y1 * (x2 - x1) / (y2 - y1) + x1


def check_cross(a, b):
    # Checks whether two points are of opposite signs
    return a * b < 0


def round_up(num):
    down = int(num)
    if num - down > 0:
        return num + 1
    else:
        return num


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
def edge_centroid(data, img_data, padding=10):
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
def poly_fitting(data, img_data, padding=10):
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
        y = np.array(img_data[start:end])
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
    width_thres = 120
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
        y = np.array(img_data[start:end])
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


# GAUSSIAN FITTING
def gaussian_fitting(data, img_data, padding=10):
    # TODO: OPTIMIZE THE AWKWARD TYPE CHECKING, ALONG WITH THE WIDTH THRESHOLD
    try:
        maxi, mini = edge_max_min(data)
    except AssertionError:
        return -1
    start, end = smart_interval(maxi - padding, mini + padding + 1, data)
    x = np.array(range(start, end))
    if type(img_data) == FastDataMatrix2D:
        img_data.segmentize(start, end)
        idata = img_data.extract_array()
    else:
        idata = np.array(img_data[start:end])
    try:
        param = gauss_reg(x, idata, p0=[10, (maxi + mini) / 2, std_dev(idata)])
    except RuntimeError:
        return -1
    return param[1]


def gaussian_fitting_params(data, img_data, padding=10):
    # TODO: OPTIMIZE THE AWKWARD TYPE CHECKING, ALONG WITH THE WIDTH THRESHOLD
    maxi, mini = max_min(data)
    width_thres = 120
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
        idata = np.array(img_data[start:end])
    return gauss_reg(x, idata, p0=[10, (maxi + mini) / 2, std_dev(idata)]), x


"""======================================
======== IMAGE PROCESSING UTIL ==========
========================================= """


def gauss_bg_deduce(x, img_data):
    # TODO: OPTIMIZE PERFORMANCE
    idata = img_data.extract_array()
    p0 = [1, len(img_data) / 2, std_dev(idata)]
    a, b, c_s = gauss_reg(x, idata, p0=p0)
    rem_gauss = gauss_hat(x, a, b, c_s)
    new_y = idata - rem_gauss
    return x, rem_gauss, new_y


def sobel_process(imgr, gks, sig):
    gksize = (gks, gks)
    sigmaX = sig
    blur = cv2.GaussianBlur(imgr, gksize, sigmaX)
    # cv2.imshow("blurred", blur)                        # REMOVES TO SHOW BLURRED IMAGE
    if len(imgr.shape) == 3:
        img = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    else:
        img = blur
    # GAUSSIAN PROCESSING                                 # UNCOMMENT TO SHOW GAUSSIAN PROCESSING
    """x = np.array(range(dimc))
    #y = np.array([img_rec.rel_lumin(img, 0, c) for c in x])
    y = np.array([img.item(dimr // 2, c) for c in x])
    a1, b1, c_s1 = gauss_reg(x, y)

    x2 = np.array(range(dimr))
    y2 = np.array([img.item(r, dimc // 2) for r in x2])
    a2, b2, c_s2 = gauss_reg(x2, y2)
    rem_gauss = gauss_mat(img.shape, (a1+a2) / 2, b1, c_s1, b2, c_s2)
    img = img - rem_gauss
    cv2.imshow("denoise", img)
    y_hat = gauss_hat(x, a1, b1, c_s1)
    plt.figure(figsize=(16, 8))
    plt.plot(x, y, 'b-', x, y_hat, 'r-')
    plt.show()
    plt.close()
    plt.figure(figsize=(16,8))
    plt.plot(x, y-y_hat, 'b-')
    plt.show()"""

    # IMAGE PROCESSING
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=-1)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=-1)
    return img, sobelx, sobely



def edge_detect_expr(edges, original):
    """
    TODO: 1. SIMPLE Approach: pairwise mask over original image, then compute total
          2. Traverse the image to mark the image.
    """
    se = 0
    row_sum = 0
    col_sum = 0
    tot_sum = 0
    for i in range(edges.shape[0]):
        for j in range(edges.shape[1]):
            if edges[i][j] > 0:
                row_sum += i * original[i][j]
                col_sum += j * original[i][j]
                tot_sum += original[i][j]
                se += 1
    print(se)
    if tot_sum != 0:
        print(row_sum / tot_sum)
        print(col_sum / tot_sum)


def hori(img, i, j, d):
    dest = j + d
    if dest < 0 or dest >= img.shape[1]:
        return img.item(i, j)
    return img.item(i, dest)


def verti(img, i, j, d):
    dest = i + d
    if dest < 0 or dest >= img.shape[0]:
        return img.item(i, j)
    return img.item(dest, j)


def test_blur_then_nr(iname):
    imgs = iname
    numIMG = 5
    imgr = None
    for i in range(1, numIMG + 1):
        target = cv2.imread(imgs.format(i), 0)
        target = cv2.GaussianBlur(target, (9, 9), 0)
        if i == 1:
            imgr = np.uint16(target)
        else:
            imgr = imgr + target
    return imgr / numIMG


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


def img_add(dest, src):
    """Destructive!"""
    row, col = dest.shape
    for i in range(row):
        for j in range(col):
            dest[i][j] += src[i][j]




""" ===================================
    ====== DATA RECORDING HELPER ======
    =================================== """


# 1. Try Noise Reduction:
# ---- 1.1 Reduction By Averaging, without FastNIMEANS  ~~ GOOD
# ---- 1.2 Reduction By FastNIMEANS                     ~~ INFERIOR
# ---- 1.3 Reduction By Combination                     ~~ INFERIOR
# ---- 1 NOTE: USE PLOTTING OF ORIGINAL GRAPH TO SEE RESULT
# 2. Try Di-Axial Scharr Kernel
# ---- 2.1 Add To Name Scheme, record graphs and compare them
# ---- 2 NOTE: GRAPH TO SAVE: 4-CONTRAST, TRANSITION PLOTS
# 3. Optimizing Data Processing:
# ---- 3.1 Marking point as edge maxima and minima according to STD Dev values
# ---- 3.2 Taking Sequential Samples
# ---- 3.3 Explore Thresholding Techniques


def register(img):
    lval = []
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img.item(i, j) > 250:
                lval.append((i, j, img.item(i, j)))
    print(lval)


""" TEST GAUSSIAN BLUR'S influence on noise distribution:
    Method:
        1. Preliminary
            1. Generate a random noise matrix with n * n, change shape to a vector
            2. Gaussian blur applied, change shape to a vector
            3. Compare noise level
        2. Secondary
            1. Generate a Gaussian / Cosine Matrix and add random noise matrix
            2. Gaussian blur applied
            3. Compare a randomly selected row of each one
"""






FM = FastDataMatrix2D


def folder_to_imgs(img_name_scheme, num_sample):
    """This function takes img files and return cv imgs"""
    return [cv2.imread(img_name_scheme.format(i)) for i in range(1, num_sample + 1)]


def center_detect(img_name_scheme, num_sample, sample_int=50, debug=False, gk=9, ks=-1, m=0, p=10, b=1, c=0):
    """This function takes in a list of images and output x, y [pixel] coordinates of the center of the cross hair
    hs: HORIZONTAL SLICE!  vs: VERTICAL SLICE!"""
    timeb = datetime.now()
    imgr = test_noise_reduce(img_name_scheme, num_sample)[0]
    dimr = imgr.shape[0]
    dimc = imgr.shape[1]
    # Image Processing
    gksize = (gk, gk)
    sigmaX = 0
    img = cv2.GaussianBlur(imgr, gksize, sigmaX)
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=ks)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=ks)
    # ------------------------------------------------------------
    # Parameter Setting
    METHODS = {0: gaussian_fitting, 1: poly_fitting, 2: edge_centroid,
               3: zero_crossing, 4: edge_converge_base, 5: edge_converge_extreme}
    edge_method = METHODS[m]
    nr = sample_int
    r_thresh = dimr / (sample_int * 3.0)
    nc = sample_int
    c_thresh = dimc / (sample_int * 3.0)
    # ------------------------------------------------------------
    # Gathering Data
    hs = []
    vs = []
    while nr < dimr:
        data_x = FM(sobelx, FM.HOR, nr)
        if m < 3:
            ec_x = edge_method(data_x, FM(img, FM.HOR, nr), p)
        else:
            ec_x = edge_method(data_x)
        nr += sample_int
        if ec_x == -1:
            continue
        else:
            hs.append((nr - sample_int, ec_x))

    while nc < dimc:
        data_y = FM(sobely, FM.VER, nc)
        if m < 3:
            ec_y = edge_method(data_y, FM(img, FM.VER, nc), p)
        else:
            ec_y = edge_method(data_y)
        nc += sample_int
        if ec_y == -1:
            continue
        else:
            vs.append((nc - sample_int, ec_y))
    len_hs = len(hs)
    len_vs = len(vs)

    # ----- Following Modules Handles Hough Line Drawing ---------
    hxs = np.zeros(len_hs)
    hys = np.zeros(len_hs)
    for i in range(len_hs):
        hxs[i] = hs[i][1]
        hys[i] = hs[i][0]
    vxs = np.zeros(len_vs)
    vys = np.zeros(len_vs)
    for i in range(len_vs):
        vxs[i] = vs[i][0]
        vys[i] = vs[i][1]
    # hough_img = img_name_scheme.format(1)
    # DATA RECORDING AND PROCESSING
    x_valid = False
    y_valid = False
    stdh = -1
    stdv = -1
    valuesH = [d[1] for d in hs]
    valuesV = [d[1] for d in vs]
    line_a = HoughLine(x=hxs, data=hys)
    line_b = HoughLine(x=vxs, data=vys)
    if len_hs >= r_thresh:
        x_valid = True
        stdh = std_dev(valuesH)
    if len_vs >= c_thresh:
        y_valid = True
        stdv = std_dev(valuesV)
    if c == 1:
        center_x = sum(valuesH) / len_hs if x_valid else -1
        center_y = sum(valuesV) / len_vs if y_valid else -1
    else:
        if x_valid and y_valid:
            center_x, center_y = HoughLine.intersect(line_a, line_b)
        else:
            center_x = sum(valuesH) / len_hs if x_valid else -1
            center_y = sum(valuesV) / len_vs if y_valid else -1
    timea = datetime.now()
    usage = timea - timeb
    print(usage)
    return center_x, center_y


if __name__ == '__main__':
    folder = "../calib4/"
    ns = folder + "img_%d_{0}.png"
    for i in range(59, 80):
        try:
            val = center_detect(ns % i, 5)
            print(val)
        except AttributeError:
            pass

