import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
#from img_rec_module import img_rec
from random import randint
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
        print(degrees(theta0))
        p0 = [theta0, x1 * cos(theta0) + y1 * sin(theta0)]
        pm, vm = curve_fit(hough_line, x, data, p0=p0)
        if pm[1] < 0:
            pm[1] = -pm[1]
            pm[0] -= HC
            #pm[0] = 2 * np.pi
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
        #print(self._r, self._t)
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
    return a * np.exp(- ((x - b1) ** 2  / (2 * c_s1) + (y - b2) ** 2) / (2 * c_s2))


def gauss_hat(x, a, b, c_s):
    return a * np.exp(- (x - b) ** 2 / (2 * c_s))


def gaussian_curve(x_input, a,  b, c_s):
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
        assert 0 <= start < end <= self._data.shape[self._ax], "start: {0}, end: {1}, ax: {2}, index: {3}".format(start, end, self._ax, self._index)
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
        assert 0 <= index < self.irange(), "start: {0}, end: {1}, ax: {2}, index: {3}".format(self.start, self.end, self._ax, index)
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
    return np.array(degrees), ps,  np.array(c)


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
    timeb = datetime.now()
    param, vm = curve_fit(gauss_hat, x, y, p0=p0)
    timea = datetime.now()
    print(timea - timeb)
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


def test_canny_detect():
    imgn = "img_89_{0}.png"
    IMGDIR = "../calib4/"
    NAME = IMGDIR + imgn

    while True:
        lowp = input("Type the Lower Bound or c to cancel:    ")
        if lowp == 'c':
            break
        while True:
            try:
                low = int(lowp)
                break
            except ValueError:
                lowp = input("Bad input Try again! ")
        highp = input("Type the Upper Bound or c to cancel:   ")
        if highp == 'c':
            break
        while True:
            try:
                high = int(highp)
                break
            except ValueError:
                highp = input("Bad input Try again! ")

        denoised, original = test_noise_reduce(NAME)
        print(get_center_val(denoised))
        NL_denoised = cv2.fastNlMeansDenoising(original)
        bdenoise = test_blur_then_nr(NAME)

        dblur, sobelx, sobely = sobel_process(denoised, 9, 0)
        nldblur = sobel_process(NL_denoised, 9, 0)[0]
        pblur, sobelx, sobely = sobel_process(original, 9, 0)

        de_edges = canny_detect(denoised, low, high)
        db_edges = canny_detect(dblur, low, high)
        nld_edges = canny_detect(nldblur, low, high)
        blur_denoise = canny_detect(bdenoise, low, high)
        blur_edges = canny_detect(pblur, low, high)
        #edge_detect_expr(db_edges, original)
        """compare_images((imgr, 'Original'), (edges, 'Canny Edge'),
                       (sobelx, 'Sobel X'), (sobely, 'Sobel Y'))"""
        compare_images((original, 'Original'), (NL_denoised, "NLMEANS"), (de_edges, 'DENOISE Edges'),
                       (blur_edges, 'Plain Blur Edges'), (denoised, 'Denoised'), (nld_edges, 'NIDE-BLUR Edges'),
                       (db_edges, 'DE-BLUR Edges'), (blur_denoise, "BDENOISE Edges"), color_map='gray')


def canny_detect(src, low=4, high=10):
    return cv2.Canny(np.uint8(src), 4, 10, L2gradient=True)


def extract_verge(edges, i, j, max_blank, dir, axis):
    """
    :param edges: img MAT resulting from canny edge procedure
    :param i: starting row i
    :param j: starting col j
    :param max_blank: maximum number of blanks to rule out the existence of a verge outside of the current verge (in which case would be a fake verge)
    :param dir: Directions of movement, 1 for right or down, -1 for left or up
    :param axis:
    :return: (r, c) index of the verge on the given 'axis' in the given 'dir' from the starting point (i, j)
    """


def significance_test(data, val):
    """Outputs whether the maximum or minimum value is a significant value."""
    return val in extract_extrema(data)[0] or val in extract_extrema(data)[1]


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
    for i in range(1, numIMG+1):
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
    ============ PLOTTING =============
    ============= HELPER ==============
    =================================== """


def quick_plot(data, xs=None):
    plt.figure()
    if xs:
        plt.plot(xs, data, 'bo-')
    else:
        plt.plot(data, 'bo-')
    plt.show()
    plt.close()


def compare_images(*args, ilist=None, color_map=None):
    # Takes in a sequence of IMG(Gray) and TITLE pairs and plot them side by side
    if ilist:
        args = ilist
    graph_amount = len(args)
    row = int(sqrt(graph_amount))
    col = round_up(float(graph_amount) / row)
    plt.figure(figsize=(16, 8))
    for i, pair in enumerate(args):
        plt.subplot(row, col, i+1)
        if color_map:
            plt.imshow(pair[0], cmap=color_map)
        plt.xticks([]), plt.yticks([])
        plt.title(pair[1])
    plt.show()


def compare_data_plots(*args, ilist=None, suptitle=None):
    # Takes in a sequence of tuples with data, xs, and title pairs
    if ilist:
        args = ilist
    graph_amount = len(args)
    row = int(sqrt(graph_amount))
    col = round_up(float(graph_amount) / row)
    plt.figure(figsize=(16, 8))
    if suptitle:
        plt.suptitle(suptitle, fontsize=14)
    for i, pair in enumerate(args):
        plt.subplot(row, col, i + 1)
        if len(pair) == 2:
            plt.plot(pair[0], 'bo-')
            plt.title(pair[1])
        else:
            plt.plot(pair[1], pair[0], 'bo-')
            plt.title(pair[2])
    plt.show()


def line_graph_contrast(img, xs, ys):
    line = HoughLine(x=xs, data=ys)
    #print(xs)
    x0 = line._c * line._r
    y0 = line._s * line._r

    print('x0:{0}, y0:{1}, cos:{2}, sin:{3}, rho:{4}, theta:{5}'.format(x0, y0, line._c, line._s, line._r, degrees(line._t)))
    x1 = int(x0 + 1000 * (-line._s))
    y1 = int(y0 + 1000 * (line._c))
    x2 = int(x0 - 1000 * (-line._s))
    y2 = int(y0 - 1000 * (line._c))
    p1, p2 = (x1, y1), (x2, y2)
    cv2.line(img, p1, p2, (255, 0, 0), 1)
    return p1, p2, line


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


def test_old():
    # SETTINGS
    imgn = "img_4_{0}"
    IMGDIR = "../new_test/"
    ROOTMEAS = "meas/"
    SAVEDIR = ROOTMEAS+imgn + "/"
    #imgr = cv2.imread(IMGDIR + imgn + ".png")
    imgr, ori = test_noise_reduce(IMGDIR + imgn + ".png")
    dimc = imgr.shape[1]
    dimr = imgr.shape[0]
    r_int = dimr // 2
    c_int = dimc // 2
    start = 7
    end = 10
    sig = 0
    ins = range(start, end, 2)
    name_scheme = imgn + ("_({0}, {1})").format(r_int, c_int)
    NAME_HEADER = SAVEDIR + name_scheme
    if not os.path.exists(SAVEDIR):
        os.mkdir(SAVEDIR)
    fwrite = open(NAME_HEADER + ".txt", "w")
    plot_save = NAME_HEADER + ".png"


    # INITIALIZATION
    cenvalsx = []
    cenvalsy = []
    edgvalsx = []
    edgvalsy = []
    ebvalsx = []
    ebvalsy = []
    ecvalsx = []
    ecvalsy = []
    ecbvalsx = []
    ecbvalsy = []
    #cv2.imshow("denoised", np.uint8(imgr))                                 # REMOVES TO SHOW IMAGE
    x = np.array(range(dimc))
    x2 = np.array(range(dimr))

    for i in ins:
        # INITIALIZATION
        gksize = (i, i)
        sigmaX = sig
        blur = cv2.GaussianBlur(imgr,gksize,sigmaX)
        #cv2.imshow("blurred", blur)                        # REMOVES TO SHOW BLURRED IMAGE
        if len(blur.shape) == 3:
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
        """kernel = np.array([[-1, -1, -1],
                           [-1, 8, -1],
                           [-1, -1, -1]])
        kerneled = cv2.filter2D(img, -1, kernel)
        cv2.imshow("kerneled", kerneled)"""                 # KERNEL EXPERIMENT
        #laplacian = cv2.Laplacian(img,cv2.CV_64F)          # LAPLACIAN FILTERING
        """plt.subplot(2,2,1),plt.imshow(cv2.cvtColor(imgr, cv2.COLOR_BGR2GRAY),cmap = 'gray')
        plt.title('Original'), plt.xticks([]), plt.yticks([])
        plt.subplot(2,2,2),plt.imshow(img,cmap = 'gray')
        plt.title('Gaussian Filter'), plt.xticks([]), plt.yticks([])
        plt.subplot(2,2,3),plt.imshow(sobelx,cmap = 'gray')
        plt.title('Scharr X'), plt.xticks([]), plt.yticks([])
        plt.subplot(2,2,4),plt.imshow(sobely,cmap = 'gray')
        plt.title('Scharr Y'), plt.xticks([]), plt.yticks([])"""
        #plt.savefig(NAME_HEADER + "filters.png")
        #plt.savefig(NAME_HEADER + "2D_filters.png")
        #plt.close()                                         # REMOVES TO SHOW CONTRAST OF FILTERS


        # DETECTION INIT
        gk_setting = "ksize:{0}, sigmax:{1}".format(gksize, sigmaX)
        y_s_x = np.array([sobelx.item(r_int, c) for c in x])
        y_s_y = np.array([sobely.item(r, c_int) for r in x2])
        imgx = [imgr.item(r_int, c) for c in x]
        imgy = [imgr.item(r, c_int) for r in x2]
        blurimgx = [img.item(r_int, c) for c in x]
        blurimgy = [img.item(r, c_int) for r in x2]
        print(gk_setting)
        print(extract_extrema(y_s_x))
        print(extract_extrema(y_s_y))
        fwrite.write(gk_setting)
        fwrite.write('\n')

        # EDGE DETECTION
        zx = zero_crossing(y_s_x)
        cenvalsx.append(zx)
        ex = edge_converge_extreme(y_s_x)
        edgvalsx.append(ex)
        ebx = edge_converge_base(y_s_x)
        ebvalsx.append(ebx)
        ecx = edge_centroid(y_s_x, imgx)
        ecvalsx.append(ecx)
        ecbx = edge_centroid(y_s_x, blurimgx)
        ecbvalsx.append(ecbx)

        zy = zero_crossing(y_s_y)
        cenvalsy.append(zy)
        ey = edge_converge_extreme(y_s_y)
        edgvalsy.append(ey)
        eby = edge_converge_base(y_s_y)
        ebvalsy.append(eby)
        ecy = edge_centroid(y_s_y, imgy)
        ecvalsy.append(ecy)
        ecby = edge_centroid(y_s_y, blurimgy)
        ecbvalsy.append(ecby)

        # DATA RECORDING
        fwrite.write("x: zero_crossing: {0}, edge_converge: {1}; ".format(zx, ex))
        fwrite.write('\n')
        fwrite.write("y: zero_crossing: {0}, edge_converge: {1}\n".format(zy, ey))

        # GK_PLOT
        """if i == 9:
            fig = plt.figure(figsize=(16, 8))
            fig.canvas.set_window_title(gk_setting)
            plt.subplot(211)
            plt.plot(y_s_x, 'b-')
            plt.ylabel("sobel_x")
            plt.subplot(212)
            plt.plot(y_s_y, 'b-')
            plt.ylabel("sobel_y")
            #plt.savefig(NAME_HEADER + "_" + gk_setting + ".png")"""

        #if i >= 13:
        fig = plt.figure(figsize=(16, 8))
        fig.canvas.set_window_title(gk_setting)
        plt.subplot(211)
        plt.plot(y_s_x, 'b-')
        plt.ylabel("sobel_x")
        plt.subplot(212)
        plt.plot(y_s_y, 'b-')
        plt.ylabel("sobel_y")
        plt.show()
        # plt.savefig(NAME_HEADER + "_" + gk_setting + ".png") # UNCOMMENT WHEN SAVING PLOTS

        # USING MIN-MAX
        fig = plt.figure(figsize=(16, 8))
        fig.canvas.set_window_title(gk_setting)
        plt.subplot(221)
        plt.plot(y_s_x, 'b-')
        plt.ylabel("sobel_x")
        plt.subplot(222)
        plt.plot(min_max(y_s_x, max(y_s_x), min(y_s_x)), 'b-')
        plt.subplot(223)
        plt.plot(y_s_y, 'b-')
        plt.ylabel("sobel_y")
        plt.xlabel("Plain")
        plt.subplot(224)
        plt.plot(min_max(y_s_y, max(y_s_y), min(y_s_y)), 'b-')
        plt.xlabel("MINMAX")
        plt.show()


    fig = plt.figure(figsize=(16, 8))
    plt.subplot(2, 1, 1)
    #plt.plot(ins, cenvalsx, 'b-', ins, edgvalsx, 'r-', ins, ebvalsx, 'g-', ins, ecvalsx, 'c-', ins, ecbvalsx, 'm-')
    #plt.legend(['zero crossing', 'edge_converging_extreme', 'edge_converging_base', 'edge_centroid', 'edge_centroid_blurred'], loc="lower left")
    #plt.plot(ins, cenvalsx, 'b-', ins, ebvalsx, 'g-', ins, ecvalsx, 'c-', ins, ecbvalsx, 'm-')
    #plt.legend(['zero crossing', 'edge_converging_base', 'edge_centroid', 'edge_centroid_blurred'],loc="lower left")
    plt.plot(ins, ebvalsx, 'g-', ins, ecvalsx, 'c-', ins, ecbvalsx, 'm-')
    plt.legend(['edge_converging_base', 'edge_centroid', 'edge_centroid_blurred'],loc="lower left")
    plt.ylabel("x_meas")
    plt.subplot(2, 1, 2)
    #plt.plot(ins, cenvalsy, 'b-', ins, ebvalsy, 'g-', ins, ecvalsy, 'c-', ins, ecbvalsy, 'm-')
    #plt.legend(['zero crossing', 'edge_converging_base', 'edge_centroid', 'edge_centroid_blurred'], loc="lower left")
    plt.plot(ins, ebvalsy, 'g-', ins, ecvalsy, 'c-', ins, ecbvalsy, 'm-')
    plt.legend(['edge_converging_base', 'edge_centroid', 'edge_centroid_blurred'], loc="lower left")
    plt.ylabel("y_meas")
    plt.xlabel("kernel size")
    plt.show()
    #fig.savefig(plot_save)
    #plt.savefig(plot_save)


def test_interactive():
    # TODO: TEST OUT FACET MODEL AND CONDITIONAL CONVOLUTIONAL KERNEL
    # SETTINGS
    while True:
        number = input("type the picture number u wanna test or c to cancel:    ")
        if number == 'c':
            break
        imgn = "img_%d_{0}" % int(number)
        print(imgn)
        IMGDIR = "../testpic/"
        IMGDIR = "../calib4/"
        ROOTMEAS = "meas/"
        SAVEDIR = ROOTMEAS + imgn + "/"
        # imgr = cv2.imread(IMGDIR + imgn + ".png")

        imgr, ori = test_noise_reduce(IMGDIR + imgn + ".png", numIMG=5)
        dimc = imgr.shape[1]
        dimr = imgr.shape[0]
        r_prompt = input("Type in the index for horizontal slice: (range: [0, {0}])   ".format(dimr - 1)) #dimr // 2
        c_prompt = input("Type in the index for vertical slice: (range: [0, {0}])   ".format(dimc - 1)) #dimc // 2
        while True:
            try:
                r_int = int(r_prompt)
                assert 0 <= r_int < dimr
                break
            except ValueError:
                r_prompt = input("Bad Input, try again: (range: [0, {0}])    ".format(dimr - 1))  # dimr // 2
            except AssertionError:
                r_prompt = input("Out of Bound, try again: (range: [0, {0}])    ".format(dimr - 1))
        while True:
            try:
                c_int = int(c_prompt)
                assert 0 <= c_int < dimc
                break
            except ValueError:
                c_prompt = input("Bad Input, try again: (range: [0, {0}])    ".format(dimc - 1))  # dimc // 2
            except AssertionError:
                c_prompt = input("Out of Bound, try again: (range: [0, {0}])    ".format(dimc - 1))  # dimc // 2
        #start = 7
        #end = 10
        sig = 0
        gk = 9
        name_scheme = imgn + ("_({0}, {1})").format(r_int, c_int)
        NAME_HEADER = SAVEDIR + name_scheme
        if not os.path.exists(SAVEDIR):
            os.mkdir(SAVEDIR)
        fwrite = open(NAME_HEADER + ".txt", "w")
        plot_save = NAME_HEADER + ".png"

        # INITIALIZATION
        # cv2.imshow("denoised", np.uint8(imgr))                                 # REMOVES TO SHOW IMAGE
        #x = np.array(range(dimc))
        #x2 = np.array(range(dimr))

        # INITIALIZATION
        gksize = (gk, gk)
        sigmaX = sig
        blur = cv2.GaussianBlur(imgr, gksize, sigmaX)
        # cv2.imshow("blurred", blur)                        # REMOVES TO SHOW BLURRED IMAGE
        if len(blur.shape) == 3:
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
        """kernel = np.array([[-1, -1, -1],
                           [-1, 8, -1],
                           [-1, -1, -1]])
        kerneled = cv2.filter2D(img, -1, kernel)
        cv2.imshow("kerneled", kerneled)"""  # KERNEL EXPERIMENT
        # laplacian = cv2.Laplacian(img,cv2.CV_64F)          # LAPLACIAN FILTERING
        """plt.subplot(2,2,1),plt.imshow(cv2.cvtColor(imgr, cv2.COLOR_BGR2GRAY),cmap = 'gray')
        plt.title('Original'), plt.xticks([]), plt.yticks([])
        plt.subplot(2,2,2),plt.imshow(img,cmap = 'gray')
        plt.title('Gaussian Filter'), plt.xticks([]), plt.yticks([])
        plt.subplot(2,2,3),plt.imshow(sobelx,cmap = 'gray')
        plt.title('Scharr X'), plt.xticks([]), plt.yticks([])
        plt.subplot(2,2,4),plt.imshow(sobely,cmap = 'gray')
        plt.title('Scharr Y'), plt.xticks([]), plt.yticks([])"""
        # plt.savefig(NAME_HEADER + "filters.png")
        # plt.savefig(NAME_HEADER + "2D_filters.png")
        # plt.close()                                         # REMOVES TO SHOW CONTRAST OF FILTERS


        # DETECTION INIT
        xh = np.array(range(dimc))
        xv = np.array(range(dimr))
        zeroh = np.zeros(dimc)
        zerov = np.zeros(dimr)
        y_s_x = FM(sobelx, FM.HOR, r_int)
        y_s_y = FM(sobely, FM.VER, c_int)
        imgx = FM(imgr, FM.HOR, r_int)
        imgy = FM(imgr, FM.VER, c_int)
        blurimgx = FM(img, FM.HOR, r_int)
        blurimgy = FM(img, FM.VER, c_int)
        gx, bgx, deducedx = gauss_bg_deduce(xh, imgx)
        gy, bgy, deducedy = gauss_bg_deduce(xv, imgy)

        #print(extract_extrema(y_s_x))
        #print(extract_extrema(y_s_y))

        # EDGE DETECTION
        zx = zero_crossing(y_s_x)
        ex = edge_converge_extreme(y_s_x)
        ebx = edge_converge_base(y_s_x)
        ecbx = edge_centroid(y_s_x, blurimgx)
        ecx = edge_centroid(y_s_x, imgx)
        try:
            #paramhp, xp1 = poly_fitting_params(y_s_x, deducedx)
            #paramhg, xg1 = gaussian_fitting_params(y_s_x, deducedx)
            paramhp, xp1 = poly_fitting_params(y_s_x, imgx)
            paramhg, xg1 = gaussian_fitting_params(y_s_x, imgx)
            xp_h_plot, yp_h_plot = poly_curve(paramhp, xp1)
            xg_h_plot, yg_h_plot = gaussian_curve(xg1, paramhg[0], paramhg[1], paramhg[2])
        except AssertionError:
            print("No image data for horizontal slice!")
            xp_h_plot, yp_h_plot = xh, zeroh
            xg_h_plot, yg_h_plot = xh, zeroh

        px, gx = xp_h_plot[np.argmax(yp_h_plot)], xg_h_plot[np.argmax(yg_h_plot)]
        print("edge: {0}, poly:{1}, gaussian:{2}".format(ecx, px, gx))

        zy = zero_crossing(y_s_y)
        ey = edge_converge_extreme(y_s_y)
        eby = edge_converge_base(y_s_y)
        ecby = edge_centroid(y_s_y, blurimgy)
        ecy = edge_centroid(y_s_y, imgy)
        try:
            paramvp, xp2 = poly_fitting_params(y_s_y, imgy)
            paramvg, xg2 = gaussian_fitting_params(y_s_y, imgy)
            xp_v_plot, yp_v_plot = poly_curve(paramvp, xp2)
            xg_v_plot, yg_v_plot = gaussian_curve(xg2, paramvg[0], paramvg[1], paramvg[2])
        except AssertionError:
            print("No image data for vertical slice!")
            xp_v_plot, yp_v_plot = xv, zerov
            xg_v_plot, yg_v_plot = xv, zerov
        py, gy = xp_v_plot[np.argmax(yp_v_plot)], xg_v_plot[np.argmax(yg_v_plot)]
        print("edge: {0} poly:{1}, gaussian:{2}".format(ecy, py, gy))


        # DATA RECORDING
        fwrite.write("x: edge_centroid: {0}, poly: {1}, gaussian: {2}; ".format(ecx, px, gx))
        fwrite.write('\n')
        fwrite.write("y: edge_centroid: {0}, poly: {1}, gaussian: {2}\n".format(ecy, py, gy))

        # GK_PLOT
        """if i == 9:
            fig = plt.figure(figsize=(16, 8))
            fig.canvas.set_window_title(gk_setting)
            plt.subplot(211)
            plt.plot(y_s_x, 'b-')
            plt.ylabel("sobel_x")
            plt.subplot(212)
            plt.plot(y_s_y, 'b-')
            plt.ylabel("sobel_y")
            #plt.savefig(NAME_HEADER + "_" + gk_setting + ".png")"""

        # if i >= 13:
        fig = plt.figure(figsize=(16, 8))
        plt.subplot(211)
        imgx.initialize()
        plt.plot(xh, imgx.extract_array(), 'b-', xh, bgx, 'm-', xh, deducedx, 'c-', xp_h_plot, yp_h_plot, 'g-', xg_h_plot, yg_h_plot, 'r-')
        plt.ylabel("Horizontal Slice, ECX: {0}".format(ecx))
        plt.legend(['Raw image data', 'gaussian background', 'remnant', 'polynomial', 'gaussian'], loc="upper right")
        plt.subplot(212)
        imgy.initialize()
        plt.plot(xv, imgy.extract_array(), 'b-', xv, bgy, 'm-', xv, deducedy, 'c-', xp_v_plot, yp_v_plot, 'g-', xg_v_plot, yg_v_plot, 'r-')
        plt.ylabel("Vertical Slice, ECY: {0}".format(ecy))
        plt.legend(['Raw image data', 'gaussian background', 'remnant', 'polynomial', 'gaussian'], loc="upper right")
        plt.show()
        # plt.savefig(NAME_HEADER + "_" + gk_setting + ".png") # UNCOMMENT WHEN SAVING PLOTS

        """fig = plt.figure(figsize=(16, 8))
        plt.subplot(2, 1, 1)"""
        # plt.plot(ins, cenvalsx, 'b-', ins, edgvalsx, 'r-', ins, ebvalsx, 'g-', ins, ecvalsx, 'c-', ins, ecbvalsx, 'm-')
        # plt.legend(['zero crossing', 'edge_converging_extreme', 'edge_converging_base', 'edge_centroid', 'edge_centroid_blurred'], loc="lower left")
        # plt.plot(ins, cenvalsx, 'b-', ins, ebvalsx, 'g-', ins, ecvalsx, 'c-', ins, ecbvalsx, 'm-')
        # plt.legend(['zero crossing', 'edge_converging_base', 'edge_centroid', 'edge_centroid_blurred'],loc="lower left")
        """plt.plot(ins, ebvalsx, 'g-', ins, ecvalsx, 'c-', ins, ecbvalsx, 'm-')
        plt.legend(['edge_converging_base', 'edge_centroid', 'edge_centroid_blurred'], loc="lower left")
        plt.ylabel("x_meas")
        plt.subplot(2, 1, 2)
        # plt.plot(ins, cenvalsy, 'b-', ins, ebvalsy, 'g-', ins, ecvalsy, 'c-', ins, ecbvalsy, 'm-')
        # plt.legend(['zero crossing', 'edge_converging_base', 'edge_centroid', 'edge_centroid_blurred'], loc="lower left")
        plt.plot(ins, ebvalsy, 'g-', ins, ecvalsy, 'c-', ins, ecbvalsy, 'm-')
        plt.legend(['edge_converging_base', 'edge_centroid', 'edge_centroid_blurred'], loc="lower left")
        plt.ylabel("y_meas")
        plt.xlabel("kernel size")
        plt.show()
        # fig.savefig(plot_save)
        # plt.savefig(plot_save)"""


def test(folder, imgn):
    # TODO: TEST OUT FACET MODEL AND CONDITIONAL CONVOLUTIONAL KERNEL
    # SETTINGS
    ROOTMEAS = "meas/"
    SAVEDIR = ROOTMEAS + imgn[:-4] + "/"
    # imgr = cv2.imread(IMGDIR + imgn + ".png")

    imgr, ori = test_noise_reduce(folder + imgn, numIMG=5)
    dimc = imgr.shape[1]
    dimr = imgr.shape[0]
    while True:
        r_prompt = input("Type in the index for horizontal slice or c to cancel: (range: [0, {0}])   ".format(dimr - 1))  # dimr // 2
        c_prompt = input("Type in the index for vertical slice: (range: [0, {0}])   ".format(dimc - 1))  # dimc // 2
        if r_prompt == 'c':
            break
        while True:
            try:
                r_int = int(r_prompt)
                assert 0 <= r_int < dimr
                break
            except ValueError:
                r_prompt = input("Bad Input, try again: (range: [0, {0}])    ".format(dimr - 1))  # dimr // 2
            except AssertionError:
                r_prompt = input("Out of Bound, try again: (range: [0, {0}])    ".format(dimr - 1))
        while True:
            try:
                c_int = int(c_prompt)
                assert 0 <= c_int < dimc
                break
            except ValueError:
                c_prompt = input("Bad Input, try again: (range: [0, {0}])    ".format(dimc - 1))  # dimc // 2
            except AssertionError:
                c_prompt = input("Out of Bound, try again: (range: [0, {0}])    ".format(dimc - 1))  # dimc // 2
        # start = 7
        # end = 10
        sig = 0
        gk = 9
        name_scheme = imgn + ("_({0}, {1})").format(r_int, c_int)
        NAME_HEADER = SAVEDIR + name_scheme
        if not os.path.exists(SAVEDIR):
            os.mkdir(SAVEDIR)
        fwrite = open(NAME_HEADER + ".txt", "w")
        plot_save = NAME_HEADER + ".png"

        # INITIALIZATION
        # cv2.imshow("denoised", np.uint8(imgr))                                 # REMOVES TO SHOW IMAGE
        # x = np.array(range(dimc))
        # x2 = np.array(range(dimr))

        # INITIALIZATION
        gksize = (gk, gk)
        sigmaX = sig
        blur = cv2.GaussianBlur(imgr, gksize, sigmaX)
        # cv2.imshow("blurred", blur)                        # REMOVES TO SHOW BLURRED IMAGE
        if len(blur.shape) == 3:
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


        # DETECTION INIT
        xh = np.array(range(dimc))
        xv = np.array(range(dimr))
        zeroh = np.zeros(dimc)
        zerov = np.zeros(dimr)
        y_s_x = FM(sobelx, FM.HOR, r_int)
        y_s_y = FM(sobely, FM.VER, c_int)
        imgx = FM(imgr, FM.HOR, r_int)
        imgy = FM(imgr, FM.VER, c_int)
        blurimgx = FM(img, FM.HOR, r_int)
        blurimgy = FM(img, FM.VER, c_int)
        gx, bgx, deducedx = gauss_bg_deduce(xh, imgx)
        gy, bgy, deducedy = gauss_bg_deduce(xv, imgy)



        # print(extract_extrema(y_s_x))
        # print(extract_extrema(y_s_y))
        compare_data_plots((y_s_x.extract_array(), 'Sobel X'), (y_s_y.extract_array(), 'Sobel Y'))
        # EDGE DETECTION
        ecx = edge_centroid(y_s_x, imgx)
        try:
            # paramhp, xp1 = poly_fitting_params(y_s_x, deducedx)
            # paramhg, xg1 = gaussian_fitting_params(y_s_x, deducedx)
            paramhp, xp1 = poly_fitting_params(y_s_x, imgx)
            paramhg, xg1 = gaussian_fitting_params(y_s_x, imgx)
            xp_h_plot, yp_h_plot = poly_curve(paramhp, xp1)
            xg_h_plot, yg_h_plot = gaussian_curve(xg1, paramhg[0], paramhg[1], paramhg[2])
        except AssertionError:
            print("No image data for horizontal slice!")
            xp_h_plot, yp_h_plot = xh, zeroh
            xg_h_plot, yg_h_plot = xh, zeroh

        px, gx = xp_h_plot[np.argmax(yp_h_plot)], xg_h_plot[np.argmax(yg_h_plot)]
        print("edge: {0}, poly:{1}, gaussian:{2}".format(ecx, px, gx))

        ecy = edge_centroid(y_s_y, imgy)
        try:
            paramvp, xp2 = poly_fitting_params(y_s_y, imgy)
            paramvg, xg2 = gaussian_fitting_params(y_s_y, imgy)
            xp_v_plot, yp_v_plot = poly_curve(paramvp, xp2)
            xg_v_plot, yg_v_plot = gaussian_curve(xg2, paramvg[0], paramvg[1], paramvg[2])
        except AssertionError:
            print("No image data for vertical slice!")
            xp_v_plot, yp_v_plot = xv, zerov
            xg_v_plot, yg_v_plot = xv, zerov
        py, gy = xp_v_plot[np.argmax(yp_v_plot)], xg_v_plot[np.argmax(yg_v_plot)]
        print("edge: {0} poly:{1}, gaussian:{2}".format(ecy, py, gy))

        # DATA RECORDING
        fwrite.write("x: edge_centroid: {0}, poly: {1}, gaussian: {2}; ".format(ecx, px, gx))
        fwrite.write('\n')
        fwrite.write("y: edge_centroid: {0}, poly: {1}, gaussian: {2}\n".format(ecy, py, gy))

        # if i >= 13:
        fig = plt.figure(figsize=(16, 8))
        plt.subplot(211)
        imgx.initialize()
        plt.plot(xh, imgx.extract_array(), 'b-', xh, bgx, 'm-', xh, deducedx, 'c-', xp_h_plot, yp_h_plot, 'g-',
                 xg_h_plot, yg_h_plot, 'r-')
        plt.ylabel("Horizontal Slice, ECX: {0}".format(ecx))
        plt.legend(['Raw image data', 'gaussian background', 'remnant', 'polynomial', 'gaussian'], loc="upper right")
        plt.subplot(212)
        imgy.initialize()
        plt.plot(xv, imgy.extract_array(), 'b-', xv, bgy, 'm-', xv, deducedy, 'c-', xp_v_plot, yp_v_plot, 'g-',
                 xg_v_plot, yg_v_plot, 'r-')
        plt.ylabel("Vertical Slice, ECY: {0}".format(ecy))
        plt.legend(['Raw image data', 'gaussian background', 'remnant', 'polynomial', 'gaussian'], loc="upper right")
        plt.show()
        # plt.savefig(NAME_HEADER + "_" + gk_setting + ".png") # UNCOMMENT WHEN SAVING PLOTS


FM = FastDataMatrix2D


def folder_to_imgs(img_name_scheme, num_sample):
    """This function takes img files and return cv imgs"""
    return [cv2.imread(img_name_scheme.format(i)) for i in range(1, num_sample+1)]


def center_detect_old(img_name_scheme, num_sample, sample_int=50):
    """This function takes in a list of images and output x, y [pixel] coordinates of the center of the cross hair"""
    imgs = folder_to_imgs(img_name_scheme, num_sample)
    num_imgs = len(imgs)
    dimr = imgs[0].shape[0]
    dimc = imgs[0].shape[1]
    xnum_imgs = num_imgs
    ynum_imgs = num_imgs
    xsum = 0
    ysum = 0
    zw = 0
    ew = 1
    gk = 9
    count = 1
    for imgr in imgs:
        if count == num_sample:
            imgshow = imgr
        count += 1
        #print(imgr.shape)
        # Image Processing
        gksize = (gk, gk)
        sigmaX = 0
        img = cv2.GaussianBlur(imgr, gksize, sigmaX)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=-1)
        sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=-1)
        # Gathering Data
        nr = sample_int
        r_thresh = dimr * 2.0 / (sample_int * 3)
        xs = []
        xdummies = []
        ys = []
        ydummies = []
        while nr < dimr:
            data_x = sobelx[nr, :]
            zc_x = zero_crossing(data_x)
            print("row:", nr)
            ed_x = edge_centroid(data_x, FM(img, FM.HOR, nr))
            print("At {0}: {1}".format(nr, ed_x))
            nr += sample_int
            xdummies.append(zc_x * zw + ed_x * ew)
            if zc_x == -1:
                continue
            else:
                xs.append(zc_x * zw + ed_x * ew)
        nc = sample_int
        c_thresh = dimc * 2.0 / (sample_int * 3)
        while nc < dimc:
            data_y = sobely[:, nc]
            zc_y = zero_crossing(data_y)
            ed_y = edge_centroid(data_y, FM(img, FM.VER, nc))
            nc += sample_int
            ydummies.append(zc_y * zw + ed_y * ew)
            if zc_y == -1:
                continue
            else:
                ys.append(zc_y * zw + ed_y * ew)
        #print("x")
        #print(xdummies)
        #print(xs)
        #print("y")
        #print(ydummies)
        #print(ys)
        len_xs = len(xs)
        len_ys = len(ys)
        print("img {0}: {1}".format(count - 1, xs))
        if len_xs < r_thresh:
            xnum_imgs -= 1
        else:
            xsum += sum(xs) / len(xs)
        if len_ys < c_thresh:
            ynum_imgs -= 1
        else:
            ysum += sum(ys) / len(ys)

    plt.show()

    center_x = -1 if xnum_imgs == 0 else xsum / xnum_imgs
    center_y = -1 if ynum_imgs == 0 else ysum / ynum_imgs
    return center_x, center_y

# The image taken is flipped horizontally, result x should be img.shape[1] - x
# The image sometimes has two peaks, try experimenting with different gaussian kernels


def center_detect_test(folder_path, img_name_scheme, num_sample, sample_int=50, debug=False, gk=9, ks=-1, m=1, p=10, b=1, c=0, hough=True):
    """This function takes in a list of images and output x, y [pixel] coordinates of the center of the cross hair
    hs: HORIZONTAL SLICE!  vs: VERTICAL SLICE!"""
    imgr = test_noise_reduce(folder_path + img_name_scheme, num_sample)[0]
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
    # ------------------------------------------------------------
    # DEBUG MODULE
    if debug:
        print((dimr, dimc))
        print("imgX:", hs)
        print("imgY:", vs)
        compare_images((sobelx, 'Sobel X'), (sobely, 'Sobel Y'), color_map='gray')
        test(folder_path, img_name_scheme)
    # ------------------------------------------------------------
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
    #hough_img = img_name_scheme.format(1)
    if hough:
        hough_img = folder_path + img_name_scheme.format(1)
        print(hough_img)
        img_h = cv2.imread(hough_img)
        if len_hs:
            hp1, hp2, line_a = line_graph_contrast(img_h, hxs, hys)
            print('Drawn H')
        if len_vs:
            vp1, vp2, line_b = line_graph_contrast(img_h, vxs, vys)
            print('Drawn V')
        namespace = 'houghs/hough_{0}{1}_{2}'.format(folder_path[3:], m, img_name_scheme.format(1))
        if len_hs:
            print(hp1, hp2)
        if len_vs:
            print(vp1, vp2)
        print(namespace)
        cv2.imwrite(namespace, img_h)
    else:
        line_a = HoughLine(x=hxs, data=hys)
        line_b = HoughLine(x=vxs, data=vys)
    # --------------------------------------------------------
    # DATA RECORDING AND PROCESSING
    x_valid = False
    y_valid = False
    stdh = -1
    stdv = -1
    valuesH = [d[1] for d in hs]
    valuesV = [d[1] for d in vs]
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
    return center_x, center_y, stdh, stdv


def center_detect(img_name_scheme, num_sample, sample_int=50, debug=False, gk=9, ks=-1, m=1, p=10, b=1):
    """This function takes in a list of images and output x, y [pixel] coordinates of the center of the cross hair
    hs: HORIZONTAL SLICE!  vs: VERTICAL SLICE!"""
    imgr = test_noise_reduce(img_name_scheme, num_sample)[0]
    dimr = imgr.shape[0]
    dimc = imgr.shape[1]
    zw = 0
    ew = 1
    # Image Processing
    gksize = (gk, gk)
    sigmaX = 0
    img = cv2.GaussianBlur(imgr, gksize, sigmaX)
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=ks)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=ks)
    # Gathering Data
    nr = sample_int
    r_thresh = dimr / (sample_int * 3.0)
    hs = []
    vs = []
    while nr < dimr:
        data_x = FM(sobelx, FM.HOR, nr)
        zc_x = zero_crossing(data_x)
        ec_x = edge_centroid(data_x, FM(img, FM.HOR, nr))
        nr += sample_int
        if ec_x == -1:
            continue
        else:
            hs.append((nr - sample_int, zc_x * zw + ec_x * ew))
    nc = sample_int
    c_thresh = dimc / (sample_int * 3.0)
    while nc < dimc:
        data_y = FM(sobely, FM.VER, nc)
        zc_y = zero_crossing(data_y)
        ec_y = edge_centroid(data_y, FM(img, FM.VER, nc))
        nc += sample_int
        if ec_y == -1:
            continue
        else:
            vs.append((nc - sample_int, zc_y * zw + ec_y * ew))
    len_hs = len(hs)
    len_vs = len(vs)
    if debug:
        print((dimr, dimc))
        print("imgX:", hs)
        print("imgY:", vs)
        compare_images((sobelx, 'Sobel X'), (sobely, 'Sobel Y'), color_map='gray')
        while True:
            ax = input("Input the axis you wanna debug, h for horizontal, v for vertical, or c to cancel:  ")
            if ax == 'c':
                break
            inds = input("Input the index you wanna debug:  ")
            while True:
                try:
                    index = int(inds)
                    axis = FM.HOR if ax == 'h' else FM.VER
                    assert 0 <= index < imgr.shape[1 - axis]
                    break
                except ValueError:
                    inds = input("Invalid input, do it again:  ")
                except AssertionError:
                    inds = input("Index Out of Bound, do it again: {0}  ".format(imgr.shape))
            sobelop = sobelx if axis else sobely
            quick_plot(FM(sobelop, axis, index).extract_array())

    # ----- Following Modules Handles Hough Line Drawing -----
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

    hough_img = img_name_scheme.format(1)
    img_h = cv2.imread(hough_img)
    if len_hs:
        hp1, hp2, lina = line_graph_contrast(img_h, hxs, hys)
        print('Drawn H')
    if len_vs:
        vp1, vp2, linb = line_graph_contrast(img_h, vxs, vys)
        print('Drawn V')
    namespace = 'houghs/hough_{0}'.format(hough_img[3:])
    if len_hs:
        print(hp1, hp2)
    if len_vs:
        print(vp1, vp2)
    print(namespace)
    cv2.imwrite(namespace, img_h)
    # --------------------------------------------------------
    x_invalid = False
    y_invalid = False
    if len_hs < r_thresh:
        x_invalid = True
    if len_vs < c_thresh:
        y_invalid = True

    center_x = -1 if x_invalid else sum([d[1] for d in hs]) / len_hs
    center_y = -1 if y_invalid else sum([d[1] for d in vs]) / len_vs
    return center_x, center_y


def random_test():
    data = [27, 21, 22, 21, 21, 18, 41, 69, 83, 62, 38, 16, 21, 20, 18, 17]
    print(len(data))
    ids = 0
    gs = 0
    for i, d in enumerate(data):
        if d == 83:
            print(i)
        ids += i * d
        gs += d
    print(ids / gs)


TEST_PIC_TRUTH = [(385, 130),
                  (385, 130),
                  (385, 130),
                  (-2, -2),
                  (176, 52),
                  (174, -1),
                  (158, -1),
                  (159, -1),
                  (242, 212),
                  (278, 215),
                  (325, 218),
                  (143, 215),
                  (53, 219),
                  (-1, 17),
                  (138, 19),
                  (355, 14),
                  (288, -1),
                  (369, 180),
                  (357, 265),
                  (225, 224),
                  (-1, 195),
                  (-1, 59),
                  (-1, -1),
                  (-1, 160),
                  (445, 166)]


def read_truth(filename):
    with open(filename) as truth:
        TRUTH = {}
        active = None
        for line in truth:
            meas = line.replace(',', ' ').split()
            if len(meas) == 0:
                pass
            elif len(meas) == 1:
                active = {}
                TRUTH[meas[0]] = active

            else:
                tp = (int(meas[1]), int(meas[2]))
                active[meas[0]] = tp
    return TRUTH


def error_calc(meas, truth):
    E = 0
    T = 0
    F = 0
    terms = 0
    #ls = ['img_5', 'img_6', 'img_7', 'img_12', 'img_13', 'img_14', 'img_15', 'img_16', 'img_17', 'img_18', 'img_20', 'img_21', 'img_22', 'img_25']
    #ms = []
    #es = []
    for m, vals in meas.items():
        """if m in ls:
            continue"""
        mx, my = vals
        tx, ty = truth[m]
        if tx == -2 or ty == -2:
            pass
        elif (mx == -1 and tx != -1) or (my == -1 and ty != -1):
            F += 1
            #ms.append(m)
        elif (mx != -1 and tx == -1) or (my != -1 and ty == -1):
            T += 1
        else:
            dx = mx - tx
            dy = my - ty
            E += (dx) ** 2 + (dy) ** 2
            #es.append((dx, dy))
            terms += 2
    # print(ms)
    # print(es)
    error = sqrt(E / terms) if terms else float('inf')
    return {'E': error, 'T': T, 'F': F}


TRUTH_FILE = "meas/truth.txt"
TRUTH = read_truth(TRUTH_FILE)


""" =================================
============= Parameter =============
=========== Optimization ============
===================================== """


# TODO: OPTIMIZE ON THE THE GAUSSIAN KERNEL PARAMETER. WHAT HAPPENED THAT MAKES THE SMALL KERNEL BETTER???
def gaussian_kernel_expr(folder, ns):
    offset = '../'
    truth_doc = TRUTH[folder]
    error = {}
    gs = range(1, 20, 2)
    Es = []
    Ts = []
    Fs = []
    for g in gs:
        meas = {}
        #print(g)
        for i in range(1, 26):
            img_name = ns.format(i)
            imgfile = offset + folder + "/%s_{0}.png" % img_name
            # if i in [11, 13, 21, 23, 25]:
            if i == 0:
                val = center_detect(imgfile, 3, debug=True, gk=g)
            else:
                val = center_detect(imgfile, 3, gk=g)
            #print(str(i), val)
            meas[img_name] = val
            # print(str(i), val)

        ed = error_calc(meas, truth_doc)
        Es.append(ed['E'])
        Ts.append(ed['T'])
        Fs.append(ed['F'])
        error[g] = ed
    print(error)
    compare_data_plots((Es, gs, 'Error'), (Ts, gs, 'False Positive'), (Fs, gs, 'False Negative'), suptitle="Gaussian Blur Test")


def sobel_kernel_expr(folder, ns):
    offset = '../'
    truth_doc = TRUTH[folder]
    error = {}
    #ks = [-1]
    #ks.extend(list(range(3, 8, 2)))
    ks = range(-1, 8, 2)
    Es = []
    Ts = []
    Fs = []
    for k in ks:
        # print(k)
        meas = {}
        for i in range(1, 26):
            img_name = ns.format(i)
            imgfile = offset + folder + "/%s_{0}.png" % img_name
            # if i in [11, 13, 21, 23, 25]:
            if i == 0:
                val = center_detect(imgfile, 3, debug=True, ks=k)
            else:
                val = center_detect(imgfile, 3, ks=k)
            #print(str(i), val)
            meas[img_name] = val
            # print(str(i), val)

        ed = error_calc(meas, truth_doc)
        Es.append(ed['E'])
        Ts.append(ed['T'])
        Fs.append(ed['F'])
        error[k] = ed
    print(error)
    compare_data_plots((Es, ks, 'Error'), (Ts, ks, 'False Positive'), (Fs, ks, 'False Negative'), suptitle="Sobel Kernel Test")


def convergence_test(folder, ns):
    offset = '../'
    convergence = {}
    line_stdh = []
    line_stdv = []
    variations = []
    startNP = 59
    startP = 80
    endP = 193
    ms = range(3)
    fwrite = open('meas/convergence.csv', 'w')
    cwriter = csv.writer(fwrite)
    cwriter.writerow(['Image Number', 'Center X', 'Center Y', 'StdDev Horizontal', 'Std Dev Vertical'])
    for m in ms:
        cwriter.writerow([str(m)])
        # Convergence
        lrh = 0
        lrhs = 0
        lrv = 0
        lrvs = 0
        # Consistency Cycled
        pv = 0
        pvs = 0
        rcount = 0
        cvx = np.zeros(4)
        cvy = np.zeros(4)
        #print(g)
        for i in range(startNP, startP):
            img_name = ns.format(i)
            fpath = offset + folder
            imgfile = "%s_{0}.png" % img_name
            # FOR NULL ROW OR COLUMN, DO NOT COUNT THE STDDEV
            try:
                x, y, stdh, stdv = center_detect_test(fpath, imgfile, 5, m=m)
                # PUT IN CSV
                cwriter.writerow([str(i), str(x), str(y), str(stdh), str(stdv)])
                # CONVERGENCE
                if x != -1:
                    lrh += stdh ** 2
                    lrhs += 1
                if y != -1:
                    lrv += stdv ** 2
                    lrvs += 1
            except AttributeError:
                print('No {0}'.format(fpath + imgfile))
                pass
        for i in range(startP, endP):
            img_name = ns.format(i)
            fpath = offset + folder
            imgfile = "%s_{0}.png" % img_name
            # FOR NULL ROW OR COLUMN, DO NOT COUNT THE STDDEV
            try:
                x, y, stdh, stdv = center_detect_test(fpath, imgfile, 5, m=m)
                # PUT IN CSV
                cwriter.writerow([str(i), str(x), str(y), str(stdh), str(stdv)])
                # CONVERGENCE
                if x != -1:
                    lrh += stdh ** 2
                    lrhs += 1
                if y != -1:
                    lrv += stdv ** 2
                    lrvs += 1
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
        mselrh = sqrt(lrh / lrhs)
        mselrv = sqrt(lrv / lrvs)
        msepv = sqrt(pv / pvs)
        cvg = {'LineConvergenceH': mselrh if lrhs else float('inf'),
               'LineConvergenceV': mselrv if lrvs else float('inf'),
               'PicConsistency': msepv}
        convergence[m] = cvg
        line_stdh.append(mselrh)
        line_stdv.append(mselrv)
        variations.append(msepv)
    print(convergence)
    compare_data_plots((line_stdh, ms, 'Line Convergence Hor'), (line_stdv, ms, 'Line Convergence Ver'),
                       (variations, ms, 'Reading Consistency'), suptitle="Convergence/Consistency Test")
    fwrite.close()





def parameter_convert(method, padding, blur):
    return method + 2 * padding + blur * 100


def main():
    # TODO: HANDLE THE MAXI MINI LEFT AND RIGHT PROBLEM (2 PEAKS)
    folder = "../calib4/"
    img_name = "img_59_{0}.jpg"
    ns = folder + "img_%d_{0}.png"
    # print("Center Detection yields: ")
    # gaussian_kernel_expr('testpic', 'img_{0}')
    #sobel_kernel_expr('testpic', 'img_{0}')
    convergence_test('calib4/', 'img_{0}')
    """for i in range(59, 193):
        #if i in [29]:
        try:
            if i == 0:
                val = center_detect(ns % i, 5, debug=True)
            else:
                val = center_detect(ns % i, 5)
            print(str(i), val)
        except TypeError:
            print('No image {0}!'.format(i))
        print('------------------------------------------------------')"""


def case_calc():
    hori_lower = 53.50 - 0.13
    hori_upper = 53.50 + 0.13
    verti_lower = 41.41 - 0.11
    verti_upper = 41.41 + 0.11
    height = 3
    width = 4
    print('hori_lower: {0}'.format(distance(width / 2, hori_lower / 2)))
    print('hori_upper: {0}'.format(distance(width / 2, hori_upper / 2)))
    print('verti_lower: {0}'.format(distance(height / 2, verti_lower / 2)))
    print('verti_upper: {0}'.format(distance(height / 2, verti_upper / 2)))


def distance(h, angle):
    return 2.54 * (h / tan(radians(angle)))


    #print(center_detect(folder + img_name, 10, debug=True))



    #print("Center Detection yields: ")
    #print(center_detect("../new_test/img_3_{0}.png", 5))
    #2.5 + 13  = 15.5 / 2 = 7.75

    """plt.figure()
    plt.plot([1,2 ,3])
    plt.show()"""

    #print(atan(56 / 200) * 180 / 3.1415926535)
    #print(atan(134 / 537) * 180 / 3.1415926535)

    #test()
    #test_canny_detect()
    #random_test()


if __name__ == '__main__':
    main()
    #print(DEGREES)
    #test_interactive()
    #test_canny_detect()



# INSIGHT:
"""Might be able to solve the problem by auto-thresholding and converting to binary.
Alternative:
    1. Mark the maximum and minimum that are certain multitudes of standard deviation above

===========================================
========= THE FOLLOWING FUNCTIONS =========
============== ARE DEPRECATED =============
==========================================-

def OMP(imDims, sparsity, measurements, A):
    display = None
    numPixels = 0
    r = measurements.copy()
    indices = []

    # Threshold to check error. If error is below this value, stop.
    THRESHOLD = 0.1

    # For iterating to recover all signal
    i = 0

    while i < sparsity and np.linalg.norm(r) > THRESHOLD:
        # Calculate the correlations
        print('%d - ' % i, end="", flush=True)
        corrs = A.T.dot(r)

        # Choose highest-correlated pixel location and add to collection
        # COMPLETE THE LINE BELOW
        best_index = np.argmax(np.abs(corrs))
        indices.append(best_index)

        # Build the matrix made up of selected indices so far
        # COMPLETE THE LINE BELOW
        Atrunc = A[:, indices]

        # Find orthogonal projection of measurements to subspace
        # spanned by recovered codewords
        b = measurements
        # COMPLETE THE LINE BELOW
        xhat = np.linalg.lstsq(Atrunc, b)[0]

        # Find component orthogonal to subspace to use for next measurement
        # COMPLETE THE LINE BELOW
        r = b - Atrunc.dot(xhat)

        # This is for viewing the recovery process
        if i % 10 == 0 or i == sparsity - 1 or np.linalg.norm(r) <= THRESHOLD:
            recovered_signal = np.zeros(numPixels)
            for j, x in zip(indices, xhat):
                recovered_signal[j] = x
            Ihat = recovered_signal.reshape(imDims)
            plt.title('estimated image')
            plt.imshow(Ihat, cmap=plt.cm.gray, interpolation='nearest')
            display.clear_output(wait=True)
            display.display(plt.gcf())

        i = i + 1

    display.clear_output(wait=True)

    # Fill in the recovered signal
    recovered_signal = np.zeros(numPixels)
    for i, x in zip(indices, xhat):
        recovered_signal[i] = x

    return recovered_signal

if ori == 'h':
    x = np.array(range(img.shape[1]))
    y = np.array([img_rec.rel_lumin(img, p, c) for c in x])
else:
    x = np.array(range(img.shape[0]))
    y = np.array([img_rec.rel_lumin(img, r, p) for r in x])
"""


