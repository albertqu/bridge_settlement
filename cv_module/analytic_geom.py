import numpy as np
from math import sqrt, asin, degrees, sin, cos
from scipy.optimize import least_squares
from utils import root_finding

""" ===================================
========== ANALYTIC GEOMETRY ==========
======================================= """


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
