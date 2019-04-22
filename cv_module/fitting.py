import numpy as np
from .utils import max_min, gauss_hat, gauss_2d
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


# Generic Regression Tools 128-133
def leastSquares(D, y):
    return np.linalg.lstsq(D, y, rcond=None)[0]


def MSE(y, y_hat, N):
    return np.linalg.norm(y - y_hat) ** 2 / N


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


# ---------  GAUSSIAN -------
# Gaussian Regression Tools 184-210
def gauss_data_matrix(data):
    return np.array([[1, x ** 2, x, 1] for x in data])


def gauss_reg_bias(x, y, p0):
    """Given a set of x and y training points, this function
    calculates the gaussian approximation function"""
    skip_x = np.array([x[i] for i in range(len(y)) if y[i] != 0])
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
    a = real / pred // 1000000000 * 10
    return a, b, c_s


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
