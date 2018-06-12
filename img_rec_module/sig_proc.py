import cv2
import numpy as np
import matplotlib.pyplot as plt
from img_rec_module import img_rec
from random import randint
import os


def gauss_data_matrix(data):
    return np.array([[1, x ** 2, x, 1] for x in data])

"""if ori == 'h':
    x = np.array(range(img.shape[1]))
    y = np.array([img_rec.rel_lumin(img, p, c) for c in x])
else:
    x = np.array(range(img.shape[0]))
    y = np.array([img_rec.rel_lumin(img, r, p) for r in x])"""

def data_matrix(input_data, degree):
    # degree is the degree of the polynomial you plan to fit the data with
    Data = np.zeros((len(input_data), degree + 1))

    for k in range(0, degree + 1):
        Data[:, k] = (list(map(lambda x: x ** k, input_data)))

    return Data


def leastSquares(D, y):
    return np.linalg.lstsq(D, y, rcond=None)[0]


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


def poly_error(params, D_a, y_a):
    '''degree=len(params)-1
    y=x_a*0

    for k in range(0,degree+1):
        coeff=params[k]
        y=y+list(map(lambda z:coeff*z**k,x_a))'''
    y = np.dot(D_a, params)
    return np.linalg.norm(y - y_a) ** 2


def MSE(y, y_hat, N):
    return np.linalg.norm(y - y_hat) ** 2 / N


def improvedCost(x, y, x_test, y_test, start, end):
    """Given a set of x and y points training points,
    this function calculates polynomial approximations of varying
    degrees from start to end. Then it returns the cost, with
    the polynomial tested on test points of each degree in an array"""
    c = []
    for degree in range(start, end):
        # YOUR CODE HERE
        D = data_matrix(x, degree)
        p = leastSquares(D, y)
        D_t = data_matrix(x_test, degree)
        y_hat = np.dot(D_t, p)
        c.append(MSE(y_test, y_hat, len(x_test)))
    return c


def OMP(imDims, sparsity, measurements, A):
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


def gauss_reg(x, y):
    skip_x = np.array([x[i] for i in range(len(y)) if y[i] != 0])
    y_prime = np.array([y[i] for i in range(len(y)) if y[i] != 0])
    logy = np.log(y_prime)

    sol = leastSquares(gauss_data_matrix(skip_x), logy)
    alp, bet, gam, lam = sol
    a = np.e ** alp
    a *= 10
    c_s = - 1 / (2 * bet)
    b = gam * c_s
    return a, b, c_s


def gauss_mat(shape, a, b1, c_s1, b2, c_s2):
    mat = np.zeros(shape)
    for r in range(mat.shape[0]):
        for c in range(mat.shape[1]):
            mat[r, c] = gauss_2d(c, r, a, b1, c_s1, b2, c_s2)

    return mat


def gauss_2d(x, y, a, b1, c_s1, b2, c_s2):
    return a * np.e ** (- ((x - b1) ** 2  / (2 * c_s1) + (y - b2) ** 2) / (2 * c_s2))

def gauss_hat(x, a, b, c_s):
    return a * np.e ** (- (x - b) ** 2 / (2 * c_s))

img = cv2.imread("../testpic/img_20_1.png")
cv2.imshow("true", img)


img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

dimc = img.shape[1]
dimr = img.shape[0]

x = np.array(range(dimc))
#y = np.array([img_rec.rel_lumin(img, 0, c) for c in x])
y = np.array([img.item(dimr // 2, c) for c in x])
a1, b1, c_s1 = gauss_reg(x, y)

x2 = np.array(range(dimr))
y2 = np.array([img.item(r, dimc // 2) for r in x2])
a2, b2, c_s2 = gauss_reg(x2, y2)
rem_gauss = gauss_mat(img.shape, (a1+a2) / 2, b1, c_s1, b2, c_s2)
#img = img - rem_gauss
cv2.imshow("denoise", img)

blur = cv2.GaussianBlur(img,(5,5),0)
cv2.imshow("ok", blur)
img = blur

kernel = np.array([[-1, -1, -1],
                   [-1, 8, -1],
                   [-1, -1, -1]])
kerneled = cv2.filter2D(img, -1, kernel)
cv2.imshow("kerneled", kerneled)



laplacian = cv2.Laplacian(img,cv2.CV_64F)
sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)
plt.subplot(2,2,1),plt.imshow(img,cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,2),plt.imshow(laplacian,cmap = 'gray')
plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,3),plt.imshow(sobelx,cmap = 'gray')
plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,4),plt.imshow(sobely,cmap = 'gray')
plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
plt.show()

y_s_x = np.array([sobelx.item(dimr // 2, c) for c in x])
y_s_y = np.array([sobely.item(r, dimc // 2) for r in x2])
plt.figure(figsize=(16, 8))
plt.subplot(211)
plt.plot(y_s_x, 'b-')
plt.ylabel("sobel_x")
plt.subplot(212)
plt.plot(y_s_y, 'b-')
plt.ylabel("sobel_y")
plt.show()
plt.close()


y_hat = gauss_hat(x, a1, b1, c_s1)
plt.figure(figsize=(16, 8))
plt.plot(x, y, 'b-', x, y_hat, 'r-')
plt.show()
plt.close()
plt.figure(figsize=(16,8))
plt.plot(x, y-y_hat, 'b-')
plt.show()