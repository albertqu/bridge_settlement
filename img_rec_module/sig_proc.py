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


def max_min(data):
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

def zero_crossing(data):
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


def root_finding(x1, x2, y1, y2):
    return - y1 * (x2 - x1) / (y2 - y1) + x1


def check_cross(a, b):
    return a * b < 0


def edge_converge(data):
    maxi, mini = max_min(data)
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

imgn = "img_20_1"
fwrite = open(imgn+"_meas.txt", "w")
cenvalsx = []
cenvalsy = []
edgvalsx = []
edgvalsy = []
ins = range(5, 20, 2)
end = 15
ins_e = range(5, end, 2)
imgr = cv2.imread("../testpic/" + imgn + ".png")
cv2.imshow("true", imgr)
dimc = imgr.shape[1]
dimr = imgr.shape[0]
x = np.array(range(dimc))
x2 = np.array(range(dimr))
for i in ins:
    sig = 0
    #img = cv2.imread("../testpic/" + imgn + ".png")
    #cv2.imshow("true", img)
    """img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    x = np.array(range(dimc))
    #y = np.array([img_rec.rel_lumin(img, 0, c) for c in x])
    y = np.array([img.item(dimr // 2, c) for c in x])
    a1, b1, c_s1 = gauss_reg(x, y)
    
    x2 = np.array(range(dimr))
    y2 = np.array([img.item(r, dimc // 2) for r in x2])
    a2, b2, c_s2 = gauss_reg(x2, y2)
    rem_gauss = gauss_mat(img.shape, (a1+a2) / 2, b1, c_s1, b2, c_s2)
    #img = img - rem_gauss
    cv2.imshow("denoise", img)"""

    gksize = (i, i)
    sigmaX = sig

    blur = cv2.GaussianBlur(imgr,gksize,sigmaX)
    #cv2.imshow("ok", blur)
    img = blur
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    kernel = np.array([[-1, -1, -1],
                       [-1, 8, -1],
                       [-1, -1, -1]])
    kerneled = cv2.filter2D(img, -1, kernel)
    #cv2.imshow("kerneled", kerneled)



    laplacian = cv2.Laplacian(img,cv2.CV_64F)
    sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=-1)
    sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=-1)
    """plt.subplot(2,2,1),plt.imshow(img,cmap = 'gray')
    plt.title('Original'), plt.xticks([]), plt.yticks([])
    plt.subplot(2,2,2),plt.imshow(laplacian,cmap = 'gray')
    plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
    plt.subplot(2,2,3),plt.imshow(sobelx,cmap = 'gray')
    plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
    plt.subplot(2,2,4),plt.imshow(sobely,cmap = 'gray')
    plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
    plt.close()"""
    #plt.show()

    y_s_x = np.array([sobelx.item(dimr // 2, c) for c in x])
    y_s_y = np.array([sobely.item(r, dimc // 2) for r in x2])
    fname = "ksize:{0}, sigmax:{1}".format(gksize, sigmaX)
    fwrite.write(fname)
    fwrite.write('\n')
    zx = zero_crossing(y_s_x)
    cenvalsx.append(zx)
    if i < end:
        ex = edge_converge(y_s_x)
        edgvalsx.append(ex)
    zy = zero_crossing(y_s_y)
    cenvalsy.append(zy)
    if i < end:
        ey = edge_converge(y_s_y)
        edgvalsy.append(ey)
    fwrite.write("x: zero_crossing: {0}, edge_converge: {1}; ".format(zx, ex))
    fwrite.write('\n')
    fwrite.write("y: zero_crossing: {0}, edge_converge: {1}\n".format(zy, ey))
    fig = plt.figure(figsize=(16, 8))
    fig.canvas.set_window_title(fname)
    plt.subplot(211)
    plt.plot(y_s_x, 'b-')
    plt.ylabel("sobel_x")
    plt.subplot(212)
    plt.plot(y_s_y, 'b-')
    plt.ylabel("sobel_y")
    #plt.show()
    plt.savefig(fname + ".png")


    """y_hat = gauss_hat(x, a1, b1, c_s1)
    plt.figure(figsize=(16, 8))
    plt.plot(x, y, 'b-', x, y_hat, 'r-')
    plt.show()
    plt.close()
    plt.figure(figsize=(16,8))
    plt.plot(x, y-y_hat, 'b-')
    plt.show()"""



plt.figure(figsize=(16, 8))
plt.subplot(2, 1, 1)
plt.plot(ins, cenvalsx, 'b-', ins_e, edgvalsx, 'r-')
plt.legend(['zero crossing', 'edge_converging'], loc="lower left")
plt.ylabel("x_meas")
plt.subplot(2, 1, 2)
plt.plot(ins, cenvalsy, 'b-', ins_e, edgvalsy, 'r-')
plt.legend(['zero crossing', 'edge_converging'], loc="lower left")
plt.ylabel("y_meas")
plt.xlabel("kernel size")
plt.show()



def folder_to_imgs(img_name_scheme, num_sample):
    """This function takes img files and return cv imgs"""
    return [cv2.imread(img_name_scheme.format(i)) for i in range(1, num_sample+1)]


def center_detect(img_name_scheme, num_sample, sample_int=50):
    """This function takes in a list of images and output x, y [pixel] coordinates of the center of the cross hair"""
    imgs = folder_to_imgs(img_name_scheme, num_sample)
    xsum = 0
    ysum = 0
    num_imgs = len(imgs)
    dimr = imgs[0].shape[0]
    dimc = imgs[0].shape[1]
    for img in imgs:
        # Image Processing
        gksize = (9, 9)
        sigmaX = 0
        img = cv2.GaussianBlur(imgr, gksize, sigmaX)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=-1)
        sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=-1)
        # Gathering Data
        nr = sample_int
        xs = []
        ys = []
        while nr < dimr:
            data_x = sobelx[nr, :, :]
            zc_x = zero_crossing(data_x)
            ed_x = edge_converge(data_x)
            nr += sample_int
            if zc_x == -1:
                continue
            else:
                xs.append(zc_x * 0.8 + ed_x * 0.2)
        nc = sample_int
        while nc < dimc:
            data_y = sobely[:, nc, :]
            zc_y = zero_crossing(data_y)
            ed_y = edge_converge(data_y)
            nc += sample_int
            if zc_y == -1:
                continue
            else:
                xs.append(zc_y * 0.8 + ed_y * 0.2)

        xsum += sum(xs) / len(xs)
        ysum += sum(ys) / len(ys)

    return xsum / num_imgs, ysum / num_imgs