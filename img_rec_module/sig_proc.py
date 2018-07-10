import cv2
import numpy as np
import matplotlib.pyplot as plt
from img_rec_module import img_rec
from random import randint
import os
from math import atan, sqrt


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


class FastDataMatrix2D:

    HOR = 1
    VER = 0

    def __init__(self, data, ax, index):
        self._data = data
        self._ax = ax
        self._index = index

    def set_axis(self, ax):
        self._ax = ax

    def set_index(self, index):
        self._index = index

    def copy(self, ax=None, index=None):
        if ax and index:
            return FastDataMatrix2D(self._data, ax, index)
        else:
            return FastDataMatrix2D(self._data, self._ax, self._index)

    def __getitem__(self, item):
        return self._data[self._index, item] if self._ax == FastDataMatrix2D.HOR else self._data[item, self._index]

    def __setitem__(self, key, value):
        if self._ax == FastDataMatrix2D.HOR:
            self._data[self._index, key] = value
        else:
            self._data[key, self._index] = value

    def __len__(self):
        return self._data.shape[self._ax]


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
    # Return the maximum and minimum for the data
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


def root_finding(x1, x2, y1, y2):
    return - y1 * (x2 - x1) / (y2 - y1) + x1


def check_cross(a, b):
    """Helper method for zero crossing"""
    return a * b < 0


def edge_converge_base(data):
    maxi, mini = max_min(data)
    width_thres = 70
    if mini - maxi > width_thres:
        return -1
    else:
        return (maxi + mini) / 2


def edge_centroid(data, img_data):
    # With Gaussian Blur might achieve the best performance
    maxi, mini = max_min(data)
    width_thres = 70
    padding = 10
    if mini - maxi > width_thres:
        return -1
    return centroid_seg(img_data, maxi - padding, mini + padding + 1)


def centroid_seg(data, start, end):
    isums = 0
    total = 0
    start = 0 if start < 0 else start
    end = len(data) if end >= len(data) else end
    for i in range(start, end):
        isums += data[i] * i
        total += data[i]
    return isums / total if total else -1


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


def min_max(data, max, min):
    g_diff = max - min
    return [d / g_diff for d in data]


def round_up(num):
    down = int(num)
    if num - down > 0:
        return num + 1
    else:
        return num


def compare_images(*args):
    # Takes in a sequence of IMG(Gray) and TITLE pairs and plot them side by side
    graph_amount = len(args)
    row = int(sqrt(graph_amount))
    col = round_up(float(graph_amount) / row)
    plt.figure(figsize=(16, 8))
    for i, pair in enumerate(args):
        plt.subplot(row, col, i+1)
        plt.imshow(pair[0], cmap='gray')
        plt.title(pair[1]), plt.xticks([]), plt.yticks([])
    plt.show()


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
    imgn = "img_3_{0}.png"
    IMGDIR = "../testpic/"
    NAME = IMGDIR + imgn

    denoised, original = test_noise_reduce(NAME)
    print(get_center_val(denoised))
    NL_denoised = cv2.fastNlMeansDenoising(original)
    bdenoise = test_blur_then_nr(NAME)

    dblur, sobelx, sobely = sobel_process(denoised, 9, 0)
    nldblur = sobel_process(NL_denoised, 9, 0)[0]
    pblur, sobelx, sobely = sobel_process(original, 9, 0)

    de_edges = canny_detect(denoised)
    db_edges = canny_detect(dblur)
    nld_edges = canny_detect(nldblur)
    blur_denoise = canny_detect(bdenoise)
    blur_edges = canny_detect(pblur)
    edge_detect_expr(db_edges, original)
    """compare_images((imgr, 'Original'), (edges, 'Canny Edge'),
                   (sobelx, 'Sobel X'), (sobely, 'Sobel Y'))"""
    compare_images((original, 'Original'), (NL_denoised, "NLMEANS"), (de_edges, 'DENOISE Edges'), (blur_edges, 'Plain Blur Edges'),
                   (denoised, 'Denoised'), (nld_edges, 'NIDE-BLUR Edges'), (db_edges, 'DE-BLUR Edges'), (blur_denoise, "BDENOISE Edges"))


def canny_detect(src):
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
    numIMG = 3
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



def test():
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
        plt.close()                                         # REMOVES TO SHOW CONTRAST OF FILTERS


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
        """fig = plt.figure(figsize=(16, 8))
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
        plt.show()"""


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
    fig.savefig(plot_save)
    #plt.savefig(plot_save)


def folder_to_imgs(img_name_scheme, num_sample):
    """This function takes img files and return cv imgs"""
    return [cv2.imread(img_name_scheme.format(i)) for i in range(1, num_sample+1)]

gk = 9


FM = FastDataMatrix2D


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

    plt.imshow(imgshow)
    plt.show()

    center_x = -1 if xnum_imgs == 0 else xsum / xnum_imgs
    center_y = -1 if ynum_imgs == 0 else ysum / ynum_imgs
    return center_x, center_y


def center_detect(img_name_scheme, num_sample, sample_int=50):
    """This function takes in a list of images and output x, y [pixel] coordinates of the center of the cross hair"""
    imgr = test_noise_reduce(img_name_scheme, num_sample)[0]
    dimr = imgr.shape[0]
    dimc = imgr.shape[1]
    zw = 0
    ew = 1
    # Image Processing
    gksize = (gk, gk)
    sigmaX = 0
    img = cv2.GaussianBlur(imgr, gksize, sigmaX)
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=-1)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=-1)
    # Gathering Data
    nr = sample_int
    r_thresh = dimr * 2.0 / (sample_int * 3)
    xs = []
    ys = []
    while nr < dimr:
        data_x = sobelx[nr, :]
        zc_x = zero_crossing(data_x)
        ed_x = edge_centroid(data_x, FM(img, FM.HOR, nr))
        nr += sample_int
        if ed_x == -1:
            continue
        else:
            xs.append(zc_x * zw + ed_x * ew)
    nc = sample_int
    c_thresh = dimc / (sample_int * 4.0)
    while nc < dimc:
        data_y = sobely[:, nc]
        zc_y = zero_crossing(data_y)
        ed_y = edge_centroid(data_y, FM(img, FM.VER, nc))
        nc += sample_int
        if ed_y == -1:
            continue
        else:
            ys.append(zc_y * zw + ed_y * ew)
    len_xs = len(xs)
    len_ys = len(ys)
    print("imgX:", xs)
    x_invalid = False
    y_invalid = False
    if len_xs < r_thresh:
        x_invalid = True
    if len_ys < c_thresh:
        y_invalid = True

    plt.imshow(imgr)
    plt.show()
    center_x = -1 if x_invalid else sum(xs) / len_xs
    center_y = -1 if y_invalid else sum(ys) / len_ys
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


#print("Center Detection yields: ")
print(center_detect("../testpic/img_15_{0}.png", 3))
#2.5 + 13  = 15.5 / 2 = 7.75

"""plt.figure()
plt.plot([1,2 ,3])
plt.show()"""

#print(atan(56 / 200) * 180 / 3.1415926535)
#print(atan(134 / 537) * 180 / 3.1415926535)

#test()
#test_canny_detect()
#random_test()

# INSIGHT:
"""Might be able to solve the problem by auto-thresholding and converting to binary.
Alternative:
    1. Mark the maximum and minimum that are certain multitudes of standard deviation above

"""
