import cv2
import numpy as np
from math import sqrt, tan, radians


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

    def extract_array(self):
        return self._data[self._index, :] if self._ax == FastDataMatrix2D.HOR else self._data[:, self._index]

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


def data_matrix(input_data, degree):
    # Polynomial fitting data matrix
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
    value_thres = 20
    padding = 10
    if data[maxi] < value_thres or mini - maxi > width_thres:
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


def sobel_process(imgr, gks, sig):
    gksize = (gks, gks)
    sigmaX = sig
    blur = cv2.GaussianBlur(imgr, gksize, sigmaX)
    # cv2.imshow("blurred", blur)                        # REMOVES TO SHOW BLURRED IMAGE
    if len(imgr.shape) == 3:
        img = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    else:
        img = blur
    # IMAGE PROCESSING
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=-1)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=-1)
    return img, sobelx, sobely



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
    for i in range(1, numIMG + 1):
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


def folder_to_imgs(img_name_scheme, num_sample):
    """This function takes img files and return cv imgs"""
    return [cv2.imread(img_name_scheme.format(i)) for i in range(1, num_sample+1)]

gk = 9


FM = FastDataMatrix2D


# The image taken is flipped horizontally, result x should be img.shape[1] - x
# The image sometimes has two peaks, try experimenting with different gaussian kernels


def center_detect(img_name_scheme, num_sample, sample_int=50, debug=False, gk=9):
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
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0,  ksize=-1)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=-1)
    # Gathering Data
    nr = sample_int
    r_thresh = dimr / (sample_int * 3.0)
    xs = []
    ys = []
    while nr < dimr:
        data_x = FM(sobelx, FM.HOR, nr)
        zc_x = zero_crossing(data_x)
        ed_x = edge_centroid(data_x, FM(img, FM.HOR, nr))
        nr += sample_int
        if ed_x == -1:
            continue
        else:
            xs.append((nr - sample_int, zc_x * zw + ed_x * ew))
    nc = sample_int
    c_thresh = dimc / (sample_int * 3.0)
    while nc < dimc:
        data_y = FM(sobely, FM.VER, nc)
        zc_y = zero_crossing(data_y)
        ed_y = edge_centroid(data_y, FM(img, FM.VER, nc))
        nc += sample_int
        if ed_y == -1:
            continue
        else:
            ys.append((nc - sample_int, zc_y * zw + ed_y * ew))
    len_xs = len(xs)
    len_ys = len(ys)
    x_invalid = False
    y_invalid = False
    if len_xs < r_thresh:
        x_invalid = True
    if len_ys < c_thresh:
        y_invalid = True

    center_x = -1 if x_invalid else sum([d[1] for d in xs]) / len_xs
    center_y = -1 if y_invalid else sum([d[1] for d in ys]) / len_ys
    return center_x, center_y


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
    for m, vals in meas.items():
        mx, my = vals
        tx, ty = truth[m]
        if tx == -2 or ty == -2:
            pass
        elif (mx == -1 and tx != -1) or (my == -1 and ty != -1):
            F += 1
        elif (mx != -1 and tx == -1) or (my != -1 and ty == -1):
            T += 1
        else:
            E += sqrt((mx - tx) ** 2 + (my - ty) ** 2)
    return {'E': E, 'T': T, 'F': F}


def main():
    folder = "../calib3/"
    img_name = "img_59_{0}.jpg"
    ns = folder + "img_%d_{0}.png"
    # print("Center Detection yields: ")
    #gaussian_kernel_expr('testpic', 'img_{0}')
    for i in range(59, 60):
        #if i in [29]:
        if i == 0:
            val = center_detect(ns % i, 5, debug=True)
        else:
            val = center_detect(ns % i, 5)
        print(str(i), val)


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


if __name__ == '__main__':
    main()
