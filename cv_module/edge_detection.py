import numpy as np
import cv2, heapq
from .utils import check_crossing, PseudoLL, max_min, check_cross, \
    root_finding, FastDataMatrix2D, poly_curve, gauss_reg
from .fitting import improvedCost

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
    std_dev = np.sqrt(std_dev / len_data)
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
        param = gauss_reg(x, idata, p0=[10, (maxi + mini) / 2, np.std(idata)])
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
        param = gauss_reg(x, idata, p0=[10, (maxi + mini) / 2, np.std(idata)])
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
    return gauss_reg(x, idata, p0=[10, (maxi + mini) / 2, np.std(idata)]), x
