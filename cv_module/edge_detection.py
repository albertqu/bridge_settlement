import numpy as np
import cv2, heapq
from utils import check_crossing, PseudoLL, FastDataMatrix2D, gauss_reg

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


def smart_interval(start, end, data):
    start = 0 if start < 0 else start
    end = len(data) if end > len(data) else end
    return start, end


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

