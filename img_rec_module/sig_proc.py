import cv2


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


def folder_to_imgs(img_name_scheme, num_sample):
    """This function takes img files and return cv imgs"""
    return [cv2.imread(img_name_scheme.format(i)) for i in range(1, num_sample+1)]


def center_detect(img_name_scheme, num_sample, sample_int=50):
    """This function takes in a list of images and output x, y [pixel] coordinates of the center of the cross hair"""
    imgs = folder_to_imgs(img_name_scheme, num_sample)
    num_imgs = len(imgs)
    dimr = imgs[0].shape[0]
    dimc = imgs[0].shape[1]
    xnum_imgs = num_imgs
    ynum_imgs = num_imgs
    xsum = 0
    ysum = 0
    zw = 0.5
    ew = 0.5
    gk = 7
    for imgr in imgs:
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
        ys = []
        while nr < dimr:
            data_x = sobelx[nr, :]
            zc_x = zero_crossing(data_x)
            ed_x = edge_converge(data_x)
            nr += sample_int
            if zc_x == -1:
                continue
            else:
                xs.append(zc_x * zw + ed_x * ew)
        nc = sample_int
        c_thresh = dimc * 2.0 / (sample_int * 3)
        while nc < dimc:
            data_y = sobely[:, nc]
            zc_y = zero_crossing(data_y)
            ed_y = edge_converge(data_y)
            nc += sample_int
            if zc_y == -1:
                continue
            else:
                ys.append(zc_y * zw + ed_y * ew)

        len_xs = len(xs)
        len_ys = len(ys)
        if len_xs < r_thresh:
            xnum_imgs -= 1
        else:
            xsum += sum(xs) / len(xs)
        if len_ys < c_thresh:
            ynum_imgs -= 1
        else:
            ysum += sum(ys) / len(ys)

    center_x = -1 if xnum_imgs == 0 else xsum / xnum_imgs
    center_y = -1 if ynum_imgs == 0 else ysum / ynum_imgs
    return center_x, center_y

