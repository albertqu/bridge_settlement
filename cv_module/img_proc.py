import cv2
import numpy as np
from .utils import gauss_hat, gauss_reg
from .edge_detection import extract_extrema

"""======================================
======== IMAGE PROCESSING UTIL ==========
========================================= """


def image_diff(img1, img2):
    # Calculate the difference in images
    i1 = img1.astype(np.int16)
    i2 = img2.astype(np.int16)
    res = i1 - i2
    dst = np.zeros_like(res, dtype=np.uint8)
    return cv2.normalize(res, dst, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=8)


def gauss_bg_deduce(x, img_data):
    # TODO: OPTIMIZE PERFORMANCE
    idata = img_data.extract_array()
    p0 = [1, len(img_data) / 2, np.std(idata)]
    a, b, c_s = gauss_reg(x, idata, p0=p0)
    rem_gauss = gauss_hat(x, a, b, c_s)
    new_y = idata - rem_gauss
    return x, rem_gauss, new_y


def significance_test(data, val):
    """Outputs whether the maximum or minimum value is a significant value."""
    return val in extract_extrema(data)[0] or val in extract_extrema(data)[1]


def get_center_val(img):
    return img.item(img.shape[0] // 2, img.shape[1] // 2)


def test_noise_reduce(iname, numIMG=5):
    imgs = iname
    imgr = None
    original = None
    for i in range(1, numIMG + 1):
        target = cv2.imread(imgs.format(i), 0)
        if target is None:
            raise AttributeError("File {0} not found".format(iname))
        if i == 1:
            imgr = np.uint16(target)
            original = target
        else:
            imgr = imgr + target
    return imgr / numIMG, original