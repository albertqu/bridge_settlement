import cv2
import numpy as np

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
