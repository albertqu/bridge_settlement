import numpy as np
import cv2, os
from edge_detection import gather_all
from analytic_geom import HoughLine
from img_proc import image_diff


""" =======================================
============== CENTER DETECT===============
=========================================== """

def center_detect(folder_path, img_name, sample_int=30, gk=9, ks=-1, l='soft_l1', norm=True, invar=True, suffix='.png', **kwargs):
    """This function takes in a list of images and output x, y [pixel] coordinates of the center of the cross hair
    hs: HORIZONTAL SLICE!  vs: VERTICAL SLICE!
    img_name: str, name scheme of image, with NO SUFFIX
    """
    if invar:
        if isinstance(img_name, str):
            assert img_name.find('{}') != -1, "img_name should be of form [name]{} for invariance mode!"
            ambi_n, laser_n = img_name.format(0), img_name.format(1)
            idx = img_name.format("0_1")
        else:
            ambi_n, laser_n = img_name
            idx = "{}_{}".format(ambi_n, laser_n)
        ambi, laser = cv2.imread(os.path.join(folder_path, ambi_n+suffix), 0),\
                      cv2.imread(os.path.join(folder_path, laser_n+suffix), 0)
        imgr = image_diff(laser, ambi)
        img_id = idx + suffix
    else:
        img_id = img_name + suffix
        imgr = cv2.imread(os.path.join(folder_path, img_id), 0)
    if norm:
        imgr = cv2.normalize(imgr, imgr, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=8)
    return center_detect_base(imgr, folder_path, img_id, sample_int, gk, ks, l, **kwargs)


def center_detect_base(imgr, folder_path, img_name, sample_int=30, gk=9, ks=-1, l='soft_l1', **kwargs):
    """ Takes in preprocessed (ambient invariant or normalization) and output x, y [pixel]
    coordinates of the center of the cross hair
    centers_v: vertical line!  hs: horizontal slice!
    So far it seems y (horizontal line) resolution is better than that of x (vertical line).
    debias: always z
    """
    ylim, xlim = imgr.shape
    centers_v, centers_h = gather_all(imgr, sample_int, gk, ks)
    centers_v, centers_h = np.array(centers_v), np.array(centers_h)
    try:
        centers_vx, centers_vy = centers_v[:, 1], centers_v[:, 0]
        centers_hx, centers_hy = centers_h[:, 1], centers_h[:, 0]
    except IndexError:
        return -1, -1, None

    line_v, line_h = HoughLine(x=centers_vx, data=centers_vy), HoughLine(x=centers_hx, data=centers_hy, loss=l)

    line_v.debias_z()
    line_h.debias_z()

    x, y = HoughLine.intersect(line_h, line_v)
    if x >= xlim or x < 0 or y < 0 or y >= ylim:
        return -1, -1, None

    return x, y
