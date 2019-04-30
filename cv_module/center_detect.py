import numpy as np
import cv2, os
import matplotlib.pyplot as plt
from edge_detection import gather_all
from analytic_geom import HoughLine
from utils import fname_suffix_free
from visualization import visualize_centers
from img_proc import image_diff


""" =======================================
============== CENTER DETECT===============
=========================================== """

def center_detect(folder_path, img_name, sample_int=30, gk=9, ks=-1, l='soft_l1', debias="z", norm=1, visual=False,
                  catch=None, stat=False, invar=True, suffix='.png', saveopt="", **kwargs):
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
        if norm == 1:
            imgr = cv2.normalize(imgr, imgr, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=8)
        else:
            aut = np.empty_like(imgr, dtype=np.uint8)
            imgr = cv2.normalize(imgr, aut, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=8)
    return center_detect_base(imgr, folder_path, img_id, sample_int, gk, ks, l, debias, visual, catch, stat, saveopt,
                              **kwargs)


def center_detect_base(imgr, folder_path, img_name, sample_int=30, gk=9, ks=-1, l='soft_l1', debias="z",
                       visual=False, catch=None, stat=False, saveopt="", **kwargs):
    """ Takes in preprocessed (ambient invariant or normalization) and output x, y [pixel]
    coordinates of the center of the cross hair
    centers_v: vertical line!  hs: horizontal slice!
    So far it seems y (horizontal line) resolution is better than that of x (vertical line).
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
    if visual:
        if "ROOT_DIR" in globals():

            ROOT_DIR = globals()["ROOT_DIR"]
        else:
            ROOT_DIR = kwargs["ROOT_DIR"]
        fn = lambda p, n: p[p.find(n)+len(n)+1:]
        namespace = os.path.join(ROOT_DIR, 'robust_reg/{0}{1}_{2}'.format(fn(folder_path, ROOT_DIR), l, img_name))
        vp1_av, vp2_av = line_v.point_gen()
        hp1_av, hp2_av = line_h.point_gen()
    if debias == 'z':
        zv_err_av, zv_err_ap = line_v.debias_z()
        zh_err_av, zh_err_ap = line_h.debias_z()
    elif debias == "y":
        zv_err_av, zv_err_ap = line_v.debias_y()
        zh_err_av, zh_err_ap = line_h.debias_y()
    else:
        zv_err_av, zv_err_ap = line_v.debias_old()
        zh_err_av, zh_err_ap = line_h.debias_old()
    if catch is not None: # TODO: remove catch
        catch[1].append(np.sqrt(np.mean(zh_err_ap)))
        catch[0].append(np.sqrt(np.mean(zv_err_ap)))

    if stat:
        lh_y, lh_pred = line_h.data, (line_h.pred if line_h.pred else line_h.fit_x(line_h.x))
        lv_y, lv_pred = line_v.data, (line_v.pred if line_v.pred else line_v.fit_x(line_v.x))
        tot_y, tot_pred = np.concatenate((lh_y, lv_y)), np.concatenate((lh_pred, lv_pred))
        from sklearn.metrics import explained_variance_score, r2_score, mean_absolute_error, mean_squared_error,\
            median_absolute_error
        mfs = explained_variance_score, r2_score, mean_absolute_error, mean_squared_error, median_absolute_error
        hseq, vseq = tuple(map(lambda f: f(lh_y, lh_pred), mfs)), tuple(map(lambda f: f(lv_y, lv_pred), mfs))
        vseq2= (np.sqrt(np.sum(zv_err_av) /len(zv_err_av)), np.sqrt(np.sum(zv_err_ap) /len(zv_err_ap)),
                np.sqrt(np.max(zv_err_av)), np.sqrt(np.max(zv_err_ap)))
        hseq2 = (np.sqrt(np.sum(zh_err_av) / len(zh_err_av)), np.sqrt(np.sum(zh_err_ap) / len(zh_err_ap)),
                np.sqrt(np.max(zh_err_av)), np.sqrt(np.max(zh_err_ap)))
        vseq = vseq
        hseq = hseq
        totseq = tuple(map(lambda f: f(tot_y, tot_pred), mfs))
        tot_err_av, tot_err_ap = np.concatenate((zv_err_av, zh_err_av)), np.concatenate((zv_err_ap, zh_err_ap))
        totseq2 = (np.sqrt(np.sum(tot_err_av) / len(tot_err_av)), np.sqrt(np.sum(tot_err_ap) / len(tot_err_ap)),
                                                    np.sqrt(np.max(tot_err_av)), np.sqrt(np.max(tot_err_ap)))
    x, y = HoughLine.intersect(line_h, line_v)
    if x >= xlim or x < 0 or y < 0 or y >= ylim:
        return -1, -1, None

    if visual:
        xv_av, xv_ap = np.arange(len(zv_err_av)), np.arange(len(zv_err_ap))
        xh_av, xh_ap = np.arange(len(zh_err_av)), np.arange(len(zh_err_ap))
        hp1_ap, hp2_ap = line_h.point_gen()
        vp1_ap, vp2_ap = line_v.point_gen()
        cenhs = list(zip(line_h.data, line_h.x))
        cenvs = list(zip(line_v.data, line_v.x))
        cv2.line(imgr, hp1_ap, hp2_ap, (0, 0, 255), 1, lineType=cv2.LINE_AA)
        cv2.line(imgr, vp1_ap, vp2_ap, (0, 0, 255), 1, lineType=cv2.LINE_AA)
        fig1 = plt.figure(figsize=(20, 10))
        ax1 = fig1.add_subplot(121)
        visualize_centers(imgr, centers_v, centers_h, ax1)
        ax1.set_title('Before')
        ax2 = fig1.add_subplot(122)
        visualize_centers(imgr, cenvs, cenhs, ax2)
        ax2.scatter([x], [y], facecolor='green', linewidths=1)
        ax2.set_title('After')
        fig2 = plt.figure(figsize=(20, 10))
        ax3 = fig2.add_subplot(211)
        ax3.plot(xh_av, zh_err_av, 'mo-', xh_ap, zh_err_ap, 'bo-')
        ax3.set_title('HOR Loss Before and After')
        ax3.legend(['Before', 'After'])
        ax4 = fig2.add_subplot(212)
        ax4.plot(xv_av, zv_err_av, 'mo-', xv_ap, zv_err_ap, 'bo-')
        ax4.set_title('VER Loss Before and After')
        ax4.legend(['Before', 'After'])
        fig1.savefig(fname_suffix_free(namespace) + '_lines' + saveopt + '.png')
        fig2.savefig(fname_suffix_free(namespace) + '_errs' + saveopt + '.png')
        plt.close('all')
    if stat:
        return x, y, (line_h.opti, line_v.opti) + hseq2 + vseq2 + totseq2 + hseq + vseq + totseq
    return x, y

