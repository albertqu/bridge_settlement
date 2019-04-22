import numpy as np
import cv2, os
import matplotlib.pyplot as plt
from .edge_detection import gather_all
from .analytic_geom import HoughLine


def center_detect_hough(img):
    centers_v, centers_h = gather_all(img)
    centpointsv = np.zeros_like(img, dtype=np.uint8)
    centpointsh = np.zeros_like(img, dtype=np.uint8)
    #print(centers_v+centers_h)
    for r, c in centers_v:
        try:
            centpointsv[int(r)][int(c)] = 255
        except:
            continue
    for r, c in centers_h:
        try:
            centpointsh[int(r)][int(c)] = 255
        except:
            continue
    linesv = cv2.HoughLines(centpointsv, 0.5, 0.005, 1)
    linesh = cv2.HoughLines(centpointsh, 0.5, 0.005, 1)
    all_lines = []
    try:
        for rho, theta in linesv[0]:
            #print(rho, theta)
            all_lines.append(HoughLine(theta, rho))
    except:
        print(linesv)
    for rho, theta in linesh[0]:
        all_lines.append(HoughLine(theta, rho))
    ct = HoughLine.intersect(all_lines[0], all_lines[-1])
    return ct


def center_detect(folder_path, img_name, sample_int=30, gk=9, ks=-1, l='soft_l1', debias="z", norm=1, visual=False,
                  catch=None, stat=False,
                  saveopt=""):
    """This function takes in a list of images and output x, y [pixel] coordinates of the center of the cross hair
    hs: HORIZONTAL SLICE!  vs: VERTICAL SLICE!"""
    # TODO: CHANGE DEBIAS AND CHANGE VISUALIZATION PART TO HAVE:
    # TODO: 1. CENTER_VISUALIZATION WITH LINE DRAWN
    # TODO: 2. ERROR BEFORE AND AFTER
    imgr = cv2.imread(os.path.join(folder_path, img_name), 0)
    if norm:
        aut = np.empty_like(imgr, dtype=np.uint8)
        if norm == 1:
            imgr = cv2.normalize(imgr, imgr, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=8)
        else:
            imgr = cv2.normalize(imgr, aut, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=8)
    return center_detect_base(imgr, folder_path, img_name, sample_int, gk, ks, l, debias, visual, catch, stat, saveopt)


def center_detect_invar(imgr, sample_int=30, gk=9, ks=-1, l='soft_l1', debias="z", visual=False,
                        folder_path="../", img_name='.png', saveopt="", stat=False):
    """This function takes in a list of images and output x, y [pixel] coordinates of the center of the cross hair
    hs: HORIZONTAL SLICE!  vs: VERTICAL SLICE!"""
    centers_v, centers_h = gather_all(imgr, sample_int, gk, ks)
    centers_v, centers_h = np.array(centers_v), np.array(centers_h)
    centers_vx, centers_vy = centers_v[:, 1], centers_v[:, 0]
    centers_hx, centers_hy = centers_h[:, 1], centers_h[:, 0]

    line_v, line_h = HoughLine(x=centers_vx, data=centers_vy), HoughLine(x=centers_hx, data=centers_hy, loss=l)
    if visual:
        namespace = 'robust_reg/{0}{1}_{2}'.format(folder_path[3:], l, img_name)
        print(namespace)
        vp1_av, vp2_av = line_v.point_gen()
        hp1_av, hp2_av = line_h.point_gen()
    if debias == 'z':
        zv_err_av, zv_err_ap = line_v.debias()
        zh_err_av, zh_err_ap = line_h.debias()
    else:
        zv_err_av, zv_err_ap = line_v.debias_y()
        zh_err_av, zh_err_ap = line_h.debias_y()

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
        fpath1 = os.path.join(ROOT_DIR, namespace[:-4]+'_lines'+saveopt+namespace[-4:])
        fig1.savefig(fpath1)
        print("Saved {}".format(fpath1))
        fig2.savefig(os.path.join(ROOT_DIR, namespace[:-4] + '_errs' + saveopt+ namespace[-4:]))
        plt.close("all")
    return HoughLine.intersect(line_h, line_v)


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
        fig1.savefig(namespace[:-4] + '_lines' + saveopt + namespace[-4:])
        fig2.savefig(namespace[:-4] + '_errs' + saveopt + namespace[-4:])
        plt.close('all')
    if stat:
        return x, y, (line_h.opti, line_v.opti) + hseq2 + vseq2 + totseq2 + hseq + vseq + totseq
    return x, y

