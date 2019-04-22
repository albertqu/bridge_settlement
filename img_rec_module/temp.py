import cv2
import numpy as np
import csv
from scipy.optimize import curve_fit
from scipy.optimize import least_squares
import heapq
from decimal import Decimal
import matplotlib
from math import sqrt, acos, asin, degrees, sin, cos
from img_rec_module.fitting import leastSquares, improvedCost
import matplotlib.pyplot as plt


""" ===================================
========== ANALYTIC GEOMETRY ==========
======================================= """

class HoughLine:
    """ Houghline class that fits a analytical line to the data, r = xcos(\theta)+ysin(\theta)

    loss: loss kernel during least square
    theta, rho: theta angle and distance to origin for the houghline

    """
    def __init__(self, theta=0, rho=0, x=None, data=None, loss='soft_l1'):
        if data is not None and x is not None:
            self.x = x
            self.data = data
            self.loss = loss
            self.reg(x, data)
        else:
            self._r = rho
            self._t = theta
            self._s = sin(theta)
            self._c = cos(theta)
        self.debias = self.debias_old
        self.pred = None

    def reg(self, x, data):
        """ Given sample points x and the labels data, find a best fit Houghline with self.loss and leastSquares
        """
        x1, x2 = np.mean(x[:int(len(x) / 2)]), np.mean(x[int(len(x) / 2):])
        y1, y2 = np.mean(data[:int(len(data) / 2)]), np.mean(data[int(len(data) / 2):])
        theta0 = theta_pred(x1, y1, x2, y2)
        p0 = [theta0, np.mean(x) * np.cos(theta0) + np.mean(data) * np.sin(theta0)]
        res = least_squares(HoughLine.get_err, p0, loss=self.loss, f_scale=3, args=(x, data))
        self.opti = res.optimality
        angle = normalize_angle(res.x[0])
        self._t = angle
        self._r = res.x[1]
        self._s = sin(angle)
        self._c = cos(angle)

    @staticmethod
    def get_err(vars, xs, ys):
        return xs * np.cos(vars[0]) + ys * np.sin(vars[0]) - vars[1]

    def fit_x(self, x):
        # print(self._r, self._t)
        fits = (self._r - x * self._c) / self._s if self._s != 0 else np.nan
        self.pred = fits
        return fits

    def debias_old(self, thres=1.0):
        """ @:returns tuple with a) zerr_before: regression error before debias
         b) zerr_after: regression error after debias"""
        x, y = self.x, self.data
        zero_hat_before = x * self._c + y * self._s - self._r
        zerr_before = np.square(zero_hat_before)
        conds = (zerr_before - np.mean(zerr_before)) / np.std(zerr_before) <= thres
        new_x, new_y = x[conds], y[conds]
        self.x, self.data = new_x, new_y
        self.reg(new_x, new_y)
        zero_hat_after = new_x * self._c + new_y * self._s - self._r
        zerr_after = np.square(zero_hat_after)
        return zerr_before, zerr_after

    def debias_z(self, thres=1.0):
        """ @:returns tuple with a) zerr_before: regression error before debias
         b) zerr_after: regression error after debias"""
        x, y = self.x, self.data
        zero_hat_before = x * self._c + y * self._s - self._r
        zerr_before = np.square(zero_hat_before)
        conds = np.abs(zero_hat_before) / np.sqrt(np.sum(zerr_before)/len(zerr_before)) <= thres
        new_x, new_y = x[conds], y[conds]
        self.x, self.data = new_x, new_y
        self.reg(new_x, new_y)
        zero_hat_after = new_x * self._c + new_y * self._s - self._r
        zerr_after = np.square(zero_hat_after)
        return zerr_before, zerr_after

    def debias_y(self, thres=1.0):
        """ @:returns tuple with a) zerr_before: regression error before debias
         b) zerr_after: regression error after debias
         Uses Standard estimate error to filter out bad regressions.
         """
        x, y = self.x, self.data
        y_hat_before = self.fit_x(x)
        yerr_before = y_hat_before - y
        yerrb2 = yerr_before ** 2
        se_reg = np.sqrt(np.sum(yerrb2) / len(yerr_before))
        conds = np.abs(yerr_before) / se_reg  <= thres
        new_x, new_y = x[conds], y[conds]
        self.x, self.data = new_x, new_y
        self.reg(new_x, new_y)
        y_hat_after = self.fit_x(new_x)
        yerr_after = np.square(y_hat_after - new_y)
        return yerrb2, yerr_after

    def point_gen(self):
        x0 = self._c * self._r
        y0 = self._s * self._r
        x1 = int(x0 + 1000 * (-self._s))
        y1 = int(y0 + 1000 * (self._c))
        x2 = int(x0 - 1000 * (-self._s))
        y2 = int(y0 - 1000 * (self._c))
        return (x1, y1), (x2, y2)

    def __str__(self):
        return 'hough line with cos:{0}, sin:{1}, rho:{2}, theta:{3}'.format(self._c, self._s,
                                                                             self._r, degrees(self._t))

    @staticmethod
    def intersect(l1, l2):
        if l1._s == 0 and l2._s == 0:
            return float('inf'), float('inf')
        if l1._s == 0:
            return l1._r, l2.fit_x(l1._r)
        elif l2._s == 0:
            return l2._r, l1.fit_x(l2._r)
        x = (l2._r / l2._s - l1._r / l1._s) / (l2._c / l2._s - l1._c / l1._s)
        y = l1.fit_x(x)
        return x, y


# QC: QUARTER_CYCLE, HC: HALF_CYCLE, TQC: THIRD_QUARTER_CYCLE, FC: FULL_CYCLE
QC = np.pi / 2
HC = np.pi
TQC = 3 * np.pi / 2
FC = 2 * np.pi


def normalize_angle(angle):
    res = angle - (angle // FC) * FC
    return res


def theta_pred(x1, y1, x2, y2):
    dx = x2 - x1
    dy = y2 - y1
    angle = sin_angle_from_points(dx, dy)
    if dx * dy <= 0:
        return np.pi / 2 - angle
    else:
        root = root_finding(x1, x2, y1, y2)
        if root < 0:
            return np.pi / 2 + angle
        else:
            return angle + 3 * np.pi / 2


def sin_angle_from_points(dx, dy):
    return asin(abs(dy) / sqrt(dx ** 2 + dy ** 2))


""" ===================================
=========== REGRESSION UTILS ==========
======================================= """


def root_finding(x1, x2, y1, y2):
    """Given two points on a line, finds its zero crossing root."""
    return - y1 * (x2 - x1) / (y2 - y1) + x1


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


""" =======================================
============ CENTER DETECTION  ============
=========================================== """

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
        cv2.line(imgr, hp1_ap, hp2_ap, (0, 0, 255), 1)
        cv2.line(imgr, vp1_ap, vp2_ap, (0, 0, 255), 1)

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
                       visual=False, catch=None, stat=False,saveopt=""):
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
        catch[1].append(sqrt(np.mean(zh_err_ap)))
        catch[0].append(sqrt(np.mean(zv_err_ap)))

    if debias == 'y':
        C1, C2 = np.corrcoef(line_h.data, line_h.pred)[1, 0], np.corrcoef(line_v.data, line_v.pred)[1, 0]

        metrics = r2_sqe(line_h.data, zh_err_ap), r2_sqe(line_v.data, zv_err_ap), C1, C2, C1 ** 2, C2 ** 2

        show_metrics = "Hor R2: {} Ver R2: {}, C_Hor, {}, C_Ver: {}, Hor_C2: {}, Ver_C2: {}".format(*metrics)
        print(show_metrics)

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

    if visual:
        xv_av, xv_ap = np.arange(len(zv_err_av)), np.arange(len(zv_err_ap))
        xh_av, xh_ap = np.arange(len(zh_err_av)), np.arange(len(zh_err_ap))
        hp1_ap, hp2_ap = line_h.point_gen()
        vp1_ap, vp2_ap = line_v.point_gen()
        cenhs = list(zip(line_h.data, line_h.x))
        cenvs = list(zip(line_v.data, line_v.x))
        cv2.line(imgr, hp1_ap, hp2_ap, (0, 0, 255), 1)
        cv2.line(imgr, vp1_ap, vp2_ap, (0, 0, 255), 1)

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
        if debias == 'y':
            fig2.suptitle(show_metrics)
            fig1.suptitle(show_metrics)
        fig1.savefig(namespace[:-4]+'_lines'+saveopt+namespace[-4:])
        fig2.savefig(namespace[:-4] + '_errs' + saveopt+namespace[-4:])
        plt.close('all')
    x, y = HoughLine.intersect(line_h, line_v)
    if x >= xlim or x < 0 or y < 0 or y >= ylim:
        return -1, -1, None
    if stat:
        return x, y, (line_h.opti, line_v.opti)+hseq2+vseq2+totseq2+hseq+vseq+totseq
    return x, y


def log_output(inpath, outpath, ns, series, invar=False, visual=False, saveopt=""):
    """ Runs center_detect with debug mode for all images in inpath+ns.format(irange), and output the test results
    in outpath/meas
    ns: str, *{}*.png, e.g. img_{}_1.png
    """
    imgseq = os.path.join(inpath, ns)
    fn = lambda p, n: p[p.find(n) + len(n) + 1:]
    fn2 = lambda segs: segs[-1] if segs[-1] else segs[-2]
    fwrite = open(os.path.join(outpath, 'meas/centerLog_{}_seriesLen{}_{}.csv'.format(path_prefix_free(inpath),
                                                                                      len(series), saveopt)), 'w+')
    cwriter = csv.writer(fwrite)
    cwriter.writerow(['img', 'x', 'y', 'HOPT', 'VOPT', 'SLD_H_0', 'SLD_H_1', 'MLD_H_0', 'MLD_H_1',
                      'SLD_V_0', 'SLD_V_1', 'MLD_V_0', 'MLD_V_1',
                      'SLD_A_0', 'SLD_A_1', 'MLD_A_0', 'MLD_A_1'
                      'EVS_h', 'r2_h', 'MAE_h', 'MSE_h', 'MEDAE_h',
                      'EVS_v', 'r2_v', 'MAE_v', 'MSE_v', 'MEDAE_v',
                      'EVS_t', 'r2_t', 'MAE_t', 'MSE_t', 'MEDAE_t'])
    i = 0
    while i < len(series):
        try:
            if invar:
                ambi, laser = cv2.imread(imgseq.format(series[i]), 0), cv2.imread(imgseq.format(series[i + 1]), 0)
                res_img = image_diff(laser, ambi)
                id = "{}_{}".format(series[i], series[i + 1])
                res = center_detect_base(res_img, inpath, ns.format(id), stat=True)
                i+=2
            else:
                res = center_detect(inpath, ns.format(series[i]), stat=True)
                id = str(series[i])
                i+=1
            cwriter.writerow([id, res[0], res[1]] + (list(res[2]) if res[2] else []))
        except AttributeError:
            print('No {0}'.format(imgseq.format(series[i])))
            i += 2 if invar else 1
    fwrite.close()

if __name__ == '__main__':
    #print(center_detect('../calib4/img_59_{0}.png', 5))
    #convergence_test_final('calib4/', 'img_{0}')
    #repertoire = expr2([10, 11, 15, 14, 18, 19, 22, 23])
    import os
    ROOT_DIR = '/Users/albertqu/Documents/7.Research/PEER Research/data'
    #repertoire = expr2_visual([14, 11, 13, 12], 'camera_tests', ns='{0}.png', visual=True, saveopt="_imgdiff_change")
    imgn = 'img_{0}_1.png'
    imgn_mul = 'img_{0}'
    test = os.path.join(ROOT_DIR, 'test1115/')
    calib4 = os.path.join(ROOT_DIR, 'calib4/')
    lab_series2 = os.path.join(ROOT_DIR, 'lab_series2')
    #convergence_test_final(calib4, imgn_mul, visual=True, tt='d', saveopt="_debias")
    print(calib4)
    #log_output(calib4, ROOT_DIR, imgn, np.arange(59, 193), saveopt='METRICS_z')
    log_output(lab_series2, ROOT_DIR, "{}.png", np.arange(1, 151), invar=True, saveopt='METRICS_z')