import cv2, os
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


def model_contrast_plot(titres, variations, mregx, mregy, path, saveopt):
    font = {'family': 'normal',
            'weight': 'bold',
            'size': 22}
    ls = np.arange(titres)
    matplotlib.rc('font', **font)
    fig = plt.figure(figsize=(200, 100))
    plt.style.use(['seaborn-dark', 'seaborn-paper'])
    plt.subplot(311)
    plt.plot(ls, variations, color='#FA5B3D', marker='o', markeredgewidth=1, linestyle='-', linewidth=2)
    plt.xticks(ls, titres)
    plt.ylabel('deviations(+/-px)')
    plt.xlabel('loss functions')
    plt.title('Convergence By {}'.format(saveopt))
    plt.subplot(312)
    plt.plot(ls, mregx, color='#FA5B3D', marker='o', markeredgewidth=1, linestyle='-', linewidth=2)
    plt.xticks(ls, titres)
    plt.ylabel('Max Reg Error X')
    plt.xlabel('loss functions')
    plt.title('Max Reg Error X')
    plt.subplot(313)
    plt.plot(ls, mregy, color='#FA5B3D', marker='o', markeredgewidth=1, linestyle='-', linewidth=2)
    plt.xticks(ls, titres)
    plt.ylabel('Max Reg Error Y')
    plt.xlabel('loss functions')
    plt.title('Max Reg Error Y')
    plt.show()
    fig.savefig(os.path.join(path, 'Convergence By {}.png'.format(saveopt)))
    print(titres[np.argmin(variations)])
    plt.close('all')


def convergence_test_final(folder, ns, visual=False, tt='d', saveopt=""):
    # tt: a, l, k, g
    convergence = {}
    titres = []
    variations = []
    startNP = 59
    startP = 80
    endP = 193
    meas = os.path.join(ROOT_DIR, 'meas/')
    fwrite = open(os.path.join(meas, 'convergence_{}{}.csv'.format(tt, saveopt)), 'w')
    cwriter = csv.writer(fwrite)
    cwriter.writerow(['Image Number', 'Center X', 'Center Y', 'StdDev Horizontal', 'Std Dev Vertical'])
    loss_types = ['huber', 'soft_l1', 'arctan', 'linear', 'cauchy']
    krange = np.arange(-1, 8, 2)
    grange = np.arange(1, 20, 2)
    sl, sk, sg = tt == 'a' or tt == 'l', tt == 'a' or tt == 'k', tt == 'a' or tt == 'g'
    swt = lambda ran, cond: ran if cond else range(0)
    ls = range(np.prod([len(loss_types) if sl else 1, len(krange) if sk else 1, len(grange) if sg else 1]))
    print(ls)
    catchx, catchy = [], []
    if tt == 'a':
        for l in loss_types:
            for ks in krange:
                for gs in grange:
                    catch = [[], []]
                    com = [l, 'Sobel:{}'.format(ks), 'Gauss:{}'.format(gs)]
                    cwriter.writerow(com)
                    # Consistency Cycled
                    pv = 0
                    pvs = 0
                    rcount = 0
                    cvx = np.zeros(4)
                    cvy = np.zeros(4)
                    cond = visual and l == 'soft_l1' and ks == -1 and gs == 7
                    # print(g)
                    for i in range(startNP, startP):
                        img_name = ns.format(i)
                        fpath = os.path.join(ROOT_DIR, folder)
                        imgfile = "%s_{0}.png" % img_name
                        # FOR NULL ROW OR COLUMN, DO NOT COUNT THE STDDEV
                        try:
                            #x, y = center_detect(fpath + imgfile, 5, m=m) TODO: RESOLVE THE CHANGE
                            x, y = center_detect(fpath, imgfile.format(1), l=l, gk=gs, ks=ks, visual=cond, catch=catch)
                            #                             # PUT IN CSV
                            cwriter.writerow([str(i), str(x), str(y)])
                            # CONVERGENCE
                        except AttributeError:
                            print('No {0}'.format(fpath + imgfile))
                            pass
                    for i in range(startP, endP):
                        img_name = ns.format(i)
                        fpath = os.path.join(ROOT_DIR, folder)
                        imgfile = "%s_{0}.png" % img_name
                        # FOR NULL ROW OR COLUMN, DO NOT COUNT THE STDDEV
                        try:
                            #x, y = center_detect(fpath + imgfile, 5, m=m) TODO: RESOLVE THE CHANGE
                            x, y = center_detect(fpath, imgfile.format(1), l=l, gk=gs, ks=ks, visual=cond, catch=catch)
                            # PUT IN CSV
                            cwriter.writerow([str(i), str(x), str(y)])
                            # CONVERGENCE
                            # Record x, y, check rcount, refresh CONSISTENCY
                            cvx[rcount] = x
                            cvy[rcount] = y
                            rcount += 1
                            if rcount == 4:
                                pv += np.var(cvx) + np.var(cvy)
                                pvs += 1
                                rcount = 0
                                cvx = np.zeros(4)
                                cvy = np.zeros(4)
                        except AttributeError:
                            print('No {0}'.format(fpath + imgfile))
                            pass
                            # print(str(i), val)
                    msepv = sqrt(pv / pvs)
                    catchx.append(max(catch[0]))
                    catchy.append(max(catch[1]))
                    cvg = {'PicConsistency': msepv, 'Max_Reg_ErrX': catchx[-1], 'Max_Reg_ErrY': catchy[-1]}
                    cwriter.writerow(cvg.items())
                    convergence[com] = cvg
                    variations.append(msepv)
                    titres.append(com)
        print(convergence)
        fwrite.close()
        model_contrast_plot(titres, variations, catchx, catchy, meas, saveopt)

    elif tt == 'l':
        for k, l in enumerate(loss_types):
            catch = [[],[]]
            com = [l]
            cwriter.writerow(com)
            # Consistency Cycled
            pv = 0
            pvs = 0
            rcount = 0
            cvx = np.zeros(4)
            cvy = np.zeros(4)
            kl = int(not visual) + k
            # print(kl, k)
            # print(g)
            for i in range(startNP, startP):
                img_name = ns.format(i)
                fpath = os.path.join(ROOT_DIR, folder)
                imgfile = "%s_{0}.png" % img_name
                # FOR NULL ROW OR COLUMN, DO NOT COUNT THE STDDEV
                try:
                    # x, y = center_detect(fpath + imgfile, 5, m=m) TODO: RESOLVE THE CHANGE
                    x, y = center_detect(fpath, imgfile.format(1), l=l, visual=True if kl == 0 else False, catch=catch)
                    # PUT IN CSV
                    cwriter.writerow([str(i), str(x), str(y)])
                    # CONVERGENCE
                except AttributeError:
                    print('No {0}'.format(fpath + imgfile))
                    pass
            for i in range(startP, endP):
                img_name = ns.format(i)
                fpath = os.path.join(ROOT_DIR, folder)
                imgfile = "%s_{0}.png" % img_name
                # FOR NULL ROW OR COLUMN, DO NOT COUNT THE STDDEV
                try:
                    # x, y = center_detect(fpath + imgfile, 5, m=m) TODO: RESOLVE THE CHANGE
                    x, y = center_detect(fpath, imgfile.format(1), l=l, visual=True if kl == 0 else False, catch=catch)
                    # PUT IN CSV
                    cwriter.writerow([str(i), str(x), str(y)])
                    # CONVERGENCE
                    # Record x, y, check rcount, refresh CONSISTENCY
                    cvx[rcount] = x
                    cvy[rcount] = y
                    rcount += 1
                    if rcount == 4:
                        pv += np.var(cvx) + np.var(cvy)
                        pvs += 1
                        rcount = 0
                        cvx = np.zeros(4)
                        cvy = np.zeros(4)
                except AttributeError:
                    print('No {0}'.format(fpath + imgfile))
                    pass
                    # print(str(i), val)
            msepv = sqrt(pv / pvs)
            catchx.append(max(catch[0]))
            catchy.append(max(catch[1]))
            cvg = {'PicConsistency': msepv, 'Max_Reg_ErrX': catchx[-1], 'Max_Reg_ErrY': catchy[-1]}
            cwriter.writerow(cvg.items())
            convergence[l] = cvg
            variations.append(msepv)
            titres.append(com)
        print(convergence)
        fwrite.close()
        model_contrast_plot(titres, variations, catchx, catchy, meas, saveopt)

    elif tt == 'd':
        norms = ['NO', 'INPLACE', 'OUTPLACE']
        for d in ['y', 'z', 'old']:
            for n in [0, 1, 2]:
                catch = [[], []]
                com = ["debias_with_{}".format(d), norms[n]]
                cwriter.writerow(com)
                # Consistency Cycled
                pv = 0
                pvs = 0
                rcount = 0
                cvx = np.zeros(4)
                cvy = np.zeros(4)
                cond = visual
                # print(g)
                for i in range(startNP, startP):
                    img_name = ns.format(i)
                    fpath = os.path.join(ROOT_DIR, folder)
                    imgfile = "%s_{0}.png" % img_name
                    # FOR NULL ROW OR COLUMN, DO NOT COUNT THE STDDEV
                    try:
                        #x, y = center_detect(fpath + imgfile, 5, m=m) TODO: RESOLVE THE CHANGE
                        print(imgfile.format(1))
                        x, y = center_detect(fpath, imgfile.format(1), debias=d, norm=n, visual=cond, catch=catch,
                                             saveopt="{}_{}_{}".format(saveopt, d, n))
                        #                             # PUT IN CSV
                        cwriter.writerow([str(i), str(x), str(y)])
                        # CONVERGENCE
                    except AttributeError:
                        print('No {0}'.format(fpath + imgfile))
                        pass
                for i in range(startP, endP):
                    img_name = ns.format(i)
                    fpath = os.path.join(ROOT_DIR, folder)
                    imgfile = "%s_{0}.png" % img_name
                    # FOR NULL ROW OR COLUMN, DO NOT COUNT THE STDDEV
                    try:
                        #x, y = center_detect(fpath + imgfile, 5, m=m) TODO: RESOLVE THE CHANGE
                        x, y = center_detect(fpath, imgfile.format(1), debias=d, norm=n, visual=cond,
                                             catch=catch, saveopt="{}_{}".format(saveopt,d))
                        # PUT IN CSV
                        cwriter.writerow([str(i), str(x), str(y)])
                        # CONVERGENCE
                        # Record x, y, check rcount, refresh CONSISTENCY
                        cvx[rcount] = x
                        cvy[rcount] = y
                        rcount += 1
                        if rcount == 4:
                            pv += np.var(cvx) + np.var(cvy)
                            pvs += 1
                            rcount = 0
                            cvx = np.zeros(4)
                            cvy = np.zeros(4)
                    except AttributeError:
                        print('No {0}'.format(fpath + imgfile))
                        pass
                        # print(str(i), val)
                msepv = sqrt(pv / pvs)
                catchx.append(max(catch[0]))
                catchy.append(max(catch[1]))
                cvg = {'PicConsistency': msepv, 'Max_Reg_ErrX': catchx[-1], 'Max_Reg_ErrY': catchy[-1]}
                cwriter.writerow(cvg.items())
                convergence[tuple(com)] = cvg
                variations.append(msepv)
                titres.append(com)
        print(convergence)
        fwrite.close()
        model_contrast_plot(titres, variations, catchx, catchy, meas, saveopt)


def path_prefix_free(path, symbol='/'):
    if path[-len(symbol):] == symbol:
        return path[path.rfind(symbol,0, -len(symbol))+len(symbol):-len(symbol)]
    else:
        return path[path.rfind(symbol)+len(symbol):]


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
            cwriter.writerow([id, res[0], res[1]] + list(res[2]) if res[2] else [])
        except AttributeError:
            print('No {0}'.format(imgseq.format(series[i])))
            i += 2 if invar else 1
    fwrite.close()