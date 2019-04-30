import matplotlib.pyplot as plt
import numpy as np
import cv2
import matplotlib
from analytic_geom import HoughLine



""" ===================================
    ============ PLOTTING =============
    ============= HELPER ==============
    =================================== """


def quick_plot(data, xs=None):
    plt.figure()
    if xs:
        plt.plot(xs, data, 'bo-')
    else:
        plt.plot(data, 'bo-')
    plt.show()
    plt.close()


def compare_images(*args, ilist=None, color_map=None, suptitle=None):
    # Takes in a sequence of IMG(Gray) and TITLE pairs and plot them side by side
    if ilist:
        args = ilist
    graph_amount = len(args)
    row = int(np.sqrt(graph_amount))
    col = np.ceil(float(graph_amount) / row)
    plt.figure(figsize=(16, 8))
    if suptitle:
        plt.suptitle(suptitle, fontsize=14)
    for i, pair in enumerate(args):
        plt.subplot(row, col, i+1)
        if color_map:
            plt.imshow(pair[0], cmap=color_map)
        else:
            plt.imshow(pair[0])
        plt.xticks([]), plt.yticks([])
        plt.title(pair[1])
    plt.show()
    plt.close()


def compare_data_plots(*args, ilist=None, suptitle=None, symbol='b-'):
    # Takes in a sequence of tuples with data, xs, and title pairs
    if ilist:
        args = ilist
    graph_amount = len(args)
    row = int(np.sqrt(graph_amount))
    col = np.ceil(float(graph_amount) / row)
    plt.figure(figsize=(16, 8))
    if suptitle:
        plt.suptitle(suptitle, fontsize=14)
    for i, pair in enumerate(args):
        plt.subplot(row, col, i + 1)
        if len(pair) == 2:
            plt.plot(pair[0], symbol)
            plt.title(pair[1])
        else:
            plt.plot(pair[1], pair[0], symbol)
            plt.title(pair[2])
    plt.show()


def line_graph_contrast(img, xs, ys):
    line = HoughLine(x=xs, data=ys)
    #print(xs)
    x0 = line._c * line._r
    y0 = line._s * line._r

    print('x0:{0}, y0:{1}, cos:{2}, sin:{3}, rho:{4}, theta:{5}'.format(x0, y0, line._c, line._s, line._r, degrees(line._t)))
    x1 = int(x0 + 1000 * (-line._s))
    y1 = int(y0 + 1000 * (line._c))
    x2 = int(x0 - 1000 * (-line._s))
    y2 = int(y0 - 1000 * (line._c))
    p1, p2 = (x1, y1), (x2, y2)
    cv2.line(img, p1, p2, (255, 0, 0), 1)
    return p1, p2, line


def plot_img(img, cmap='coolwarm'):
    img = cv2.convertScaleAbs(img)
    r, c = img.shape
    rs = np.arange(0, r)
    cs = np.arange(0, c)
    xs, ys = np.meshgrid(cs, rs)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(xs, ys, img, cmap=cmap, linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()
    plt.close()


def visualize_centers(img, centers_v, centers_h, ax):
    xsv = [c[1] for c in centers_v]
    ysv = [c[0] for c in centers_v]
    xsh = [c[1] for c in centers_h]
    ysh = [c[0] for c in centers_h]
    ax.imshow(img, cmap='gray')
    ax.scatter(xsv, ysv, facecolor='blue', linewidths=1)
    ax.scatter(xsh, ysh, facecolor='red', linewidths=1)


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

