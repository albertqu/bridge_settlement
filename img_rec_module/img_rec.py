import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


def identity(img):
    return img


def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def grad_detect(img, n, ori, func):
    #img = np.array(img)
    if ori == 'h':
        grad = [func(img, n, i+1) - func(img, n, i) for i in range(img.shape[1]-1)]
    else:
        grad = [func(img, i+1, n) - func(img, i, n) for i in range(img.shape[0]-1)]
    return grad


def abs_detect(img, n, ori, func):
    if ori == 'h':
        return [func(img, n, i) for i in range(img.shape[1])]
    else:
        return [func(img, i, n) for i in range(img.shape[0])]


def visualize_data(folder, oname, sample, orient, func, data_p=grad_detect, img_p=identity):
    dirs = os.listdir(folder)
    for f in dirs:
        try:
            img = img_p(cv2.imread(folder + f))
            interval = int(img.shape[1 if orient == 'v' else 0] / sample)
            for j in range(sample):
                n = j * interval
                data = data_p(img, n, orient, func)
                plt_save_data(data, oname + 'data_{0}{1}_'.format(orient, n) + f)
        except:
            print(f)


def max_entry(img):
    max_e = 0
    for row in img:
        temp = max(row)
        if temp > max_e:
            max_e = temp


def record_file(vals, fname):
    f = open(fname, 'w')
    for v in vals:
        f.write(str(v) + '\n')
    f.close()


def rel_lumin(img, r, c):
    b = img.item(r, c, 0)
    g = img.item(r, c, 1)
    r = img.item(r, c, 2)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def max_method(img):
    index = []
    s = 0
    for i in range(img.shape[1]):
        val = np.argmax(img[:, i])
        index.append(val)
        s += val
    return s / len(index)


def plot_show_data(data):
    plt.figure(figsize=(16, 8))
    #plt.xticks(range(0, img.shape[0], 20), range(0, img.shape[0], 20))
    plt.plot(data, 'b-')
    plt.show()
    plt.close()


def plt_save_data(data, fname):
    plt.figure(figsize=(16, 8))
    plt.plot(data, 'b-')
    plt.savefig(fname)
    plt.close()


def centroid(img, func):
    r = img.shape[0]
    c = img.shape[1]
    aggr_i = 0
    aggr_j = 0
    aggr = 0
    for i in range(r):
        for j in range(c):
            lum = func(img, i, j)
            aggr_i += lum * i
            aggr_j += lum * j
            aggr += lum
    dim = r * c
    return aggr_i / aggr, aggr_j / aggr
    # return aggr_i / dim, aggr_j / dim


def centroid_test(fname, n, trials, func, img_p=identity):
    for i in range(1, n+1):
        for j in range(1, trials+1):
            cf = fname.format(i, j)
            print(cf)
            im = img_p(cv2.imread(cf))
            print(centroid(im, func))

gray_p = lambda img, r, c: img.item(r, c)

if __name__ == '__main__':
    centroid_test("../testpic/img_{0}_{1}.png", 25, 3, gray_p, img_p=grayscale)
    # centroid_test("../testpic/img_{0}_{1}.png", 25, 3, rel_lumin)

    """img = cv2.imread("../testpic/png_1_1.png")
    plot_show_data(abs_detect(img, 0, 'v', rel_lumin))"""

    """img = cv2.imread("../testpic/png_1_1.png")
    print(img.shape)
    cv2.imshow('all', img)
    gs = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('gray', gs)
    print(sum(centers) / len(centers))
    print(max_method(gs))
    record_file(centers, 'centers.txt')
    
    while True:
        prompt = input("type in an index to draw or q to quit")
        if prompt == 'q':
            break
        else:
            plot_i(int(prompt))
    
    centers = []
    for i in range(gs.shape[1]):
        grad = grad_detect(gs, i, 'v', rel_lumin)
        max_v = max(range(len(grad)), key=lambda i: grad[i])
        min_v = min(range(len(grad)), key=lambda i: grad[i])
        centers.append((max_v + min_v) / 2)"""

    #visualize_data("../testpic/", "../absdataplot/", 10, 'h', rel_lumin, data_p=abs_detect)

