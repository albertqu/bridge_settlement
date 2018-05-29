import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

img = cv2.imread("../testpic/png_1_1.png")
print(img.shape)
img[:, :, 0] = 0
img[:,:, 1] = 0
cv2.imshow('all', img)
gs = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('gray', gs)


def visualize_data(folder, sample, orient, func):
    dirs = os.listdir(folder)
    for i, f in enumerate(dirs):
        img = cv2.imread(f)
        interval = int(img.shape[1 if orient == 'h' else 0] / sample)
        for j in range(sample):
            data = grad_detect(img, j, orient, func)
            plt_save_data(data, '../dataplot/data_' + orient + f)


def max_entry(img):
    max = 0
    for row in img:
        temp = max(row)
        if temp > max:
            max = temp


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


def grad_detect(img, n, ori, func):
    img = np.array(img)
    if ori == 'h':
        grad = [func(img, n, i+1) - func(img, n, i) for i in range(img.shape[1]-1)]
    else:
        grad = [func(img, i+1, n) - func(img, i, n) for i in range(img.shape[0]-1)]
    return grad

def max_method(img):
    index = []
    s = 0
    for i in range(img.shape[1]):
        val = np.argmax(img[:, i])
        index.append(val)
        s +=val
    return s / len(index)

def plot_show_data(data):
    plt.figure()
    plt.plot(data, 'b-')
    plt.show()
    plt.close()

def plt_save_data(data, fname):
    plt.figure()
    plt.plot(data, 'b-')
    plt.savefig(fname)
    plt.close()

"""print(sum(centers) / len(centers))
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

